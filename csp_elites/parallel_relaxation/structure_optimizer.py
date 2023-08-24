from __future__ import annotations

import copy
import time
from collections import defaultdict
from pprint import pprint
from typing import List

import numpy as np
from ase import Atoms
from chgnet.model import CHGNet
from chgnet.model.dynamics import TrajectoryObserver, \
    StructOptimizer, CHGNetCalculator
from mp_api.client import MPRester
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

from csp_elites.parallel_relaxation.fire import OverridenFire
from csp_elites.parallel_relaxation.unit_cell_filter import AtomsFilterForRelaxation


class MultiprocessOptimizer:
    def __init__(self, batch_size=10):
        self.overriden_optimizer = OverridenFire()
        self.atoms_filter = AtomsFilterForRelaxation()
        self.model = CHGNet.load()
        self.batch_size = batch_size
        self.timings = None
        self.reset_timings()
        self.timings_list = []

    def reset_timings(self):
        self.timings = {
            "list_comprehension": {

            },
            "for_loop": {},
            "vectorised": {},
            "chgnet": 0
        }

    def relax(self, list_of_atoms: List[Atoms], n_relaxation_steps: int, verbose: bool = False):
        self.reset_timings()
        all_relaxed = False

        v = None
        Nsteps = 0
        dt = np.full(len(list_of_atoms) , 0.1)
        a = np.full(len(list_of_atoms) , 0.1)
        n_relax_steps = np.zeros(len(list_of_atoms) )
        fmax_over_time = []


        trajectories = defaultdict(list)
        while not all_relaxed:
            forces, energies, stresses = self._evaluate_list_of_atoms(list_of_atoms)

            tic = time.time()
            original_cells = np.array([atoms.cell.array for atoms in list_of_atoms])

            forces, _ = self.atoms_filter.get_forces_exp_cell_filter(
                forces_from_chgnet=forces,
                stresses_from_chgnet=stresses,
                list_of_atoms=list_of_atoms,
                original_cells=original_cells,
                current_atom_cells=[atoms.cell.array for atoms in list_of_atoms],
                cell_factors=np.array([1] * len(list_of_atoms)),
            )
            self.timings["vectorised"][
                "expcell_filter_processing_some_for_loops"] = time.time() - tic
            if n_relaxation_steps == 0:
                all_relaxed = True
            else:
                fmax = np.max((forces ** 2).sum(axis=2), axis=1) ** 0.5
                fmax_over_time.append(fmax)
                if verbose:
                    print(Nsteps, energies * 24, fmax)

                tic = time.time()
                v, dt, n_relax_steps, a, dr = \
                    self.overriden_optimizer.step_override(forces, v, dt, n_relax_steps, a)
                self.timings["vectorised"]["relax1_get_position_change"] = time.time() - tic

                tic = time.time()
                positions = self.atoms_filter.get_positions(
                    original_cells,
                    [atoms.cell for atoms in list_of_atoms],
                    list_of_atoms,
                    np.array([1] * len(list_of_atoms)),
                )
                self.timings["vectorised"]["relax2_get_positions_to_update"] = time.time() - tic

                tic = time.time()
                list_of_atoms = self.atoms_filter.set_positions(original_cells,
                                                                list_of_atoms, np.array(positions + dr),
                                                                np.array([1] * len(list_of_atoms)))
                self.timings["vectorised"]["relax3_update_positions"] = time.time() - tic
                tic = time.time()
                converged_mask = self.overriden_optimizer.converged(forces, fmax)
                Nsteps += 1
                all_relaxed = self._end_relaxation(Nsteps, n_relaxation_steps, converged_mask)
                self.timings["vectorised"]["check_convergence"] = time.time() - tic
            trajectories["forces"].append(forces)
            trajectories["energies"].append(energies)
            trajectories["stresses"].append(stresses)

        tic = time.time()
        final_structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in list_of_atoms]
        reformated_output = []
        for i in range(len(final_structures)):
            reformated_output.append(
                {"final_structure": final_structures[i],
                 "trajectory": {
                     "energies":trajectories["energies"][-1][i],
                     "forces": trajectories["forces"][-1][i],
                     "stresses": trajectories["stresses"][-1][i],
                 }
                }
            )
        self.timings["for_loop"]["reformating_output"] = time.time() - tic
        # pprint(self.timings)
        self.timings_list.append(copy.deepcopy(self.timings))
        return reformated_output, list_of_atoms

    def _evaluate_list_of_atoms(self, list_of_atoms: List[Atoms]):
        tic = time.time()
        if isinstance(list_of_atoms[0], Atoms):
            list_of_structures = [AseAtomsAdaptor.get_structure(list_of_atoms[i]) for i in range(len(list_of_atoms))]
        elif isinstance(list_of_atoms[0], Structure):
            list_of_structures = list_of_atoms
        self.timings["list_comprehension"]["convert_atoms_to_structures"] = time.time() - tic

        hotfix_graphs = False
        tic = time.time()
        graphs = [self.model.graph_converter(struct, on_isolated_atoms="warn") for struct in list_of_structures]
        self.timings["list_comprehension"]["convert_structures_to_graphs"] = time.time() - tic

        tic = time.time()
        if None in graphs:
            indices_to_pop = []
            print("isolated atomssss")
            print(f"graphs starting length {len(graphs)}")
            hotfix_graphs = True
            indices_to_update = []
            for i in range(len(graphs)):
                print(i)
                if graphs[i] is None:
                    indices_to_update.append(i)
                    indices_to_pop.append(i)
            for j in indices_to_pop:
                graphs.pop(j)

            print(f"graphs end length {len(graphs)}")

        self.timings["for_loop"]["check_graph_for_isolated"] = time.time() - tic

        tic = time.time()
        predictions = self.model.predict_graph(
            graphs,
            task="efs",
            return_atom_feas=False,
            return_crystal_feas=False,
            batch_size=self.batch_size,
        )
        self.timings["chgnet"] = (time.time() - tic) + self.timings["list_comprehension"]["convert_atoms_to_structures"]

        # predictions = self.model.predict_structure(list_of_structures, batch_size=10)
        if isinstance(predictions, dict):
            predictions = [predictions]

        tic = time.time()
        forces = np.array([pred["f"] for pred in predictions])
        energies = np.array([pred["e"] for pred in predictions])
        stresses = np.array([pred["s"] for pred in predictions])
        self.timings["list_comprehension"]["get_results_from_chgnet_predictions"] = time.time() - tic

        tic = time.time()
        if hotfix_graphs:
            print("hotfix graph")
            # todo: make this dynamic
            for i in indices_to_update:
                forces = np.insert(forces, i, np.full((24,3), 100), axis=0)
                energies = np.insert(energies, i, 10000)
                stresses = np.insert(stresses, i, np.full((3,3), 100), axis=0)
        self.timings["for_loop"]["update_predictions_if_graph_hotfix"] = time.time() - tic

        return forces, energies, stresses

    def _update_trajectories(self, trajectories: List[TrajectoryObserver], forces, energies, stresses) -> List[TrajectoryObserver]:
        for i in range(len(trajectories)):
            trajectories[i].energies.append(energies[i])
            trajectories[i].forces.append(forces)
            trajectories[i].stresses.append(stresses)
        return trajectories


    def _end_relaxation(self, nsteps: int, max_steps: int, forces_mask:np.ndarray):
        return (nsteps > max_steps) or forces_mask.all()


if __name__ == '__main__':

    n_relaxation_steps = 10
    number_of_individuals = 100
    batch_size = 10

    optimizer = MultiprocessOptimizer(batch_size=batch_size)
    optimizer_ref = StructOptimizer()


    with MPRester(api_key="4nB757V2Puue49BqPnP3bjRPksr4J9y0") as mpr:
        one_structure = mpr.get_structure_by_material_id("mp-1840", final=True)
    atoms_2 = AseAtomsAdaptor.get_atoms(one_structure)
    atoms_2.calc = CHGNetCalculator()
    atoms_2.rattle(0.1)

    list_of_atoms = [copy.deepcopy(atoms_2) for i in range(number_of_individuals)]

    if n_relaxation_steps != 0:
        time_ase = 0
        for i in tqdm(range(number_of_individuals)):
            ref_atoms = copy.deepcopy(atoms_2)
            tic = time.time()
            optimizer_ref.relax(ref_atoms, steps=n_relaxation_steps, verbose=False)
            time_ase += time.time() - tic
        print(f"{n_relaxation_steps} relaxation steps ase: {time_ase}")

    tic = time.time()

    relax_results, atoms_returned = optimizer.relax(list_of_atoms, n_relaxation_steps=n_relaxation_steps, verbose=False)
    print(f"Time for {number_of_individuals} individual relax {time.time() - tic}")


    structure = [copy.deepcopy(AseAtomsAdaptor.get_structure(atoms_2)) for _ in range(number_of_individuals)]
    tic = time.time()
    optimizer.model.predict_structure(structure, batch_size=batch_size)
    print(f"Time for chgnet object only {time.time() - tic}")

    tic = time.time()
    structures = [structure] if isinstance(structure, Structure) else structure
    graphs = [optimizer.model.graph_converter(struct) for struct in structures]

    optimizer.model.predict_graph(
        graphs,
        task="efs",
        return_atom_feas=False,
        return_crystal_feas=False,
        batch_size=batch_size,
    )
    print(f"Time for chgnet  only {time.time() - tic}")


    tic = time.time()
    optimizer._evaluate_list_of_atoms(structures)
    print(f"Time for evaluate fn {time.time() - tic}")


    # initialise new model
    model = CHGNet.load()
    structure = [copy.deepcopy(AseAtomsAdaptor.get_structure(atoms_2)) for _ in range(number_of_individuals)]
    tic = time.time()
    model.predict_structure(structure, batch_size=batch_size)
    print(f"Time for chgnet new model object only {time.time() - tic}")

    tic = time.time()
    structures = [structure] if isinstance(structure, Structure) else structure
    graphs = [model.graph_converter(struct) for struct in structures]

    model.predict_graph(
        graphs,
        task="efsm",
        return_atom_feas=False,
        return_crystal_feas=False,
        batch_size=batch_size,
    )
    print(f"Time for chgnet new model objectonly {time.time() - tic}")

    print()
