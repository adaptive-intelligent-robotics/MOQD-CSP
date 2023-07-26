from __future__ import annotations

import contextlib
import copy
import io
import sys
import time
import warnings
from typing import List, Tuple

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import ExpCellFilter
from ase.optimize import FIRE
from ase.optimize.optimize import Dynamics, Optimizer

from chgnet.model import StructOptimizer, CHGNet
from chgnet.model.dynamics import TrajectoryObserver
from matgl.ext.ase import Relaxer
from matplotlib import pyplot as plt
from mp_api.client import MPRester
from numba import jit, njit
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm


class MultiprocessOptimizer(StructOptimizer):
    def __init__(self, model: CHGNet | None = None,
        optimizer_class: Optimizer | str | None = "FIRE",
        use_device: str | None = None,
        stress_weight: float = 1 / 160.21766208,
        atoms: Atoms = None # not optional
                 ):
        super().__init__(model, optimizer_class, use_device, stress_weight)
        self.overriden_optimizer = OverridenFire(atoms)

    def relax(
            self,
            atoms: Structure | Atoms,
            fmax: float | None = 0.1,
            steps: int | None = 500,
            relax_cell: bool | None = False,
            save_path: str | None = None,
            trajectory_save_interval: int | None = 1,
            verbose: bool = True,
            **kwargs,
    ) -> dict[str, Structure | TrajectoryObserver]:
        """Relax the Structure/Atoms until maximum force is smaller than fmax.

        Args:
            atoms (Structure | Atoms): A Structure or Atoms object to relax.
            fmax (float | None): The maximum force tolerance for relaxation.
                Default = 0.1
            steps (int | None): The maximum number of steps for relaxation.
                Default = 500
            relax_cell (bool | None): Whether to relax the cell as well.
                Default = True
            save_path (str | None): The path to save the trajectory.
                Default = None
            trajectory_save_interval (int | None): Trajectory save interval.
                Default = 1
            verbose (bool): Whether to print the output of the ASE optimizer.
                Default = True
            **kwargs: Additional parameters for the optimizer.

        Returns:
            dict[str, Structure | TrajectoryObserver]:
                A dictionary with 'final_structure' and 'trajectory'.
        """
        if isinstance(atoms, Structure):
            atoms = AseAtomsAdaptor.get_atoms(atoms)

        atoms.calc = self.calculator  # assign model used to predict forces

        stream = sys.stdout if verbose else io.StringIO()
        with contextlib.redirect_stdout(stream):
            obs = TrajectoryObserver(atoms)
            if relax_cell:
                atoms = ExpCellFilter(atoms)
            optimizer = self.optimizer_class(atoms, **kwargs)
            optimizer.attach(obs, interval=trajectory_save_interval)
            optimizer.run(fmax=fmax, steps=steps)
            obs()

        if save_path is not None:
            obs.save(save_path)

        if isinstance(atoms, ExpCellFilter):
            atoms = atoms.atoms
        struct = AseAtomsAdaptor.get_structure(atoms)
        for key in struct.site_properties:
            struct.remove_site_property(property_name=key)
        struct.add_site_property(
            "magmom", [float(magmom) for magmom in atoms.get_magnetic_moments()]
        )
        return {"final_structure": struct, "trajectory": obs}


    def override_relax(self, list_of_atoms: List[Atoms], n_relaxation_steps: int):
        system_size = (len(list_of_atoms), (len(list_of_atoms[0])))
        forces, energies, stresses = self._evaluate_list_of_atoms(list_of_atoms)
        list_of_atoms = set_calculators_for_atoms_list(list_of_atoms, forces, energies, stresses)
        # list_of_atoms = [ExpCellFilter(atoms) for atoms in list_of_atoms]

        velocity, e_last, r_last, v_last = None, None, None, None
        dt, Nsteps, a = 0.1, 0, 0.1

        trajectories = [TrajectoryObserver(atoms) for atoms in list_of_atoms]

        for _ in tqdm(range(n_relaxation_steps)):
            v, e_last, r_last, v_last, dt, Nsteps, a, dr = \
                self.overriden_optimizer.step_override(system_size, forces, energies, velocity, e_last, r_last, v_last, dt,
                                             Nsteps, a)

            list_of_atoms = self._update_positions_post_relaxation_step(list_of_atoms, position_change=dr)
            forces, energies, stresses = self._evaluate_list_of_atoms(list_of_atoms)
            trajectories = self._update_trajectories(trajectories, forces, energies, stresses)

        # list_of_atoms = [atoms.atoms for atoms in list_of_atoms]

        final_structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in list_of_atoms]
        reformated_output = []
        for i in range(len(final_structures)):
            reformated_output.append(
                {"final_structure": final_structures[i],
                 "trajectory": trajectories[i],
                }
            )

        return reformated_output

    def _evaluate_list_of_atoms(self, list_of_atoms: List[Atoms]):
        list_of_structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in list_of_atoms]

        predictions = chgnet.predict_structure(list_of_structures, batch_size=len(list_of_atoms))
        if isinstance(predictions, dict):
            predictions = [predictions]

        forces = np.array([pred["f"] for pred in predictions])
        energies = np.array([pred["e"] for pred in predictions])
        stresses = np.array([pred["s"] for pred in predictions])
        return forces, energies, stresses

    def _set_atom_calulators(self, list_of_atoms: List[Atoms],
        forces: np.ndarray,
        energies: np.ndarray,
        stresses: np.ndarray
    ):
        for i in range(len(list_of_atoms)):
            calculator = SinglePointCalculator(
                list_of_atoms[i], energy=energies[i], forces=forces[i], stress=stresses[i],
            )
            list_of_atoms[i].calc = calculator
        return atoms

    def _update_positions_post_relaxation_step(self, list_of_atoms: List[Atoms], position_change: np.ndarray):
        for i in range(len(list_of_atoms)):
            r = list_of_atoms[i].get_positions()
            list_of_atoms[i].set_positions(r + position_change[i])
        return list_of_atoms

    def _update_trajectories(self, trajectories: List[TrajectoryObserver], forces, energies, stresses) -> List[TrajectoryObserver]:
        for i in range(len(trajectories)):
            trajectories[i].energies.append(energies[i])
            trajectories[i].forces.append(forces)
            trajectories[i].stresses.append(stresses)

        return trajectories

class OverridenFire(Optimizer):
    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 dt=0.1, maxstep=None, maxmove=None, dtmax=1.0, Nmin=5,
                 finc=1.1, fdec=0.5,
                 astart=0.1, fa=0.99, a=0.1, master=None, downhill_check=False,
                 position_reset_callback=None, force_consistent=None):
        Optimizer.__init__(self, atoms, restart, logfile, trajectory,
                       master, force_consistent=force_consistent)

        self.dt = dt

        self.Nsteps = 0

        if maxstep is not None:
            self.maxstep = maxstep
        elif maxmove is not None:
            self.maxstep = maxmove
            warnings.warn('maxmove is deprecated; please use maxstep',
                          np.VisibleDeprecationWarning)
        else:
            self.maxstep = self.defaults['maxstep']

        self.dtmax = dtmax
        self.Nmin = Nmin
        self.finc = finc
        self.fdec = fdec
        self.astart = astart
        self.fa = fa
        self.a = a
        self.downhill_check = downhill_check
        self.position_reset_callback = position_reset_callback

    # def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
    #              dt=0.1, maxstep=None, maxmove=None, dtmax=1.0, Nmin=5,
    #              finc=1.1, fdec=0.5,
    #              astart=0.1, fa=0.99, a=0.1, master=None, downhill_check=False,
    #              position_reset_callback=None, force_consistent=None):
    #     super().__init__(atoms, restart=None, logfile='-', trajectory=None,
    #              dt=0.1, maxstep=None, maxmove=None, dtmax=1.0, Nmin=5,
    #              finc=1.1, fdec=0.5,
    #              astart=0.1, fa=0.99, a=0.1, master=None, downhill_check=False,
    #              position_reset_callback=None, force_consistent=None)

    # @jit(nopython=True)
    def step_override(
        self,
        system_size: Tuple[int],
        f: np.ndarray,
        e: np.ndarray,
        v: np.ndarray,
        e_last: np.ndarray,
        r_last: np.ndarray,
        v_last: np.ndarray,
        dt: np.ndarray, 
        Nsteps: np.ndarray,
        a: np.ndarray,
    ):
        # atoms = self.atoms

        # if f is None:
        #     f = atoms.get_forces()

        if v is None:
            v = np.zeros((system_size[0], system_size[1], 3))
            # if downhill_check:
            #     e_last = atoms.get_potential_energy(
            #         force_consistent=force_consistent)
            #     r_last = atoms.get_positions().copy()
            #     v_last = v.copy()
        else:
            is_uphill = False
            # if downhill_check:
                # e = atoms.get_potential_energy(
                #     force_consistent=force_consistent)
                # # Check if the energy actually decreased
                # if e > e_last:
                #     # If not, reset to old positions...
                #     if position_reset_callback is not None:
                #         position_reset_callback(atoms, r_last, e,
                #                                      e_last)
                #     atoms.set_positions(r_last)
                #     is_uphill = True
                # e_last = atoms.get_potential_energy(
                #     force_consistent=force_consistent)
                # r_last = atoms.get_positions().copy()
                # v_last = v.copy()

            vf = np.vdot(f, v)
            if vf > 0.0 and not is_uphill:
                v = (1.0 - a) * v + a * f / np.sqrt(
                    np.vdot(f, f)) * np.sqrt(np.vdot(v, v))
                if Nsteps > self.Nmin:
                    dt = min(dt * self.finc, self.dtmax)
                    a *= self.fa
                Nsteps += 1
            else:
                v[:] *= 0.0
                a = self.astart
                dt *= self.fdec
                Nsteps = 0

        v += dt * f
        dr = dt * v
        normdr = np.sqrt(np.vdot(dr, dr))
        if normdr > self.maxstep:
            dr = self.maxstep * dr / normdr

        # self.dump((v, dt))
        return v, e_last, r_last, v_last, dt, Nsteps, a, dr

    def step(self, atoms, f=None):
        # atoms = self.atoms
        #
        # if f is None:
        #     f = atoms.get_forces()

        if self.v is None:
            self.v = np.zeros((len(atoms), 3))
            # if self.downhill_check:
            #     self.e_last = atoms.get_potential_energy(
            #         force_consistent=self.force_consistent)
            #     self.r_last = atoms.get_positions().copy()
            #     self.v_last = self.v.copy()
        else:
            is_uphill = False
            # if self.downhill_check:
            #     e = atoms.get_potential_energy(
            #         force_consistent=self.force_consistent)
            #     # Check if the energy actually decreased
            #     if e > self.e_last:
            #         # If not, reset to old positions...
            #         if self.position_reset_callback is not None:
            #             self.position_reset_callback(atoms, self.r_last, e,
            #                                          self.e_last)
            #         atoms.set_positions(self.r_last)
            #         is_uphill = True
            #     self.e_last = atoms.get_potential_energy(
            #         force_consistent=self.force_consistent)
            #     self.r_last = atoms.get_positions().copy()
            #     self.v_last = self.v.copy()

            vf = np.vdot(f, self.v)
            if vf > 0.0 and not is_uphill:
                self.v = (1.0 - self.a) * self.v + self.a * f / np.sqrt(
                    np.vdot(f, f)) * np.sqrt(np.vdot(self.v, self.v))
                if self.Nsteps > self.Nmin:
                    self.dt = min(self.dt * self.finc, self.dtmax)
                    self.a *= self.fa
                self.Nsteps += 1
            else:
                self.v[:] *= 0.0
                self.a = self.astart
                self.dt *= self.fdec
                self.Nsteps = 0

        self.v += self.dt * f
        dr = self.dt * self.v
        normdr = np.sqrt(np.vdot(dr, dr))
        if normdr > self.maxstep:
            dr = self.maxstep * dr / normdr
        r = atoms.get_positions()
        atoms.set_positions(r + dr)
        self.dump((self.v, self.dt))

def set_calculators_for_atoms_list(
    atoms: List[Atoms],
    forces: np.ndarray,
    energies: np.ndarray,
    stresses: np.ndarray
):
    for i in range(len(atoms)):
        calculator = SinglePointCalculator(
            atoms, energy=energies[i], forces=forces[i], stress=stresses[i],
        )
        atoms[i].calc = calculator
    return atoms

def evaluate_list_of_atoms(list_of_atoms: List[Atoms], chgnet):
    # ase_adapter = AseAtomsAdaptor
    tic = time.time()
    list_of_structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in list_of_atoms]
    print(f"convert structures: {time.time() - tic}")

    tic = time.time()
    predictions = chgnet.predict_structure(list_of_structures, batch_size=len(list_of_atoms))
    if isinstance(predictions, dict):
        predictions = [predictions]
    print(f"Get energy etc: {time.time() - tic}")
    # print(f"multiple evals {time.time() - tic}")
    forces = np.array([pred["f"] for pred in predictions])
    energies = np.array([pred["e"] for pred in predictions])
    stresses = np.array([pred["s"] for pred in predictions])

    return forces, energies, stresses

# def evaluate_with_m3gnet(list_of_atoms, relaxer):
#     _, relaxation_results = None
#     energy = float(
#         relaxation_results["trajectory"].energies[-1] / len(atoms.get_atomic_numbers()))
#     forces = relaxation_results["trajectory"].forces[-1]
#     stresses = relaxation_results["trajectory"].stresses[-1]
#     forces = np.array([pred["f"] for pred in predictions])
#     energies = np.array([pred["e"] for pred in predictions])
#     stresses = np.array([pred["s"] for pred in predictions])
#     pass

if __name__ == '__main__':
    chgnet = CHGNet.load()
    with MPRester(api_key="4nB757V2Puue49BqPnP3bjRPksr4J9y0") as mpr:
        one_structure = mpr.get_structure_by_material_id("mp-1341203", final=True)

    tic = time.time()
    prediction = chgnet.predict_structure(one_structure)
    # print(f"1 prediction {time.time() - tic}")

    atoms = AseAtomsAdaptor.get_atoms(structure=one_structure)
    atoms_2 = AseAtomsAdaptor.get_atoms(structure=one_structure)
    atoms_3 = AseAtomsAdaptor.get_atoms(structure=one_structure)

    calc = SinglePointCalculator(atoms, energy=prediction["e"], forces=prediction["f"],
                                 stress=prediction["s"])
    atoms.calc = calc
    atoms_2.calc = calc
    atoms_2.rattle(0.1)
    print(f"atoms 2 parameters {atoms_2.get_cell_lengths_and_angles()}")

    list_of_atoms = [copy.deepcopy(atoms_2),
                     copy.deepcopy(atoms_2),
                     copy.deepcopy(atoms_2),
                     copy.deepcopy(atoms_2),
                     copy.deepcopy(atoms_2),
                     copy.deepcopy(atoms_2),
                     copy.deepcopy(atoms_2),
                     copy.deepcopy(atoms_2),
                     copy.deepcopy(atoms_2),
                     copy.deepcopy(atoms_2),
        # AseAtomsAdaptor.get_atoms(structure=one_structure),
        #              AseAtomsAdaptor.get_atoms(structure=one_structure),
        #              AseAtomsAdaptor.get_atoms(structure=one_structure),
        #              AseAtomsAdaptor.get_atoms(structure=one_structure),
        #              AseAtomsAdaptor.get_atoms(structure=one_structure),
        #              AseAtomsAdaptor.get_atoms(structure=one_structure),
        #              AseAtomsAdaptor.get_atoms(structure=one_structure),
        #              AseAtomsAdaptor.get_atoms(structure=one_structure),
        #              AseAtomsAdaptor.get_atoms(structure=one_structure),
        #              AseAtomsAdaptor.get_atoms(structure=one_structure),
                     ]

    # print(f"list_of_atoms[0] parameters {list_of_atoms[0].get_cell_lengths_and_angles()}")
    # assert np.isclose(atoms_2.get_positions(), list_of_atoms[0].get_positions()).all()
    # structure_optimiser = MultiprocessOptimizer(atoms=atoms)
    # relaxation_time_chgnet = []
    # tic = time.time()
    # for i in range(len(list_of_atoms)):
    #     relaxation_results = structure_optimiser.relax(atoms_2, steps=2)
    #     relaxation_time_chgnet.append(time.time() - tic)
    # # print(atoms_2.get_cell_lengths_and_angles())
    # print(f"Chgnet time {sum(relaxation_time_chgnet)}")
    # toc = time.time()
    # results = structure_optimiser.override_relax(list_of_atoms, n_relaxation_steps=2)
    # print(time.time() - toc)
    #
    # print(np.sum(
    #     np.isclose(
    #         AseAtomsAdaptor.get_atoms(relaxation_results["final_structure"]).get_positions(),
    #         AseAtomsAdaptor.get_atoms(results[0]["final_structure"]).get_positions()
    #     )
    #     )
    # )
    # print(
    #     np.sum(
    #         AseAtomsAdaptor.get_atoms(relaxation_results["final_structure"]).get_positions()
    #         - AseAtomsAdaptor.get_atoms(results[0]["final_structure"]).get_positions()
    #     )
    # )
    print()

   #  # list_of_atoms = 10 * list_of_atoms
    forces, energies, stresses = evaluate_list_of_atoms(list_of_atoms, chgnet)
    list_of_atoms = set_calculators_for_atoms_list(list_of_atoms, forces, energies, stresses)



    fire = FIRE(atoms)
    overriden_fire = OverridenFire(atoms_2)

   # intialise all parameters that normally are attributes
    system_size = (len(list_of_atoms), (len(list_of_atoms[0])))
    f = forces
    e = energies
    v = None
    e_last = None
    r_last = None
    v_last = None
    dt = 0.1
    Nsteps = 0
    a = 0.1
    # relaxer = Relaxer()
    # relaxer.relax(steps=0)

    time_1_structure = []
    time_10_structures = []
    number_steps = 3
    for i in tqdm(range(number_steps)):
        # print(i)
        tic = time.time()
        fire.step(f=prediction["f"])

        time_1_structure.append(time.time() - tic)
        prediction = chgnet.predict_structure(AseAtomsAdaptor.get_structure(atoms))
        # print(f" 1 step fire {time.time() - tic}")
        # overriden_fire.step(atoms, f=prediction["f"])
        # overriden_fire.step(atoms_2, f=prediction["f"])
        tac = time.time()

        v, e_last, r_last, v_last, dt, Nsteps, a, dr = \
            overriden_fire.step_override(system_size, f, e, v, e_last, r_last, v_last, dt, Nsteps, a)
        # v, e_last, r_last, v_last, dt, Nsteps, a, dr = \
        #     overriden_fire.step_override(system_size, f, e, v, e_last, r_last, v_last, dt, Nsteps, a)

        for i in range(len(list_of_atoms)):
            r = list_of_atoms[i].get_positions()
            list_of_atoms[i].set_positions(r + dr[i])

        time_10_structures.append(time.time() - tac)
        f, e, s = evaluate_list_of_atoms(list_of_atoms, chgnet)
        # print(f" 1 step fire overriden {time.time() - tac}")

        for i in range(len(list_of_atoms)):
            print(sum(np.isclose(atoms.get_positions(), list_of_atoms[i].get_positions())))
        # print(f"atoms fire difference {np.sum(atoms.get_positions() - atoms_3.get_positions())}")
        # for
        # print(
        #     f"atoms overriden fire difference {np.sum(atoms_2.get_positions() - atoms_3.get_positions())}")
    time_1_structure = len(list_of_atoms) * np.array(time_1_structure)
    time_10_structures = np.array(time_10_structures)
    plt.plot(range(number_steps), time_1_structure, label="1 structure at a time")
    plt.plot(range(number_steps), time_10_structures, label=f"{len(list_of_atoms)} structures at a time")
    plt.legend()
    plt.show()

    print(f"average time 1 structure at a time {time_1_structure.mean()}, std {time_1_structure.std()}")
    print(
        f"average time 10 structures at a time {time_10_structures.mean()}, std {time_10_structures.std()}")
    # print(f"atoms fire difference {np.sum(atoms.get_positions() - atoms_3.get_positions())}")
    # print(
    #     f"atoms overriden fire difference {np.sum(atoms_2.get_positions() - atoms_3.get_positions())}")

    print()
