import warnings
from multiprocessing.managers import BaseManager
from typing import Optional, Tuple, List, Dict

import matgl
import numpy as np
from chgnet.model import CHGNet
from chgnet.model.dynamics import TrajectoryObserver, StructOptimizer
from megnet.utils.models import load_model as megnet_load_model
import torch
from ase import Atoms
from ase.build import niggli_reduce
from ase.calculators.singlepoint import SinglePointCalculator
from ase.ga import set_raw_score
from ase.ga.ofp_comparator import OFPComparator
from ase.ga.utilities import CellBounds
from matplotlib import pyplot as plt
# from numba import jit, njit, prange
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from csp_elites.crystal.materials_data_model import BandGapEnum, MaterialProperties
from csp_elites.map_elites.elites_utils import Species

warnings.simplefilter("ignore")


class CrystalEvaluator:
    def __init__(self,
        comparator: OFPComparator = None,

):
        # device = torch.device('cpu' if torch.cuda.is_available() else 'cuda')
        # relax_model = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        # relax_model = relax_model.to(device)
        # self.relaxer = OverridenRelaxer(potential=relax_model)
        # self.relaxer = StructOptimizer()
        # self.relaxer = MultiprocessingOptimizer()
        self.comparator = comparator
        self.band_gap_calculator = matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi")
        # self.band_gap_calculator.to(device)
        # self.formation_energy_calculator = matgl.load_model("M3GNet-MP-2018.6.1-Eform")
        self.shear_modulus_calculator = megnet_load_model("logG_MP_2018")
        # self.formation_energy_calculator.to(device)
        self.property_to_function_mapping = {
            MaterialProperties.BAND_GAP: self.compute_band_gap,
            MaterialProperties.ENERGY_FORMATION: self.compute_formation_energy,
            MaterialProperties.SHEAR_MODULUS: self.compute_shear_modulus
        }
        self.model = CHGNet.load()

        self.element_to_number_map = {
            "Ti": 22,
            "O": 8,
            "S": 16,
        }

    # todo:  @jit(cache=True)
    def compute_energy(self, atoms: Atoms, really_relax, n_steps: int = 10) -> float:
        # relaxation_results = self.relaxer.relax(atoms, steps=n_steps, verbose=False)

        # energy = float(
        #     relaxation_results["trajectory"].energies[-1] / len(atoms.get_atomic_numbers()))
        # forces = relaxation_results["trajectory"].forces[-1]
        # stresses = relaxation_results["trajectory"].stresses[-1]
        # self._finalize_atoms(atoms, energy=energy, forces=forces, stress=stresses)
        # relaxation_results = None

        structure = AseAtomsAdaptor.get_structure(atoms)
        prediction = self.model.predict_structure(structure)
        # traj_observer = TrajectoryObserver(atoms)
        # traj_observer.energies.append(prediction["e"])
        # traj_observer.forces.append(prediction["f"])
        # traj_observer.stresses.append(prediction["s"])
        self._finalize_atoms(atoms, energy=prediction["e"], forces=prediction["f"], stress=prediction["s"])
        relaxation_results = {
            "final_structure": structure,
            "trajectory": {
                "energies": prediction["e"],
                "forces": prediction["f"],
                "stresses": prediction["s"],
            }
        }
        return -prediction["e"], relaxation_results
        # return float(-atoms.get_potential_energy()), relaxation_results


    #todo: @jit(parallel=True, cache=True)
    # @jit(parallel=True)
    def compute_fitness_and_bd(self,
                               atoms: Dict[str, np.ndarray],
                               cellbounds: CellBounds,
                               really_relax: bool,
                               behavioral_descriptor_names: List[MaterialProperties],
                               n_relaxation_steps: int,
                               ) -> Tuple[float, Tuple[float, float], bool]:
        # convert dictionary to atoms object
        # print(atoms)
        atoms = Atoms.fromdict(atoms)
        if not cellbounds.is_within_bounds(atoms.get_cell()):
            niggli_reduce(atoms)
            if not cellbounds.is_within_bounds(atoms.get_cell()):
                return 0, (0, 0), True

        # Compute fitness (i.e. energy)
        fitness_score, relaxation_results = self.compute_energy(atoms=atoms, really_relax=really_relax,
                                                                n_steps=n_relaxation_steps)

        cell = atoms.get_cell()

        # Check whether the individual is valid to be added to the archive
        kill_individual = False
        if not cellbounds.is_within_bounds(cell):
            kill_individual = True

        # Compute bd dimension

        behavioral_descriptor_values = []

        for i in range(len(behavioral_descriptor_names)):
            bd_value = self.property_to_function_mapping[behavioral_descriptor_names[i]](relaxed_structure=relaxation_results["final_structure"])
            behavioral_descriptor_values.append(bd_value)

        updated_atoms = AseAtomsAdaptor.get_atoms(relaxation_results["final_structure"])
        new_atoms_dict = updated_atoms.todict()
        del relaxation_results
        new_atoms_dict["info"] = atoms.info
        return new_atoms_dict, fitness_score, behavioral_descriptor_values, kill_individual

    def compute_band_gap(self, relaxed_structure, bandgap_type: Optional[
        BandGapEnum] = BandGapEnum.SCAN):
        if bandgap_type is None:
            for i, method in ((0, "PBE"), (1, "GLLB-SC"), (2, "HSE"), (3, "SCAN")):
                graph_attrs = torch.tensor([i])
                bandgap = self.band_gap_calculator.predict_structure(
                    structure=relaxed_structure, state_feats=graph_attrs
                )

                print(f"{method} band gap")
                print(f"\tRelaxed STO = {float(bandgap):.2f} eV.")
        else:
            graph_attrs = torch.tensor([bandgap_type.value])
            bandgap = self.band_gap_calculator.predict_structure(
                structure=relaxed_structure, state_feats=graph_attrs
            )
        return float(bandgap) * 25 # TODO CHANGE THIS

    def compute_formation_energy(self, relaxed_structure):
        return float(self.formation_energy_calculator.predict_structure(relaxed_structure))

    def _finalize_atoms(self, atoms, energy=None, forces=None, stress=None):
        # atoms.wrap() # todo: what does atoms.wrap() do? Why does it not work with M3gnet
        calc = SinglePointCalculator(atoms, energy=energy, forces=forces,
                                     stress=stress)
        atoms.calc = calc
        raw_score = float(-atoms.get_potential_energy())
        set_raw_score(atoms, raw_score)

    def compare_to_target_structures(self,
        generated_structures: List[Atoms],
        target_structures: Dict[str, List[Atoms]],
        directory_string: str,
    ):
        scores = []

        for target_id, target_structure in target_structures.items():
            scores_for_target = []
            for generated_structure in generated_structures:
                cosine_distance = self.comparator._compare_structure(generated_structure,
                                                                target_structure)
                scores_for_target.append(cosine_distance)
                if cosine_distance <= 0.15:
                    print(f"cosine distance {cosine_distance}")
                    print(
                        f"target structure params {target_structure.get_cell_lengths_and_angles()}")
                    print(
                        f"predicted structure params {generated_structure.get_cell_lengths_and_angles()}")

            scores.append(scores_for_target)
            fig, ax = plt.subplots()
            ax.hist(scores_for_target, range=(0, 0.5), bins=20)
            ax.set_title(f"{target_id}")
            ax.set_xlabel("Cosine distance from structure")
            ax.set_ylabel("Number of generated structures")
            if directory_string is None:
                plt.show()
            else:
                plt.savefig(f"{directory_string}/hist_comp_{target_id}.svg", format="svg")

    def compute_shear_modulus(self, relaxed_structure: Structure) -> float:
        predicted_G = 10 ** self.shear_modulus_calculator.predict_structure(relaxed_structure).ravel()[0]
        return predicted_G

    def reduce_structure_to_conventional(self, atoms: Atoms):
        structure = AseAtomsAdaptor.get_structure(atoms)
        conventional_structure = SpacegroupAnalyzer(structure=structure)
        return conventional_structure

    def compute_local_order(self, atoms: Atoms):
        all_local_orders = self.comparator.get_local_orders(atoms)
        pass

    def compute_mixing_energy(
            self, crystal_energy: float,
            crystal_1_proportion: float,
            edge_crystal_1_energy: float,
            edge_crystal_2_energy: float,
    ):
        return crystal_energy * (
                (1 - crystal_1_proportion) * edge_crystal_2_energy +
                crystal_1_proportion * edge_crystal_1_energy
        )

    def compute_fitness_and_bd_mixing_system(
            self, atoms: Atoms, cellbounds, really_relax: bool, crystal_1_proportion: float,
            edge_crystal_1_energy: float, edge_crystal_2_energy: float,
    ):
        if not cellbounds.is_within_bounds(atoms.get_cell()):
            niggli_reduce(atoms)
            if not cellbounds.is_within_bounds(atoms.get_cell()):
                return 0, (0, 0), True

        # Compute fitness (i.e. energy)
        crystal_energy, relaxation_results = self.compute_energy(atoms=atoms,
                                                                 really_relax=really_relax)

        cell = atoms.get_cell()

        # Check whether the individual is valid to be added to the archive
        kill_individual = False
        if not cellbounds.is_within_bounds(cell):
            kill_individual = True

        fitness_score = self.compute_mixing_energy(
            crystal_energy=crystal_energy,
            crystal_1_proportion=crystal_1_proportion,
            edge_crystal_1_energy=edge_crystal_1_energy,
            edge_crystal_2_energy=edge_crystal_2_energy,
        )

        band_gap = self.compute_band_gap(relaxed_structure=relaxation_results["final_structure"])

        return fitness_score(band_gap, crystal_energy), kill_individual

    def batch_compute_fitness_and_bd(self,
                                     list_of_atoms: List[Atoms], cellbounds: CellBounds,
                                     really_relax: bool, behavioral_descriptor_names: List[
                MaterialProperties],
                                     n_relaxation_steps: int,
                                     ):

        list_of_atoms = [Atoms.fromdict(atoms) for atoms in list_of_atoms]
        structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in list_of_atoms]
        kill_list = self.check_atoms_in_cellbounds(list_of_atoms, cellbounds)
        # todo: filter evaluations to remove killed_individuals
        fitness_scores, relaxation_results = self.batch_compute_energy(
            list_of_structures=structures,
            really_relax=really_relax,
            n_steps=n_relaxation_steps,
        )

        band_gaps = self._batch_band_gap_compute(structures)
        shear_moduli = self._batch_shear_modulus_compute(structures)
        # todo: finalise atoms here? currently no need
        updated_atoms = [AseAtomsAdaptor.get_atoms(relaxation_results[i]["final_structure"])
                         for i in range(len(list_of_atoms))]
        new_atoms_dict = [atoms.todict() for atoms in updated_atoms]
        del relaxation_results
        for i in range(len(list_of_atoms)):
            new_atoms_dict[i]["info"] = list_of_atoms[i].info
        return new_atoms_dict, fitness_scores, (band_gaps, shear_moduli), kill_list

    def batch_create_species(self, list_of_atoms, fitness_scores, descriptors, kill_list):
        # todo: here could do dict -> atoms conversion

        kill_list = np.array(kill_list)
        individual_indexes_to_add = np.arange(len(list_of_atoms))[~kill_list]
        species_list = []
        for i in individual_indexes_to_add:
            new_specie = Species(
                x=list_of_atoms[i],
                fitness=fitness_scores[i],
                desc=(descriptors[0][i], descriptors[1][i])
            )
            species_list.append(new_specie)

        return species_list

    def _batch_band_gap_compute(self, list_of_structures: List[Structure]):
        # structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in list_of_atoms]
        band_gaps = []
        for i in range(len(list_of_structures)):
            band_gap = self.compute_band_gap(relaxed_structure=list_of_structures[i])
            band_gaps.append(band_gap)
        return band_gaps

    def _batch_shear_modulus_compute(self, list_of_structures: List[Structure]):
        shear_moduli = 10 ** self.shear_modulus_calculator.predict_structures(list_of_structures).ravel()
        return shear_moduli

    def check_atoms_in_cellbounds(self, list_of_atoms: List[Atoms],
                                  cellbounds: CellBounds,
                                  ) -> List[bool]:
        kill_list = []
        for i, atoms in enumerate(list_of_atoms):
            if not cellbounds.is_within_bounds(atoms.get_cell()):
                niggli_reduce(atoms)
                if not cellbounds.is_within_bounds(atoms.get_cell()):
                    kill_individual = True
                else:
                    kill_individual = True
            else:
                kill_individual = False

            kill_list.append(kill_individual)
        return kill_list

    def batch_compute_energy(self, list_of_structures: List[Structure], really_relax, n_steps: int = 10) -> float:
        # trajectories = [TrajectoryObserver(atoms) for atoms in list_of_atoms]
        forces, energies, stresses = self._evaluate_list_of_atoms(list_of_structures)
        # trajectories = self._update_trajectories(trajectories, forces, energies, stresses)

        # final_structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in list_of_atoms]
        reformated_output = []
        for i in range(len(list_of_structures)):
            reformated_output.append(
                {"final_structure": list_of_structures[i],
                 "trajectory": {
                     "energies": energies[i],
                     "forces": forces[i],
                     "stresses": stresses[i],
                 },
                 }
            )
        return -1 * energies, reformated_output

    def _evaluate_list_of_atoms(self, list_of_structures: List[Structure]):
        # list_of_structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in list_of_atoms]

        predictions = self.model.predict_structure(list_of_structures,
                                                   batch_size=len(list_of_structures))
        if isinstance(predictions, dict):
            predictions = [predictions]

        forces = np.array([pred["f"] for pred in predictions])
        energies = np.array([pred["e"] for pred in predictions])
        stresses = np.array([pred["s"] for pred in predictions])
        return forces, energies, stresses

    def _update_trajectories(self, trajectories: List[TrajectoryObserver], forces, energies,
                             stresses) -> List[TrajectoryObserver]:
        for i in range(len(trajectories)):
            trajectories[i].energies.append(energies[i])
            trajectories[i].forces.append(forces)
            trajectories[i].stresses.append(stresses)

        return trajectories


    def compute_composition(self, element_blocks: List[List[int]]):
        element_blocks = np.array(element_blocks)
        counts = np.unique(element_blocks, return_counts=True)
        oxygen_count = counts[1][counts[0] == 8] if counts[1][counts[0] == 8] else 0
        non_titanium = len(element_blocks) - counts[1][counts[0] == 22]
        return oxygen_count / non_titanium

def compute_composition_test(element_blocks: List[List[int]]):
    element_blocks = np.array(element_blocks)
    counts = np.unique(element_blocks, return_counts=True, axis=1)
    oxygen_count = np.sum(np.array(counts[0] == 8, int) * counts[1], axis=1)
    non_titanium = np.sum(np.array(counts[0]!=22, int) * counts[1], axis =1)
    return oxygen_count / non_titanium


if __name__ == '__main__':
    pure_tio2 = [22] * 8 + [8] * 16
    pure_tis2 = [22] * 8 + [16] * 16
    half_half = [22] * 8 + [16] * 8 + [8] * 8
    mix = [22] * 8 + [16] * 4 + [8] * 12

    compute_composition_test([pure_tio2, pure_tis2, half_half, mix])

    for el in [pure_tio2, pure_tis2, half_half, mix]:
        print(compute_composition_test(el))
