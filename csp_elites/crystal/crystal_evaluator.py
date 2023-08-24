import warnings
from typing import Optional, List, Dict

import matgl
import numpy as np
import torch
from ase import Atoms
from ase.build import niggli_reduce
from ase.ga.ofp_comparator import OFPComparator
from ase.ga.utilities import CellBounds
from matplotlib import pyplot as plt
from megnet.utils.models import load_model as megnet_load_model
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from csp_elites.crystal.materials_data_model import BandGapEnum, MaterialProperties
from csp_elites.map_elites.elites_utils import Species
from csp_elites.parallel_relaxation.structure_optimizer import MultiprocessOptimizer

warnings.simplefilter("ignore")


class CrystalEvaluator:
    def __init__(self,
                 comparator: OFPComparator = None,
                 with_force_threshold=True,
                 constrained_qd=False,
                 relax_every_n_generations=0,
                 ):

        self.relaxer = MultiprocessOptimizer()
        self.comparator = comparator
        self.band_gap_calculator = matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi")
        self.shear_modulus_calculator = megnet_load_model("logG_MP_2018")
        self.fmax_threshold = 0.2
        self.with_force_threshold = with_force_threshold
        self.constrained_qd = constrained_qd
        self.relax_every_n_generations = relax_every_n_generations

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

    def batch_compute_fitness_and_bd(self,
                                     list_of_atoms: List[Dict[str, np.ndarray]], cellbounds: CellBounds,
                                     really_relax: bool, behavioral_descriptor_names: List[
                MaterialProperties],
                                     n_relaxation_steps: int,

                                     ):

        list_of_atoms = [Atoms.fromdict(atoms) for atoms in list_of_atoms]

        kill_list = self.check_atoms_in_cellbounds(list_of_atoms, cellbounds)
        # todo: filter evaluations to remove killed_individuals
        if n_relaxation_steps == 0:
            structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in list_of_atoms]
            fitness_scores, relaxation_results = self.batch_compute_energy(
                list_of_structures=structures,
                really_relax=really_relax,
                n_steps=n_relaxation_steps,
            )
            updated_atoms = list_of_atoms
        else:
            relaxation_results, updated_atoms = self.relaxer.relax(list_of_atoms, n_relaxation_steps)
            energies = - np.array([relaxation_results[i]["trajectory"]["energies"] for i in range(len(relaxation_results))])
            structures = [relaxation_results[i]["final_structure"] for i in range(len(relaxation_results))]
            if self.with_force_threshold:
                forces = np.array([relaxation_results[i]["trajectory"]["forces"] for i in
                            range(len(relaxation_results))])

                fitness_scores = self._apply_force_threshold(energies, forces)
            else:
                fitness_scores = energies

        band_gaps = self._batch_band_gap_compute(structures)
        shear_moduli = self._batch_shear_modulus_compute(structures)

        new_atoms_dict = [atoms.todict() for atoms in updated_atoms]

        for i in range(len(list_of_atoms)):
            new_atoms_dict[i]["info"] = list_of_atoms[i].info

        if self.constrained_qd:
            forces = np.array([relaxation_results[i]["trajectory"]["forces"] for i in
                               range(len(relaxation_results))])
            distance_to_0_force_normalised_to_100 = self.compute_fmax(forces) * 100 # TODO: change this normalisation
            descriptors = (band_gaps, shear_moduli, distance_to_0_force_normalised_to_100)
        else:
            descriptors = (band_gaps, shear_moduli)

        del relaxation_results
        del structures
        del list_of_atoms
        return updated_atoms, new_atoms_dict, fitness_scores, descriptors, kill_list

    def batch_create_species(self, list_of_atoms, fitness_scores, descriptors, kill_list):
        # todo: here could do dict -> atoms conversion

        kill_list = np.array(kill_list)
        individual_indexes_to_add = np.arange(len(list_of_atoms))[~kill_list]
        species_list = []
        for i in individual_indexes_to_add:
            new_specie = Species(
                x=list_of_atoms[i],
                fitness=fitness_scores[i],
                desc=tuple([descriptors[j][i] for j in range(len(descriptors))])
                # desc=(descriptors[0][i], descriptors[1][i])
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
        forces, energies, stresses = self.relaxer._evaluate_list_of_atoms(list_of_structures)
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

        if self.with_force_threshold:
            fitnesses = self._apply_force_threshold(energies, forces)
        else:
            fitnesses = -1 * energies
        return fitnesses, reformated_output

    def _apply_force_threshold(self, energies: np.ndarray, forces: np.ndarray) -> np.ndarray:
        fitnesses = np.array(energies)
        if self.with_force_threshold:
            fmax = self.compute_fmax(forces)
            indices_above_threshold = np.argwhere(fmax > self.fmax_threshold).reshape(-1)
            forces_above_threshold = -1 * np.abs(fmax[fmax > self.fmax_threshold] - self.fmax_threshold)
            np.put(fitnesses, indices_above_threshold, forces_above_threshold)
        return fitnesses

    def compute_composition(self, element_blocks: List[List[int]]):
        element_blocks = np.array(element_blocks)
        counts = np.unique(element_blocks, return_counts=True)
        oxygen_count = counts[1][counts[0] == 8] if counts[1][counts[0] == 8] else 0
        non_titanium = len(element_blocks) - counts[1][counts[0] == 22]
        return oxygen_count / non_titanium

    def compute_fmax(self, forces: np.ndarray):
        return np.max((forces ** 2).sum(axis=2), axis=1) ** 0.5

    def compute_structure_density(self, list_of_structures: List[Structure]):
        density_list = []
        for i in range(len(list_of_structures)):
            density_list.append(list_of_structures[i].density)
        return density_list









    # def compute_local_order(self, atoms: Atoms):
    #     all_local_orders = self.comparator.get_local_orders(atoms)
    #     pass

    # def compute_mixing_energy(
    #         self, crystal_energy: float,
    #         crystal_1_proportion: float,
    #         edge_crystal_1_energy: float,
    #         edge_crystal_2_energy: float,
    # ):
    #     return crystal_energy * (
    #             (1 - crystal_1_proportion) * edge_crystal_2_energy +
    #             crystal_1_proportion * edge_crystal_1_energy
    #     )
    #
    # def compute_fitness_and_bd_mixing_system(
    #         self, atoms: Atoms, cellbounds, really_relax: bool, crystal_1_proportion: float,
    #         edge_crystal_1_energy: float, edge_crystal_2_energy: float,
    # ):
    #     if not cellbounds.is_within_bounds(atoms.get_cell()):
    #         niggli_reduce(atoms)
    #         if not cellbounds.is_within_bounds(atoms.get_cell()):
    #             return 0, (0, 0), True
    #
    #     # Compute fitness (i.e. energy)
    #     crystal_energy, relaxation_results = self.compute_energy(atoms=atoms,
    #                                                              really_relax=really_relax)
    #
    #     cell = atoms.get_cell()
    #
    #     # Check whether the individual is valid to be added to the archive
    #     kill_individual = False
    #     if not cellbounds.is_within_bounds(cell):
    #         kill_individual = True
    #
    #     fitness_score = self.compute_mixing_energy(
    #         crystal_energy=crystal_energy,
    #         crystal_1_proportion=crystal_1_proportion,
    #         edge_crystal_1_energy=edge_crystal_1_energy,
    #         edge_crystal_2_energy=edge_crystal_2_energy,
    #     )
    #
    #     band_gap = self.compute_band_gap(relaxed_structure=relaxation_results["final_structure"])
    #
    #     return fitness_score(band_gap, crystal_energy), kill_individual

    # def _update_trajectories(self, trajectories: List[TrajectoryObserver], forces, energies,
    #                          stresses) -> List[TrajectoryObserver]:
    #     for i in range(len(trajectories)):
    #         trajectories[i].energies.append(energies[i])
    #         trajectories[i].forces.append(forces)
    #         trajectories[i].stresses.append(stresses)
    #
    #     return trajectories

    # def _finalize_atoms(self, atoms, energy=None, forces=None, stress=None):
    #     # atoms.wrap() # todo: what does atoms.wrap() do? Why does it not work with M3gnetx
    #     calc = SinglePointCalculator(atoms, energy=energy, forces=forces,
    #                                  stress=stress)
    #     atoms.calc = calc
    #     raw_score = float(-atoms.get_potential_energy())
    #     set_raw_score(atoms, raw_score)
    # def compute_formation_energy(self, relaxed_structure):
    #     return float(self.formation_energy_calculator.predict_structure(relaxed_structure))
