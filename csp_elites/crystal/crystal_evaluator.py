import warnings
from typing import Optional, List, Dict, Tuple, Union

import numpy as np
import torch
from ase import Atoms
from ase.build import niggli_reduce
from ase.ga.utilities import CellBounds
from pymatgen.core import Structure

from csp_elites.crystal.materials_data_model import BandGapEnum
from csp_elites.map_elites.elites_utils import Species
from csp_elites.parallel_relaxation.structure_optimizer import BatchedStructureOptimizer
from csp_elites.property_calculators.band_gap_calculator import BandGapCalculator
from csp_elites.property_calculators.shear_modulus_calculator import (
    ShearModulusCalculator,
)

warnings.simplefilter("ignore")


class CrystalEvaluator:
    def __init__(
        self,
        with_force_threshold=True,
        constrained_qd=False,
        relax_every_n_generations=0,
        fmax_relaxation_convergence: float = 0.2,
        force_threshold_fmax: float = 1.0,
        compute_gradients: bool = True,
        cellbounds: Optional[CellBounds] = None,
        bd_normalisation: Union[List[Optional[Tuple[float, float]]]] = None,
    ):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.relaxer = BatchedStructureOptimizer(
            fmax_threshold=fmax_relaxation_convergence
        )
        if bd_normalisation is not None:
            band_gap_normalisation = (bd_normalisation[0][0], bd_normalisation[1][0])
            shear_modulus_normalisation = (
                bd_normalisation[0][1],
                bd_normalisation[1][1],
            )
        else:
            band_gap_normalisation, shear_modulus_normalisation = None, None

        self.band_gap_calculator = BandGapCalculator(band_gap_normalisation)

        self.shear_modulus_calculator = ShearModulusCalculator(
            shear_modulus_normalisation
        )
        self.fmax_relaxation_convergence = fmax_relaxation_convergence
        self.with_force_threshold = with_force_threshold
        self.force_threshold_fmax = force_threshold_fmax
        self.constrained_qd = constrained_qd
        self.relax_every_n_generations = relax_every_n_generations
        self.ground_state_data = {
            "energy": 9.407774,
            "band_gap": 25.6479144096375,
            "shear_modulus": 53.100777,
        }
        self.compute_gradients = compute_gradients
        self.cellbounds = (
            cellbounds
            if cellbounds is not None
            else CellBounds(
                bounds={
                    "phi": [20, 160],
                    "chi": [20, 160],
                    "psi": [20, 160],
                    "a": [2, 40],
                    "b": [2, 40],
                    "c": [2, 40],
                }
            )
        )

    def batch_compute_fitness_and_bd(
        self,
        list_of_atoms: List[Dict[str, np.ndarray]],
        n_relaxation_steps: int,
    ):
        list_of_atoms = [Atoms.fromdict(atoms) for atoms in list_of_atoms]
        kill_list = self._check_atoms_in_cellbounds(list_of_atoms)
        relaxation_results, updated_atoms = self.relaxer.relax(
            list_of_atoms, n_relaxation_steps
        )
        energies = -np.array(
            [
                relaxation_results[i]["trajectory"]["energies"]
                for i in range(len(relaxation_results))
            ]
        )
        structures = [
            relaxation_results[i]["final_structure"]
            for i in range(len(relaxation_results))
        ]
        if self.with_force_threshold:
            forces = np.array(
                [
                    relaxation_results[i]["trajectory"]["forces"]
                    for i in range(len(relaxation_results))
                ]
            )

            fitness_scores = self._apply_force_threshold(energies, forces)
        else:
            fitness_scores = energies
            forces = np.array(
                [
                    relaxation_results[i]["trajectory"]["forces"]
                    for i in range(len(relaxation_results))
                ]
            )

        band_gaps, band_gap_gradients = self._batch_band_gap_compute(structures)
        shear_moduli, shear_moduli_gradients = self._batch_shear_modulus_compute(
            structures
        )

        new_atoms_dict = [atoms.todict() for atoms in updated_atoms]

        for i in range(len(list_of_atoms)):
            new_atoms_dict[i]["info"] = list_of_atoms[i].info

        if self.constrained_qd:
            distance_to_bg = self.ground_state_data["band_gap"] - np.array(band_gaps)
            distance_to_shear = self.ground_state_data["shear_modulus"] - np.array(
                shear_moduli
            )
            forces = np.array(
                [
                    relaxation_results[i]["trajectory"]["forces"]
                    for i in range(len(relaxation_results))
                ]
            )
            distance_to_0_force_normalised_to_100 = (
                self.compute_fmax(forces) * 100
            )  # TODO: change this normalisation
            descriptors = (
                distance_to_bg,
                distance_to_shear,
                distance_to_0_force_normalised_to_100,
            )
        else:
            descriptors = (band_gaps, shear_moduli)

        del relaxation_results
        del structures
        del list_of_atoms

        if self.compute_gradients:
            all_gradients = [
                (forces[i], band_gap_gradients[i], shear_moduli_gradients[i])
                for i in range(len(new_atoms_dict))
            ]
        else:
            all_gradients = None
        return (
            updated_atoms,
            new_atoms_dict,
            fitness_scores,
            descriptors,
            kill_list,
            all_gradients,
        )

    def batch_create_species(
        self, list_of_atoms, fitness_scores, descriptors, kill_list, all_gradients
    ):
        kill_list = np.array(kill_list)
        individual_indexes_to_add = np.arange(len(list_of_atoms))[~kill_list]
        species_list = []

        for i in individual_indexes_to_add:
            if self.compute_gradients:
                fitness_gradient = all_gradients[i][0]
                descriptor_gradients = all_gradients[i][1:]
            else:
                fitness_gradient, descriptor_gradients = None, None

            new_specie = Species(
                x=list_of_atoms[i],
                fitness=fitness_scores[i],
                desc=tuple([descriptors[j][i] for j in range(len(descriptors))]),
                fitness_gradient=fitness_gradient,
                descriptor_gradients=descriptor_gradients,
            )
            species_list.append(new_specie)

        return species_list

    def _batch_band_gap_compute(self, list_of_structures: List[Structure]):
        band_gaps = []
        all_gradients = []
        for i in range(len(list_of_structures)):
            band_gap, gradients = self._compute_band_gap(
                relaxed_structure=list_of_structures[i]
            )
            band_gaps.append(band_gap)
            all_gradients.append(gradients)
        return band_gaps, all_gradients

    def _compute_band_gap(
        self,
        relaxed_structure: Structure,
        bandgap_type: Optional[BandGapEnum] = BandGapEnum.SCAN,
    ):
        if bandgap_type is None:
            for i, method in ((0, "PBE"), (1, "GLLB-SC"), (2, "HSE"), (3, "SCAN")):
                graph_attrs = torch.tensor([i])
                bandgap, gradients = self.band_gap_calculator.compute(
                    structure=relaxed_structure,
                    band_gap_type=graph_attrs,
                    compute_gradients=self.compute_gradients,
                )

                print(f"{method} band gap")
                print(f"\tRelaxed STO = {float(bandgap):.2f} eV.")
        else:
            graph_attrs = torch.tensor([bandgap_type.value])

            bandgap, gradients = self.band_gap_calculator.compute(
                structure=relaxed_structure,
                band_gap_type=graph_attrs,
                compute_gradients=self.compute_gradients,
            )

        return float(bandgap), gradients

    def _batch_shear_modulus_compute(self, list_of_structures: List[Structure]):
        shear_moduli = []
        all_gradients = []
        for structure in list_of_structures:
            shear_modulus, gradients = self.shear_modulus_calculator.compute(
                structure,
                compute_gradients=self.compute_gradients,
            )
            shear_moduli.append(shear_modulus)
            all_gradients.append(gradients)
        return shear_moduli, all_gradients

    def _check_atoms_in_cellbounds(
        self,
        list_of_atoms: List[Atoms],
    ) -> List[bool]:
        kill_list = []
        for i, atoms in enumerate(list_of_atoms):
            if not self.cellbounds.is_within_bounds(atoms.get_cell()):
                niggli_reduce(atoms)
                if not self.cellbounds.is_within_bounds(atoms.get_cell()):
                    kill_individual = True
                else:
                    kill_individual = True
            else:
                kill_individual = False

            kill_list.append(kill_individual)
        return kill_list

    def _apply_force_threshold(
        self, energies: np.ndarray, forces: np.ndarray
    ) -> np.ndarray:
        fitnesses = np.array(energies)
        if self.with_force_threshold:
            fmax = self.compute_fmax(forces)
            indices_above_threshold = np.argwhere(
                fmax > self.force_threshold_fmax
            ).reshape(-1)
            forces_above_threshold = -1 * np.abs(
                fmax[fmax > self.force_threshold_fmax] - self.force_threshold_fmax
            )
            np.put(fitnesses, indices_above_threshold, forces_above_threshold)
        return fitnesses

    def compute_fmax(self, forces: np.ndarray):
        return np.max((forces**2).sum(axis=2), axis=1) ** 0.5
