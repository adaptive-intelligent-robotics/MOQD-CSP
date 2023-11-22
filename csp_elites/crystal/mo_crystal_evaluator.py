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
from csp_elites.crystal.crystal_evaluator import CrystalEvaluator

warnings.simplefilter("ignore")


class MOCrystalEvaluator(CrystalEvaluator):

    def __init__(
            self,
            with_force_threshold=True,
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
        magmoms = np.array(
            [
                relaxation_results[i]["trajectory"]["magmoms"]
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

            energy_fitness_scores = self._apply_force_threshold(energies, forces)
        else:
            energy_fitness_scores = energies
            forces = np.array(
                [
                    relaxation_results[i]["trajectory"]["forces"]
                    for i in range(len(relaxation_results))
                ]
            )
            
        fitness_scores = np.column_stack((energy_fitness_scores, magmoms))
        
        band_gaps, band_gap_gradients = self._batch_band_gap_compute(structures)
        shear_moduli, shear_moduli_gradients = self._batch_shear_modulus_compute(
            structures
        )

        new_atoms_dict = [atoms.todict() for atoms in updated_atoms]

        for i in range(len(list_of_atoms)):
            new_atoms_dict[i]["info"] = list_of_atoms[i].info

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
