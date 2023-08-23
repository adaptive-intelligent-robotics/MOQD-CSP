from typing import List, Dict, Tuple, Optional

from ase import Atoms
from ase.ga.cutandsplicepairing import CutAndSplicePairing
from ase.ga.offspring_creator import OperationSelector
from ase.ga.soft_mutation import SoftMutation
from ase.ga.standardmutations import StrainMutation, PermutationMutation
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import CellBounds, closest_distances_generator
import numpy as np
# from jax import jit
# from numba import prange, jit


class CrystalSystem:
    def __init__(self,
        atom_numbers_to_optimise: List[float],
        volume: int = 240,
        ratio_of_covalent_radii: float = 0.5,
        splits: Dict[Tuple[int], int] = None,
        cellbounds: CellBounds = None,
        operator_probabilities: List[float] = (4., 2., 2., 2.),
        compound_formula: Optional[str] = None,
    ):

        self.volume = volume
        self.atom_numbers_to_optimise = atom_numbers_to_optimise
        self.atomic_numbers = np.unique(np.array(self.atom_numbers_to_optimise))
        self.ratio_of_covalent_radii = ratio_of_covalent_radii
        self.splits = splits if splits is not None else {(2,): 1, (1,): 1}
        self.cellbounds = cellbounds if cellbounds is not None else CellBounds(bounds={'phi': [20, 160], 'chi': [20, 160],
                                    'psi': [20, 160], 'a': [2, 60],
                                    'b': [2, 60], 'c': [2, 60]})
        self._start_generator = self._initialise_start_generator()
        self._strain_mutation = None
        self._cut_and_splice = None
        self._soft_mutation = None
        self._permutation_mutation = None
        self.operators = self._initialise_operators(operator_probabilities)
        self.compound_formula = compound_formula

    def create_one_individual(self, individual_id: Optional[int]):
        try:
            individual = self._start_generator.get_new_candidate()
        except AssertionError:
            individual = self._start_generator.get_new_candidate()
            print("Stupid ase error")

        individual.info["confid"] = individual_id
        individual.info["curiosity"] = 0
        return individual

    # @jit(parallel=True)
    def create_n_individuals(self, number_of_individuals: int) -> List[Dict[str, np.ndarray]]:
        individuals = []
        for i in range(number_of_individuals):
            new_individual = self.create_one_individual(individual_id=i)
            new_individual = new_individual.todict()
            individuals.append(new_individual)
        return individuals

    def _initialise_start_generator(self):
        closest_distances = closest_distances_generator(atom_numbers=self.atom_numbers_to_optimise,
                                                        ratio_of_covalent_radii=self.ratio_of_covalent_radii)  # equivalent to blmin
        return StartGenerator(Atoms('', pbc=True), self.atom_numbers_to_optimise, closest_distances, box_volume=self.volume,
                       number_of_variable_cell_vectors=3,
                       cellbounds=self.cellbounds, splits=self.splits)

    def _initialise_operators(self, operator_probabilities: List[float]):
        closest_distances = closest_distances_generator(atom_numbers=self.atomic_numbers, ratio_of_covalent_radii=self.ratio_of_covalent_radii)

        self._cut_and_splice = CutAndSplicePairing(
            Atoms('', pbc=True), len(self.atom_numbers_to_optimise),
            closest_distances, p1=1., p2=0., minfrac=0.15,
            number_of_variable_cell_vectors=3, cellbounds=self.cellbounds, use_tags=False,
        )

        self._strain_mutation = StrainMutation(
            closest_distances, stddev=0.7, cellbounds=self.cellbounds,
            number_of_variable_cell_vectors=3, use_tags=False,
        )

        closest_distances_soft_mutation = closest_distances_generator(self.atom_numbers_to_optimise, 0.1)
        self._soft_mutation = SoftMutation(
            closest_distances_soft_mutation, bounds=[2., 5.], use_tags=False,
        )

        self._permutation_mutation = PermutationMutation(len(self.atom_numbers_to_optimise))
        return OperationSelector(
            operator_probabilities,
            [self._cut_and_splice, self._soft_mutation, self._strain_mutation, self._permutation_mutation],
        )

    def update_operator_scaling_volumes(self, population: List[Atoms]):
        self._strain_mutation.update_scaling_volume(population, w_adapt=0.5, n_adapt=4)
        self._cut_and_splice.update_scaling_volume(population, w_adapt=0.5, n_adapt=4)
