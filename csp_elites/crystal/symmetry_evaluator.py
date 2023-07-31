# from spglib import get_spacegroup
import copy
import pathlib
import pickle
from collections import defaultdict
from enum import Enum
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.spacegroup import get_spacegroup
from pymatgen.io.ase import AseAtomsAdaptor

from csp_elites.utils.get_mpi_structures import get_all_materials_with_formula
from csp_elites.utils.plot import load_archive_from_pickle

class SpaceGroups(str, Enum):
    PYMATGEN = "pymatgen"
    SPGLIB = "spglib"


class SymmetryEvaluation:
    def __init__(self, formula: str="TiO2", tolerance: float=0.1, fitness_threshold: float= 8.5):
        self.known_space_groups_pymatgen, self.known_space_group_spglib = self._get_reference_spacegroups(formula, tolerance)
        self.tolerance = tolerance
        self.spacegroup_matching = {
            SpaceGroups.PYMATGEN: self.known_space_groups_pymatgen,
            SpaceGroups.SPGLIB: self.known_space_group_spglib,
        }
        self.fitness_threshold = fitness_threshold

    def find_individuals_with_reference_symmetries(
        self,
        individuals: List[Atoms],
        fitnesses: np.ndarray,
        spacegroup_type: SpaceGroups=SpaceGroups.SPGLIB,
    ):
        pass

    def compute_symmetries_from_individuals(self, individuals: List[Atoms], fitnesses: np.ndarray) -> Dict[str, List[int]]:
        indices_to_check = self._get_indices_to_check(fitnesses)
        spacegroup_dictionary = defaultdict(list)
        for index in indices_to_check:
            for symprec in [self.tolerance]:
                try:
                    spacegroup = get_spacegroup(individuals[index], symprec=symprec).symbol
                    spacegroup_dictionary[spacegroup].append(index)
                except RuntimeError:
                    # print("Spacegroup not found")
                    continue

        return spacegroup_dictionary

    def _get_reference_spacegroups(self, formula: str, tolerance: float):
        docs, atom_objects = get_all_materials_with_formula(formula)
        experimentally_observed = [docs[i] for i in range(len(docs)) if
                                   not docs[i].theoretical]
        experimetnally_observed_atoms = [AseAtomsAdaptor.get_atoms(docs[i].structure) for i in
                                         range(len(experimentally_observed))]
        pymatgen_spacegeroups = [docs[i].structure.get_space_group_info() for i in
                                     range(len(experimentally_observed))]

        spglib_spacegroups = []
        for el in experimetnally_observed_atoms:
            spglib_spacegroups.append(get_spacegroup(el, symprec=tolerance).symbol)

        return pymatgen_spacegeroups, spglib_spacegroups

    def _get_indices_to_check(self, fitnesses: np.ndarray) -> np.ndarray:
        return np.argwhere(fitnesses > self.tolerance).reshape(-1)

    def plot_histogram(
        self,
        spacegroup_dictionary: Dict[str, List[int]],
        against_reference: bool = True,
        spacegroup_type: SpaceGroups = SpaceGroups.SPGLIB,
    ):
        if against_reference:
            spacegroups = self.spacegroup_matching[spacegroup_type]
            spacegroup_counts = []
            for key in spacegroups:
                if key in spacegroup_dictionary.keys():
                    spacegroup_counts.append(len(spacegroup_dictionary[key]))
                else:
                    spacegroup_counts.append(0)
        else:
            spacegroups = spacegroup_dictionary.keys()
            spacegroup_counts = [len(value) for value in spacegroup_dictionary.values()]

        plt.bar(spacegroups, spacegroup_counts)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    docs, atom_objects = get_all_materials_with_formula("TiO2")
    experimentally_observed = [docs[i] for i in range(len(docs)) if
                               not docs[i].theoretical]
    experimetnally_observed_atoms = [AseAtomsAdaptor.get_atoms(docs[i].structure) for i in range(len(experimentally_observed))]
    experimental_space_groups = [docs[0].structure.get_space_group_info() for i in range(len(experimentally_observed))]

    # get_spacegroup()
    experiment_tag = "20230730_05_02_TiO2_200_niches_10_relaxation_steps"
    archive_number = 25030
    relaxed_archive_location = pathlib.Path(__file__).parent.parent / ".experiment.nosync" / "experiments" /experiment_tag / f"relaxed_archive_{archive_number}.pkl"
    unrelaxed_archive_location = pathlib.Path(__file__).parent.parent / ".experiment.nosync" / "experiments" /experiment_tag / f"archive_{archive_number}.pkl"
    with open(relaxed_archive_location, "rb") as file:
        relaxed_archive = pickle.load(file)

    fintesses_unrelaxed, _, _, unrelaxed_individuals = load_archive_from_pickle(unrelaxed_archive_location)
    fitnesses_unrelaxed = np.array(fintesses_unrelaxed)
    ind_to_check_unrelaxed = np.argwhere(fintesses_unrelaxed > 8.5).reshape(-1)
    fitnesses_relaxed = [relaxed_archive[0][i] for i in range(len(relaxed_archive[0]))]
    fitnesses_relaxed = np.array(fitnesses_relaxed)
    ind_to_check_relaxed = np.argwhere(fitnesses_relaxed > 9).reshape(-1)
    unrelaxed_individuals = [Atoms.fromdict(unrelaxed_individuals[i]) for i in
                           range(len(unrelaxed_individuals))]

    relaxed_individuals = [Atoms.fromdict(relaxed_archive[3][i]) for i in
                           range(len(relaxed_archive[3]))]
    symmetry_evaluation = SymmetryEvaluation()
    relaxed_dict = symmetry_evaluation.compute_symmetries_from_individuals(individuals=relaxed_individuals, fitnesses=fitnesses_relaxed)
    print()
