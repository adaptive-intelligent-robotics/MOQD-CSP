
import pathlib
from collections import defaultdict
from enum import Enum
from typing import List, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase import Atoms
from ase.ga.ofp_comparator import OFPComparator
from ase.spacegroup import get_spacegroup
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.vis.structure_vtk import StructureVis

from csp_elites.map_elites.archive import Archive
from csp_elites.utils.get_mpi_structures import get_all_materials_with_formula


class SpaceGroups(str, Enum):
    PYMATGEN = "pymatgen"
    SPGLIB = "spglib"


class SymmetryEvaluation:
    def __init__(
        self,
        formula: str = "TiO2",
        tolerance: float = 0.1,
        maximum_number_of_atoms_in_reference: int = 12,
        number_of_atoms_in_system: int = 24,
    ):
        self.known_structures_docs, self.known_atoms = self.initialise_reference_structures(formula, maximum_number_of_atoms_in_reference)
        self.material_ids = [self.known_structures_docs[i].material_id for i in range(len(self.known_structures_docs))]
        self.known_space_groups_pymatgen, self.known_space_group_spglib = self._get_reference_spacegroups(
            tolerance)
        self.tolerance = tolerance
        self.spacegroup_matching = {
            SpaceGroups.PYMATGEN: self.known_space_groups_pymatgen,
            SpaceGroups.SPGLIB: self.known_space_group_spglib,
        }
        self.comparator = OFPComparator(n_top=number_of_atoms_in_system, dE=1.0,
                                   cos_dist_max=1e-3, rcut=10., binwidth=0.05,
                                   pbc=[True, True, True], sigma=0.05, nsigma=4,
                                   recalculate=False)
        self.structure_viewer = StructureVis(show_polyhedron=False, show_bonds=True)
        self.structure_matcher = StructureMatcher()

        # self.fitness_threshold = fitness_threshold


    def initialise_reference_structures(self, formula: str = "TiO2", max_number_of_atoms: int=12):
        docs, atom_objects = get_all_materials_with_formula(formula)
        experimentally_observed = [docs[i] for i in range(len(docs)) if
                                   (not docs[i].theoretical) and
                                   (len(docs[i].structure) <= max_number_of_atoms)
                                   ]
        experimetnally_observed_atoms = [AseAtomsAdaptor.get_atoms(docs[i].structure) for i in
                                         range(len(experimentally_observed))]
        return experimentally_observed, experimetnally_observed_atoms
    def find_individuals_with_reference_symmetries(
        self,
        individuals: List[Atoms],
        indices_to_check: Optional[List[int]],
        spacegroup_type: SpaceGroups=SpaceGroups.SPGLIB,
    ):
        spacegroup_dictionary = self.compute_symmetries_from_individuals(individuals, None)

        reference_spacegroups = self.spacegroup_matching[spacegroup_type]

        archive_to_reference_mapping = defaultdict(list)
        for key in reference_spacegroups:
            if key in spacegroup_dictionary.keys():
                archive_to_reference_mapping[key] += spacegroup_dictionary[key]

        return archive_to_reference_mapping, spacegroup_dictionary

    def compute_symmetries_from_individuals(
        self, individuals: List[Atoms], indices_to_check: Optional[List[int]] = None,
    ) -> Dict[str, List[int]]:
        if indices_to_check is None:
            indices_to_check = range(len(individuals))

        spacegroup_dictionary = defaultdict(list)
        for index in indices_to_check:
            for symprec in [self.tolerance]:
                spacegroup = self.get_spacegroup_for_individual(individuals[index])
                if spacegroup is not None:
                    spacegroup_dictionary[spacegroup].append(index)

        return spacegroup_dictionary

    def get_spacegroup_for_individual(self, individual: Atoms):
        for symprec in [self.tolerance]:
            try:
                spacegroup = get_spacegroup(individual, symprec=symprec).symbol
            except RuntimeError:
                spacegroup = None
        return spacegroup

    def _get_reference_spacegroups(self, tolerance: float):
        pymatgen_spacegeroups = [self.known_structures_docs[i].structure.get_space_group_info() for i in
                                     range(len(self.known_atoms))]

        spglib_spacegroups = []
        for el in self.known_atoms:
            spglib_spacegroups.append(get_spacegroup(el, symprec=tolerance).symbol)

        return pymatgen_spacegeroups, spglib_spacegroups

    def _get_indices_to_check(self, fitnesses: np.ndarray) -> np.ndarray:
        return np.argwhere(fitnesses > self.tolerance).reshape(-1)

    def plot_histogram(
        self,
        spacegroup_dictionary: Dict[str, List[int]],
        against_reference: bool = True,
        spacegroup_type: SpaceGroups = SpaceGroups.SPGLIB,
        save_directory: Optional[pathlib.Path] = None
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
        plt.ylabel("Count of individuals in structure")
        plt.xlabel("Symmetry type computed using spglib")


        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_directory is None:
            plt.show()
        else:
            reference_string = "with_ref" if against_reference else ""
            plt.savefig(save_directory / f"ind_symmetries_histogram_{reference_string}.png", format="png")

    def save_structure_visualisations(
        self, archive: Archive, structure_indices: List[int], directory_to_save: pathlib.Path,
        file_tag: str, save_primitive: bool = False,
    ):
        primitive_string = "_primitive" if save_primitive else ""
        for individual_index in structure_indices:
            individual_as_structure = AseAtomsAdaptor.get_structure(archive.individuals[individual_index])
            if save_primitive:
                individual_as_structure = SpacegroupAnalyzer(structure=individual_as_structure).find_primitive()
            self.structure_viewer.set_structure(individual_as_structure)
            individual_confid = archive.individuals[individual_index].info["confid"]


            self.structure_viewer.write_image(
                str(directory_to_save / f"{file_tag}_{individual_confid}_list_index_{individual_index}{primitive_string}.png"),
                magnification=5,
            )

    def save_best_structures_by_energy(
        self, archive: Archive, fitness_range: Optional[Tuple[float, float]],
        top_n_individuals_to_save: int, directory_to_save: pathlib.Path,
        save_primitive: bool = False,
    ) -> List[int]:
        sorting_indices = np.argsort(archive.fitnesses)
        sorting_indices = np.flipud(sorting_indices)

        top_n_individuals = sorting_indices[:top_n_individuals_to_save]
        individuals_in_fitness_range = np.argwhere((archive.fitnesses >= fitness_range[0]) * (archive.fitnesses <= fitness_range[1])).reshape(-1)

        indices_to_check = np.unique(np.hstack([top_n_individuals, individuals_in_fitness_range]))

        self.save_structure_visualisations(
            archive=archive,
            structure_indices=list(indices_to_check),
            directory_to_save=directory_to_save,
            file_tag="ind_best_by_energy_confid",
            save_primitive=save_primitive,
        )
        return list(indices_to_check)

    def save_best_structures_by_symmetry(
        self, archive: Archive, matched_space_group_dict: Optional[Dict[str, np.ndarray]],
            directory_to_save: pathlib.Path, save_primitive: bool = False,
    ) -> List[int]:
        if matched_space_group_dict is None:
            space_group_matching_dict, _ = symmetry_evaluation.find_individuals_with_reference_symmetries(
                archive.individuals,
                None,
            )

        for desired_symmetry, indices_to_check in matched_space_group_dict.items():
            self.save_structure_visualisations(
                archive=archive,
                structure_indices=list(indices_to_check),
                directory_to_save=directory_to_save,
                file_tag=f"ind_best_by_symmetry_{desired_symmetry}",
                save_primitive=save_primitive,
            )
        return list(matched_space_group_dict.values())

    def compare_archive_to_references(
        self, archive: Archive, indices_to_compare: List[int], directory_to_save: pathlib.Path,
    ) -> pd.DataFrame:
        logging_data = []

        for structure_index in indices_to_compare:
            structure = AseAtomsAdaptor.get_structure(archive.individuals[structure_index])
            primitive_structure = SpacegroupAnalyzer(structure).find_primitive()
            new_row = {
                "individual_confid": archive.individuals[structure_index].info["confid"],
                "list_index": structure_index,
                "symmetry": self.get_spacegroup_for_individual(archive.individuals[structure_index]),
                "fitness": archive.fitnesses[structure_index],
                "descriptors": archive.descriptors[structure_index],
                "number_of_cells_in_primitive_cell": len(primitive_structure),
                "matches": {},
                "distances": {},
            }

            for known_structure in self.known_structures_docs:
                new_row["matches"][str(known_structure.material_id)+"_match"] = \
                    self.structure_matcher.fit(structure, known_structure.structure)
                if len(primitive_structure) == len(known_structure.structure):
                    primitive_structure.sort()
                    known_structure.structure.sort()
                    new_row["distances"][str(known_structure.material_id) + "_distance_to"] = \
                        float(self.comparator._compare_structure(
                            AseAtomsAdaptor.get_atoms(primitive_structure),
                            AseAtomsAdaptor.get_atoms(known_structure.structure)))

                else:
                    new_row["distances"][
                        str(known_structure.material_id) + "_distance_to"] = 1
            logging_data.append(new_row)

        df = pd.DataFrame(logging_data)
        df = pd.concat([df, df["matches"].apply(pd.Series)], axis=1)
        df.drop(columns="matches", inplace=True)
        df = pd.concat([df, df["distances"].apply(pd.Series)], axis=1)
        df.drop(columns="distances", inplace=True)
        df.to_csv(str(directory_to_save / "ind_top_symmetry_statistic.csv"))
        return df


if __name__ == '__main__':

    # get_spacegroup()
    experiment_tag = "20230730_05_02_TiO2_200_niches_10_relaxation_steps"
    archive_number = 25030
    relaxed_archive_location = pathlib.Path(__file__).parent.parent.parent / ".experiment.nosync" / "experiments" /experiment_tag / f"relaxed_archive_{archive_number}.pkl"
    unrelaxed_archive_location = pathlib.Path(__file__).parent.parent.parent / ".experiment.nosync" / "experiments" /experiment_tag / f"archive_{archive_number}.pkl"

    archive = Archive.from_archive(unrelaxed_archive_location)

    symmetry_evaluation = SymmetryEvaluation()
    relaxed_dict = symmetry_evaluation.compute_symmetries_from_individuals(
        individuals=archive.individuals,
        fitnesses=archive.fitnesses,
    )
    symmetry_evaluation.plot_histogram(relaxed_dict, False)

    symmetry_mapping_to_references = symmetry_evaluation.find_individuals_with_reference_symmetries(individuals=archive.individuals, fitnesses=archive.fitnesses)
    print(symmetry_mapping_to_references)
