import os
import pathlib
from collections import defaultdict
from enum import Enum
from typing import List, Dict, Optional, Tuple

import imageio
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
        maximum_number_of_atoms_in_reference: int = 24,
        number_of_atoms_in_system: int = 24,
        filter_for_experimental_structures: bool = True
    ):
        self.known_structures_docs, self.known_atoms = \
            self.initialise_reference_structures(
                formula, maximum_number_of_atoms_in_reference, filter_for_experimental_structures)
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


    def initialise_reference_structures(self, formula: str = "TiO2", max_number_of_atoms: int=12, experimental:bool = True):
        docs, atom_objects = get_all_materials_with_formula(formula)
        if experimental:
            experimentally_observed = [docs[i] for i in range(len(docs)) if
                                       (not docs[i].theoretical) and
                                       (len(docs[i].structure) <= max_number_of_atoms)
                                       ]
            experimetnally_observed_atoms = [AseAtomsAdaptor.get_atoms(docs[i].structure) for i in
                                             range(len(experimentally_observed))]
            return experimentally_observed, experimetnally_observed_atoms
        else:
            references_filtered_for_size = [docs[i] for i in range(len(docs)) if
                                       (len(docs[i].structure) <= max_number_of_atoms)
                                       ]
            atoms_filtered_for_size = [AseAtomsAdaptor.get_atoms(docs[i].structure) for i in
                                             range(len(references_filtered_for_size))]
            return references_filtered_for_size, atoms_filtered_for_size
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
            spacegroups = list(spacegroup_dictionary.keys())
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

            filename = f"ind_{archive.centroid_ids[individual_index]}_{file_tag}_{individual_confid}{primitive_string}.png"
            self.structure_viewer.write_image(
                str(directory_to_save / filename),
                magnification=5,
            )
        return filename

    def save_best_structures_by_energy(
        self, archive: Archive, fitness_range: Optional[Tuple[float, float]],
        top_n_individuals_to_save: int, directory_to_save: pathlib.Path,
        save_primitive: bool = False,
        save_visuals: bool = True,
    ) -> List[int]:
        sorting_indices = np.argsort(archive.fitnesses)
        sorting_indices = np.flipud(sorting_indices)

        top_n_individuals = sorting_indices[:top_n_individuals_to_save]
        individuals_in_fitness_range = np.argwhere((archive.fitnesses >= fitness_range[0]) * (archive.fitnesses <= fitness_range[1])).reshape(-1)

        indices_to_check = np.unique(np.hstack([top_n_individuals, individuals_in_fitness_range]))
        if save_visuals:
            self.save_structure_visualisations(
                archive=archive,
                structure_indices=list(indices_to_check),
                directory_to_save=directory_to_save,
                file_tag="best_by_energy",
                save_primitive=save_primitive,
            )
        return list(indices_to_check)

    def save_best_structures_by_symmetry(
        self, archive: Archive, matched_space_group_dict: Optional[Dict[str, np.ndarray]],
            directory_to_save: pathlib.Path, save_primitive: bool = False, save_visuals: bool = True
    ) -> List[int]:
        if matched_space_group_dict is None:
            space_group_matching_dict, _ = symmetry_evaluation.find_individuals_with_reference_symmetries(
                archive.individuals,
                None,
            )

        if save_visuals:
            for desired_symmetry, indices_to_check in matched_space_group_dict.items():
                self.save_structure_visualisations(
                    archive=archive,
                    structure_indices=list(indices_to_check),
                    directory_to_save=directory_to_save,
                    file_tag=f"best_by_symmetry_{desired_symmetry}",
                    save_primitive=save_primitive,
                )

        structure_indices = []
        for el in list(matched_space_group_dict.values()):
            structure_indices += el

        return structure_indices

    def compare_archive_to_references(
        self, archive: Archive, indices_to_compare: List[int], directory_to_save: pathlib.Path,
    ) -> pd.DataFrame:
        logging_data = []

        for structure_index in indices_to_compare:
            structure = AseAtomsAdaptor.get_structure(archive.individuals[structure_index])
            primitive_structure = SpacegroupAnalyzer(structure).find_primitive()
            new_row = {
                "individual_confid": archive.individuals[structure_index].info["confid"],
                "centroid_index": archive.centroid_ids[structure_index],
                "symmetry": self.get_spacegroup_for_individual(archive.individuals[structure_index]),
                "symmetry_match": None,
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

    def quick_view_structure(self, archive: Archive, individual_index: int):
        structure = AseAtomsAdaptor.get_structure(archive.individuals[individual_index])
        self.structure_viewer.set_structure(structure)
        self.structure_viewer.show()

    def gif_centroid_over_time(
        self, experiment_directory_path: pathlib.Path, centroid_filepath: pathlib.Path,
            centroid_index: int,
        save_primitive: bool = False, number_of_frames_for_gif: int = 50,
    ):
        list_of_files = [name for name in os.listdir(f"{experiment_directory_path}") if
                         not os.path.isdir(name)]
        list_of_archives = [filename for filename in list_of_files if
                            ("archive_" in filename) and (".pkl" in filename)]

        temp_dir = experiment_directory_path / "tempdir"
        temp_dir.mkdir(exist_ok=False)

        archive_ids = []
        plots = []
        for i, filename in enumerate(list_of_archives):
            if "relaxed_" in filename:
                continue
            else:
                archive_id = list_of_archives[i].lstrip("relaxed_archive_").rstrip(".pkl")
                archive = Archive.from_archive(pathlib.Path(experiment_directory_path / filename),
                                               centroid_filepath=centroid_filepath)
                archive_ids.append(archive_id)

                plot_name = self.save_structure_visualisations(
                    archive=archive,
                    structure_indices=[centroid_index],
                    directory_to_save=temp_dir,
                    file_tag=archive_id,
                    save_primitive=save_primitive,
                )
                plots.append(plot_name)

        frames = []
        sorting_ids = np.argsort(np.array(archive_ids, dtype=int))
        for id in sorting_ids:
            image = imageio.v2.imread(str(temp_dir / plots[id]))
            frames.append(image)

        imageio.mimsave(f"{experiment_directory_path}/structure_over_time_{centroid_index}.gif",  # output gif
                        frames, duration=400, )
        for plot_name in plots:
            image_path = temp_dir / plot_name
            image_path.unlink()
        temp_dir.rmdir()



if __name__ == '__main__':

    experiment_tag = "20230808_20_31_TiO2_100_niches_with_threshold"
    centroid_path = "centroids_200_2_band_gap_0_100_shear_modulus_0_120.dat"
    # archive_number = 2086
    # structure_number = 42
    experiment_directory_path = pathlib.Path(__file__).parent.parent.parent / ".experiment.nosync" / "experiments" /experiment_tag
    centroid_full_path = pathlib.Path(__file__).parent.parent.parent / ".experiment.nosync" / "experiments" / "centroids" / centroid_path
    # relaxed_archive_location = pathlib.Path(__file__).parent.parent.parent / ".experiment.nosync" / "experiments" /experiment_tag / f"relaxed_archive_{archive_number}.pkl"
    # unrelaxed_archive_location = experiment_directory_path / f"archive_{archive_number}.pkl"
    #
    # archive = Archive.from_archive(unrelaxed_archive_location, centroid_filepath=centroid_full_path)

    symmetry_evaluation = SymmetryEvaluation()
    # symmetry_evaluation.quick_view_structure(archive, structure_number)

    symmetry_evaluation.gif_centroid_over_time(
        experiment_directory_path=experiment_directory_path, centroid_filepath=centroid_full_path, centroid_index=25,
    )
