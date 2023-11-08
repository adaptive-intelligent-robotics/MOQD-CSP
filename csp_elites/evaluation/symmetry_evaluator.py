import json
import os
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Tuple, Union

import imageio
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageDraw
from ase import Atoms
from ase.ga.ofp_comparator import OFPComparator
from ase.spacegroup import get_spacegroup
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.vis.structure_vtk import StructureVis

plt.rcParams["savefig.dpi"] = 1000

from csp_elites.map_elites.archive import Archive
from csp_elites.utils.get_mpi_structures import get_all_materials_with_formula
from csp_elites.utils.plot import load_centroids, get_voronoi_finite_polygons_2d


class SpaceGroups(str, Enum):
    PYMATGEN = "pymatgen"
    SPGLIB = "spglib"


class ConfidenceLevels(int, Enum):
    GOLD = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    NO_MATCH = 0

    @staticmethod
    def get_string(confidence_level):
        confidence_dictionary = {
            ConfidenceLevels.GOLD: "Gold Standard",
            ConfidenceLevels.HIGH: "High",
            ConfidenceLevels.MEDIUM: "Medium",
            ConfidenceLevels.LOW: "Low",
            ConfidenceLevels.NO_MATCH: "No Match",
        }
        return confidence_dictionary[confidence_level]


class PlottingMode(str, Enum):
    MP_REFERENCE_VIEW = "mp_reference_view"
    ARCHIVE_MATCHES_VIEW = "archive_matches_view"


@dataclass
class PlottingMatches:
    centroid_indices: List[int]
    mp_references: List[str]
    confidence_levels: List[ConfidenceLevels]
    euclidian_distances: List[float]
    descriptors: List[np.ndarray]
    energy_difference: List[float]
    plotting_mode: PlottingMode


class SymmetryEvaluation:
    def __init__(
        self,
        formula: str = "TiO2",
        tolerance: float = 0.1,
        maximum_number_of_atoms_in_reference: int = 24,
        number_of_atoms_in_system: int = 24,
        filter_for_experimental_structures: bool = True,
        reference_data_archive: Optional[Archive] = None,
    ):
        self.known_structures_docs = self.initialise_reference_structures(
            formula,
            maximum_number_of_atoms_in_reference,
            filter_for_experimental_structures,
        )
        self.material_ids = [
            self.known_structures_docs[i].material_id
            for i in range(len(self.known_structures_docs))
        ]
        (
            self.known_space_groups_pymatgen,
            self.known_space_group_spglib,
        ) = self._get_reference_spacegroups(tolerance)
        self.tolerance = tolerance
        self.spacegroup_matching = {
            SpaceGroups.PYMATGEN: self.known_space_groups_pymatgen,
            SpaceGroups.SPGLIB: self.known_space_group_spglib,
        }
        self.comparator = OFPComparator(
            n_top=number_of_atoms_in_system,
            dE=1.0,
            cos_dist_max=1e-3,
            rcut=10.0,
            binwidth=0.05,
            pbc=[True, True, True],
            sigma=0.05,
            nsigma=4,
            recalculate=False,
        )
        self.structure_viewer = StructureVis(show_polyhedron=False, show_bonds=True)
        self.structure_matcher = StructureMatcher()
        self.fingerprint_distance_threshold = 0.1
        self.reference_data = (
            reference_data_archive.to_dataframe()
            if reference_data_archive is not None
            else None
        )

    def initialise_reference_structures(
        self,
        formula: str = "TiO2",
        max_number_of_atoms: int = 12,
        experimental: bool = True,
    ):
        docs, atom_objects = get_all_materials_with_formula(formula)
        if experimental:
            experimentally_observed = [
                docs[i]
                for i in range(len(docs))
                if (not docs[i].theoretical)
                and (len(docs[i].structure) <= max_number_of_atoms)
            ]

            return experimentally_observed
        else:
            references_filtered_for_size = [
                docs[i]
                for i in range(len(docs))
                if (len(docs[i].structure) <= max_number_of_atoms)
            ]

            return references_filtered_for_size

    def find_individuals_with_reference_symmetries(
        self,
        individuals: List[Atoms],
        indices_to_check: Optional[List[int]],
        spacegroup_type: SpaceGroups = SpaceGroups.SPGLIB,
    ):
        spacegroup_dictionary = self.compute_symmetries_from_individuals(
            individuals, None
        )

        reference_spacegroups = self.spacegroup_matching[spacegroup_type]

        archive_to_reference_mapping = defaultdict(list)
        for key in reference_spacegroups:
            if key in spacegroup_dictionary.keys():
                archive_to_reference_mapping[key] += spacegroup_dictionary[key]

        return archive_to_reference_mapping, spacegroup_dictionary

    def compute_symmetries_from_individuals(
        self,
        individuals: List[Atoms],
        archive_indices_to_check: Optional[List[int]] = None,
    ) -> Dict[str, List[int]]:
        if archive_indices_to_check is None:
            archive_indices_to_check = range(len(individuals))

        spacegroup_dictionary = defaultdict(list)
        for index in archive_indices_to_check:
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
        pymatgen_spacegeroups = [
            self.known_structures_docs[i].structure.get_space_group_info()
            for i in range(len(self.known_structures_docs))
        ]

        spglib_spacegroups = []
        for el in self.known_structures_docs:
            spglib_spacegroups.append(
                get_spacegroup(
                    AseAtomsAdaptor.get_atoms(el.structure), symprec=tolerance
                ).symbol
            )

        return pymatgen_spacegeroups, spglib_spacegroups

    def _get_indices_to_check(self, fitnesses: np.ndarray) -> np.ndarray:
        return np.argwhere(fitnesses > self.tolerance).reshape(-1)

    def plot_histogram(
        self,
        spacegroup_dictionary: Dict[str, List[int]],
        against_reference: bool = True,
        spacegroup_type: SpaceGroups = SpaceGroups.SPGLIB,
        save_directory: Optional[pathlib.Path] = None,
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
            plt.savefig(
                save_directory / f"ind_symmetries_histogram_{reference_string}.png",
                format="png",
            )
        plt.clf()

    def save_structure_visualisations(
        self,
        archive: Archive,
        structure_indices: List[int],
        directory_to_save: pathlib.Path,
        file_tag: str,
        save_primitive: bool = False,
    ):
        primitive_string = "_primitive" if save_primitive else ""
        for individual_index in structure_indices:
            individual_as_structure = AseAtomsAdaptor.get_structure(
                archive.individuals[individual_index]
            )
            if save_primitive:
                individual_as_structure = SpacegroupAnalyzer(
                    structure=individual_as_structure
                ).find_primitive()
            self.structure_viewer.set_structure(individual_as_structure)
            individual_confid = archive.individuals[individual_index].info["confid"]

            filename = f"ind_{archive.centroid_ids[individual_index]}_{file_tag}_{individual_confid}{primitive_string}.png"
            self.structure_viewer.write_image(
                str(directory_to_save / filename),
                magnification=5,
            )
        return filename

    def save_best_structures_by_energy(
        self,
        archive: Archive,
        fitness_range: Optional[Tuple[float, float]],
        top_n_individuals_to_save: int,
        directory_to_save: pathlib.Path,
        save_primitive: bool = False,
        save_visuals: bool = True,
    ) -> List[int]:
        sorting_indices = np.argsort(archive.fitnesses)
        sorting_indices = np.flipud(sorting_indices)

        top_n_individuals = sorting_indices[:top_n_individuals_to_save]
        individuals_in_fitness_range = np.argwhere(
            (archive.fitnesses >= fitness_range[0])
            * (archive.fitnesses <= fitness_range[1])
        ).reshape(-1)

        indices_to_check = np.unique(
            np.hstack([top_n_individuals, individuals_in_fitness_range])
        )
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
        self,
        archive: Archive,
        matched_space_group_dict: Optional[Dict[str, np.ndarray]],
        directory_to_save: pathlib.Path,
        save_primitive: bool = False,
        save_visuals: bool = True,
    ) -> List[int]:
        if matched_space_group_dict is None:
            (
                space_group_matching_dict,
                _,
            ) = self.find_individuals_with_reference_symmetries(
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

    def make_symmetry_to_material_id_dict(self):
        spacegroup_to_reference_dictionary = defaultdict(list)

        for el in self.known_structures_docs:
            spacegroup = self.get_spacegroup_for_individual(
                AseAtomsAdaptor.get_atoms(el.structure)
            )
            spacegroup_to_reference_dictionary[spacegroup].append(str(el.material_id))
        return spacegroup_to_reference_dictionary

    def _load_reference_data_path(self, reference_data_path: Optional[pathlib.Path]):
        if reference_data_path is not None:
            reference_data = pd.read_csv(reference_data_path)
            reference_data.index = reference_data["Unnamed: 0"].to_list()
            reference_data.drop(columns="Unnamed: 0", inplace=True)
        else:
            reference_data = None
        return reference_data

    def executive_summary_csv(
        self,
        archive: Archive,
        indices_to_compare: List[int],
        directory_to_save: pathlib.Path,
        reference_data_path: Optional[pathlib.Path] = None,
    ) -> pd.DataFrame:
        summary_data = []

        symmetry_to_material_id_dict = self.make_symmetry_to_material_id_dict()
        if indices_to_compare is None:
            indices_to_compare = list(range(len(archive.individuals)))

        for structure_index in indices_to_compare:
            structure = AseAtomsAdaptor.get_structure(
                archive.individuals[structure_index]
            )
            primitive_structure = SpacegroupAnalyzer(structure).find_primitive()
            spacegroup = self.get_spacegroup_for_individual(
                archive.individuals[structure_index]
            )
            symmetry_match = (
                symmetry_to_material_id_dict[spacegroup]
                if spacegroup in symmetry_to_material_id_dict.keys()
                else []
            )

            summary_row = {
                "individual_confid": archive.individuals[structure_index].info[
                    "confid"
                ],
                "centroid_index": archive.centroid_ids[structure_index],
                "fitness": archive.fitnesses[structure_index],
                "descriptors": archive.descriptors[structure_index],
                "symmetry": spacegroup,
                "number_of_cells_in_primitive_cell": len(primitive_structure),
                "matches": [],
            }

            for known_structure in self.known_structures_docs:
                reference_id = str(known_structure.material_id)
                structure_matcher_match = self.structure_matcher.fit(
                    structure, known_structure.structure
                )

                distance_to_known_structure = self._compute_distance(
                    primitive_structure,
                    known_structure.structure,
                )

                if (
                    distance_to_known_structure <= 0.1
                    or structure_matcher_match
                    or str(reference_id) in symmetry_match
                ):
                    if self.reference_data is not None:
                        ref_band_gap = self.reference_data[reference_id]["band_gap"]
                        ref_shear_modulus = self.reference_data[reference_id][
                            "shear_modulus"
                        ]
                        ref_energy = self.reference_data[reference_id]["energy"]
                        ref_centroid = self.reference_data[reference_id]["centroid_id"]
                        error_to_bg = (
                            (ref_band_gap - archive.descriptors[structure_index][0])
                            / ref_band_gap
                            * 100
                        )
                        error_to_shear = (
                            (
                                ref_shear_modulus
                                - archive.descriptors[structure_index][1]
                            )
                            / ref_shear_modulus
                            * 100
                        )
                        error_to_energy = (
                            (ref_energy - archive.fitnesses[structure_index])
                            / ref_energy
                            * 100
                        )
                        distance_in_bd_space = np.sqrt(
                            (ref_band_gap - archive.descriptors[structure_index][0])
                            ** 2
                            + (
                                ref_shear_modulus
                                - archive.descriptors[structure_index][1]
                            )
                            ** 2
                        )
                    else:
                        (
                            error_to_energy,
                            error_to_bg,
                            error_to_shear,
                            ref_centroid,
                            distance_in_bd_space,
                        ) = (None, None, None, None, None)
                    summary_row["matches"].append(
                        {
                            reference_id: {
                                "symmetry": reference_id in symmetry_match,
                                "structure_matcher": structure_matcher_match,
                                "distance": distance_to_known_structure,
                                "centroid": ref_centroid,
                                "reference_energy_perc_difference": error_to_energy,
                                "reference_band_gap_perc_difference": error_to_bg,
                                "reference_shear_modulus_perc_difference": error_to_shear,
                                "euclidian_distance_in_bd_space": distance_in_bd_space,
                            }
                        }
                    )

            summary_data.append(summary_row)
        individuals_with_matches = [
            individual for individual in summary_data if individual["matches"]
        ]

        df = pd.DataFrame(individuals_with_matches)
        try:
            df = df.explode("matches")
            df["matches"] = df["matches"].apply(lambda x: x.items())
            df = df.explode("matches")
        except KeyError:
            print("No matches found")
        df[["reference", "match_info"]] = pd.DataFrame(
            df["matches"].tolist(), index=df.index
        )
        df.drop(columns="matches", inplace=True)
        df = pd.concat([df, df["match_info"].apply(pd.Series)], axis=1)
        df.drop(columns="match_info", inplace=True)
        df.to_csv(directory_to_save / "ind_executive_summary.csv")
        return df, individuals_with_matches

    def matches_for_plotting(self, individuals_with_matches):
        centroids_with_matches = []
        mp_reference_of_matches = []
        confidence_levels = []
        euclidian_distance_to_matches = []
        all_descriptors = []
        true_centroid_indices = []
        energy_difference = []
        for i, individual in enumerate(individuals_with_matches):
            sorted_matches = sorted(
                individual["matches"],
                key=lambda x: list(x.values())[0]["euclidian_distance_in_bd_space"],
            )
            match = sorted_matches[0]
            match_dictionary = list(match.values())[0]
            centroids_with_matches.append(int(individual["centroid_index"]))
            mp_reference_of_matches.append(list(match.keys())[0])
            confidence_levels.append(
                self._assign_confidence_level_in_match(
                    match_dictionary, individual["centroid_index"]
                )
            )
            euclidian_distance_to_matches.append(
                match_dictionary["euclidian_distance_in_bd_space"]
            )
            all_descriptors.append(individual["descriptors"])
            true_centroid_indices.append(match_dictionary["centroid"])
            energy_difference.append(
                match_dictionary["reference_energy_perc_difference"]
            )

        plotting_matches_from_archive = PlottingMatches(
            centroids_with_matches,
            mp_reference_of_matches,
            confidence_levels,
            euclidian_distance_to_matches,
            all_descriptors,
            energy_difference,
            PlottingMode.ARCHIVE_MATCHES_VIEW,
        )

        unique_matches, counts = np.unique(mp_reference_of_matches, return_counts=True)

        ref_centroids_with_matches = []
        ref_mp_reference_of_matches = []
        ref_confidence_levels = []
        ref_euclidian_distance_to_matches = []
        ref_all_descriptors = []
        ref_energy_difference = []

        for match_mp_ref in unique_matches:
            match_indices = np.argwhere(
                np.array(mp_reference_of_matches) == match_mp_ref
            ).reshape(-1)
            confidence_levels_for_ref = [
                confidence_levels[i].value for i in match_indices
            ]
            best_confidence_indices = np.argwhere(
                confidence_levels_for_ref == np.max(confidence_levels_for_ref)
            ).reshape(-1)
            match_indices = np.take(match_indices, best_confidence_indices)
            euclidian_distances = [
                euclidian_distance_to_matches[i] for i in match_indices
            ]
            euclidian_distances = np.array(euclidian_distances)

            closest_euclidian_distance_index = np.argwhere(
                euclidian_distances == np.min(euclidian_distances)
            ).reshape(-1)
            best_match_index = match_indices[closest_euclidian_distance_index]
            # print(f"{match_mp_ref} {best_match_index}")

            index_in_archive_list = best_match_index[0]
            ref_centroids_with_matches.append(
                int(true_centroid_indices[index_in_archive_list])
            )
            ref_mp_reference_of_matches.append(match_mp_ref)
            ref_confidence_levels.append(confidence_levels[index_in_archive_list])
            ref_euclidian_distance_to_matches.append(
                euclidian_distance_to_matches[index_in_archive_list]
            )
            ref_all_descriptors.append(all_descriptors[index_in_archive_list])
            ref_energy_difference.append(energy_difference[index_in_archive_list])

        plotting_matches_from_mp = PlottingMatches(
            ref_centroids_with_matches,
            ref_mp_reference_of_matches,
            ref_confidence_levels,
            ref_euclidian_distance_to_matches,
            ref_all_descriptors,
            ref_energy_difference,
            PlottingMode.MP_REFERENCE_VIEW,
        )

        return plotting_matches_from_archive, plotting_matches_from_mp

    def write_report_summary_json(
        self,
        plotting_matches_from_archive: PlottingMatches,
        directory_string: Union[pathlib.Path, str],
    ):
        if bool("mp-390" in plotting_matches_from_archive.mp_references):
            indices_to_check = np.argwhere(
                np.array(plotting_matches_from_archive.mp_references) == "mp-390"
            ).reshape(-1)
            confidence_scores = np.take(
                plotting_matches_from_archive.confidence_levels, indices_to_check
            ).reshape(-1)
            ground_state_match = max(confidence_scores)
        else:
            ground_state_match = ConfidenceLevels.NO_MATCH
        if bool("mp-34688" in plotting_matches_from_archive.mp_references):
            indices_to_check = np.argwhere(
                np.array(plotting_matches_from_archive.mp_references) == "mp-34688"
            ).reshape(-1)
            confidence_scores = np.take(
                plotting_matches_from_archive.confidence_levels, indices_to_check
            ).reshape(-1)
            fooled_ground_state_match = max(confidence_scores)
        else:
            fooled_ground_state_match = ConfidenceLevels.NO_MATCH

        summary_dict = {
            "ground_state_match": ConfidenceLevels.get_string(
                ConfidenceLevels(ground_state_match)
            ),
            "fooled_ground_state_match": ConfidenceLevels.get_string(
                ConfidenceLevels(fooled_ground_state_match)
            ),
            "unique_reference_matches": len(
                np.unique(plotting_matches_from_archive.mp_references)
            ),
            "number_gold": len(
                np.argwhere(
                    np.array(plotting_matches_from_archive.confidence_levels)
                    == ConfidenceLevels.GOLD.value
                )
            ),
            "number_high": len(
                np.argwhere(
                    np.array(plotting_matches_from_archive.confidence_levels)
                    == ConfidenceLevels.HIGH.value
                )
            ),
            "number_medium": len(
                np.argwhere(
                    np.array(plotting_matches_from_archive.confidence_levels)
                    == ConfidenceLevels.MEDIUM.value
                )
            ),
            "number_low": len(
                np.argwhere(
                    np.array(plotting_matches_from_archive.confidence_levels)
                    == ConfidenceLevels.LOW.value
                )
            ),
            "total_matches": len(plotting_matches_from_archive.mp_references),
        }
        with open(f"{directory_string}/ind_report_summary.json", "w") as file:
            json.dump(summary_dict, file)
        return summary_dict

    def _get_maximum_confidence_for_centroid_id(
        self, centroid_indices_to_check: np.ndarray, plotting_matches: PlottingMatches
    ):
        for centroid_id in centroid_indices_to_check:
            confidence_levels_ids = np.argwhere(
                np.array(plotting_matches.centroid_indices) == centroid_id
            ).reshape(-1)
            confidence_scores = [
                plotting_matches.confidence_levels[int(id)].value
                for id in confidence_levels_ids
            ]
            max_confidence_id = np.argwhere(
                np.array(confidence_scores) == max(confidence_scores)
            ).reshape(-1)
        return plotting_matches.confidence_levels[
            confidence_levels_ids[int(max_confidence_id)]
        ]

    def _assign_confidence_level_in_match(self, match_dictionary, centroid_id: int):
        structure_matcher_match = match_dictionary["structure_matcher"]
        ff_distance_match = (
            match_dictionary["structure_matcher"] <= self.fingerprint_distance_threshold
        )
        symmetry_match = match_dictionary["symmetry"]
        centroid_match = match_dictionary["centroid"] == centroid_id

        if structure_matcher_match:
            if centroid_match:
                return ConfidenceLevels.GOLD
            else:
                return ConfidenceLevels.HIGH
        else:
            if ff_distance_match and symmetry_match:
                return ConfidenceLevels.MEDIUM
            elif centroid_match and (ff_distance_match or symmetry_match):
                return ConfidenceLevels.MEDIUM
            else:
                return ConfidenceLevels.LOW

    def _compute_distance(
        self, structure_to_check: Structure, reference_structure: Structure
    ):
        if len(structure_to_check) == len(reference_structure):
            structure_to_check.sort()
            structure_to_check.sort()
            distance_to_known_structure = float(
                self.comparator._compare_structure(
                    AseAtomsAdaptor.get_atoms(structure_to_check),
                    AseAtomsAdaptor.get_atoms(reference_structure),
                )
            )
        else:
            distance_to_known_structure = 1
        return distance_to_known_structure

    def quick_view_structure(self, archive: Archive, individual_index: int):
        structure = AseAtomsAdaptor.get_structure(archive.individuals[individual_index])
        self.structure_viewer.set_structure(structure)
        self.structure_viewer.show()

    def gif_centroid_over_time(
        self,
        experiment_directory_path: pathlib.Path,
        centroid_filepath: pathlib.Path,
        centroid_index: int,
        save_primitive: bool = False,
    ):
        list_of_files = [
            name
            for name in os.listdir(f"{experiment_directory_path}")
            if not os.path.isdir(name)
        ]
        list_of_archives = [
            filename
            for filename in list_of_files
            if ("archive_" in filename) and (".pkl" in filename)
        ]

        temp_dir = experiment_directory_path / "tempdir"
        temp_dir.mkdir(exist_ok=False)

        archive_ids = []
        plots = []
        for i, filename in enumerate(list_of_archives):
            if "relaxed_" in filename:
                continue
            else:
                archive_id = (
                    list_of_archives[i].lstrip("relaxed_archive_").rstrip(".pkl")
                )
                archive = Archive.from_archive(
                    pathlib.Path(experiment_directory_path / filename),
                    centroid_filepath=centroid_filepath,
                )
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
            img = Image.open(str(temp_dir / plots[id]))
            # Call draw Method to add 2D graphics in an image
            I1 = ImageDraw.Draw(img)
            # Add Text to an image
            I1.text((28, 36), archive_ids[id], fill=(255, 0, 0))
            # Display edited image
            img.show()
            # Save the edited image
            img.save(str(temp_dir / plots[id]))

            image = imageio.v2.imread(str(temp_dir / plots[id]))
            frames.append(image)

        imageio.mimsave(
            f"{experiment_directory_path}/structure_over_time_{centroid_index}.gif",  # output gif
            frames,
            duration=400,
        )
        for plot_name in plots:
            image_path = temp_dir / plot_name
            image_path.unlink()
        temp_dir.rmdir()

    def group_structures_by_symmetry(
        self,
        archive: Archive,
        experiment_directory_path: pathlib.Path,
        centroid_full_path,
        filename_tag: Optional[str] = None,
        x_axis_limits=None,
        y_axis_limits=None,
    ):
        structures = archive.get_individuals_as_structures()
        groups = self.structure_matcher.group_structures(structures)
        ids_by_group = []
        for group in groups:
            id_in_group = []
            for el in group:
                match = [structures[i] == el for i in range(len(structures))]
                match_id = np.argwhere(np.array(match)).reshape(-1)
                id_in_group.append(archive.centroid_ids[match_id[0]])
            ids_by_group.append(id_in_group)

        with open(experiment_directory_path / "number_of_groups.json", "w") as file:
            json.dump({"n_groups": len(groups)}, file)

        all_centroids = load_centroids(centroid_full_path)

        bd_min_values, bd_max_values = self.get_limits_from_centroid_path(
            centroid_full_path
        )
        color_indices = np.linspace(0, 1, len(ids_by_group))
        cmap = cm.get_cmap("rainbow")
        list_of_colors = []
        for color_id in color_indices:
            list_of_colors.append(cmap(color_id)[:3])

        self.plot_2d_groups_of_structures(
            centroids=all_centroids,
            list_of_centroid_groups=ids_by_group,
            list_of_colors=list_of_colors,
            directory_string=experiment_directory_path,
            filename=f"cvt_by_structure_similarity_{filename_tag}",
            minval=bd_min_values,
            maxval=bd_max_values,
            x_axis_limits=x_axis_limits,
            y_axis_limits=y_axis_limits,
        )

    def get_limits_from_centroid_path(self, centroid_path: pathlib.Path):
        filename = centroid_path.name.rstrip(".dat")
        limits_as_string = filename.split("band_gap")[1].split("shear_modulus")
        limits = [limit.split("_") for limit in limits_as_string]
        return (int(limits[0][1]), int(limits[1][1])), (
            int(limits[0][2]),
            int(limits[1][2]),
        )

    def plot_2d_groups_of_structures(
        self,
        centroids: np.ndarray,
        list_of_centroid_groups: List[int],
        list_of_colors: List[str],
        minval: np.ndarray,
        maxval: np.ndarray,
        target_centroids: Optional[np.ndarray] = None,
        directory_string: Optional[str] = None,
        filename: Optional[str] = "cvt_plot",
        axis_labels: List[str] = ["Band Gap, eV", "Shear Modulus, GPa"],
        annotate: bool = True,
        x_axis_limits: Optional[Tuple[float, float]] = None,
        y_axis_limits: Optional[Tuple[float, float]] = None,
    ) -> Tuple[Optional[Figure], Axes]:
        """Adapted from qdax plot 2d cvt centroids function"""
        num_descriptors = centroids.shape[1]
        if num_descriptors != 2:
            raise NotImplementedError("Grid plot supports 2 descriptors only for now.")
        # my_cmap = cm.rainbow(len(list_of_centroid_groups))
        my_cmap = plt.get_cmap("rainbow", len(list_of_centroid_groups))

        # create the plot object
        mpl.rcParams["figure.figsize"] = [3, 3]
        fig, ax = plt.subplots(facecolor="white", edgecolor="white")

        if len(np.array(minval).shape) == 0 and len(np.array(maxval).shape) == 0:
            ax.set_xlim(minval, maxval)
            ax.set_ylim(minval, maxval)
        else:
            ax.set_xlim(minval[0], maxval[0])
            ax.set_ylim(minval[1], maxval[1])

        ax.set(adjustable="box", aspect="equal")

        # create the regions and vertices from centroids
        regions, vertices = get_voronoi_finite_polygons_2d(centroids)

        # fill the plot with contours
        for i, region in enumerate(regions):
            polygon = vertices[region]
            ax.fill(
                *zip(*polygon), alpha=0.05, edgecolor="black", facecolor="white", lw=1
            )
            if target_centroids is not None:
                if centroids[i] in np.array(target_centroids):
                    ax.fill(*zip(*polygon), edgecolor="black", facecolor="none", lw=4)
        # fill the plot with the colors
        for group_id, group in enumerate(list_of_centroid_groups):
            for idx in group:
                region = regions[idx]
                polygon = vertices[region]
                ax.fill(*zip(*polygon), alpha=0.8, color=list_of_colors[group_id])
                if annotate:
                    ax.annotate(
                        group_id, (centroids[idx, 0], centroids[idx, 1]), fontsize=4
                    )
        np.set_printoptions(2)
        # aesthetic
        ax.set_xlabel(f"{axis_labels[0]}")
        ax.set_ylabel(f"{axis_labels[1]}")

        if x_axis_limits is not None and y_axis_limits is not None:
            x_tick_labels = np.linspace(x_axis_limits[0], x_axis_limits[1], 6)
            y_tick_labels = np.linspace(y_axis_limits[0], y_axis_limits[1], 6)
            ax.set_xticklabels([np.around(el, 1) for el in x_tick_labels])
            ax.set_yticklabels([np.around(el, 1) for el in y_tick_labels])

        ax.set_title("Individuals Grouped by Similarity")
        ax.set_aspect("equal")

        if directory_string is None:
            fig.show()
        else:
            fig.savefig(f"{directory_string}/{filename}.png", format="png")
        plt.clf()
        return fig, ax

    def plot_matches_mapped_to_references(
        self,
        plotting_matches: PlottingMatches,
        centroids: np.ndarray,
        minval: np.ndarray,
        maxval: np.ndarray,
        centroids_from_archive: Optional[np.ndarray] = None,
        directory_string: Optional[str] = None,
        filename: Optional[str] = "cvt_matches_from_archive",
        axis_labels: List[str] = ["Band Gap, eV", "Shear Modulus, GPa"],
        annotate: bool = True,
        x_axis_limits: Optional[Tuple[float, float]] = None,
        y_axis_limits: Optional[Tuple[float, float]] = None,
    ):
        """Adapted from qdax plot 2d cvt centroids function"""
        if centroids_from_archive is None:
            centroids_from_archive = []

        num_descriptors = centroids.shape[1]
        if num_descriptors != 2:
            raise NotImplementedError("Grid plot supports 2 descriptors only for now.")

        # my_cmap = cm.viridis
        my_cmap = cm.get_cmap("inferno", 5)

        # create the plot object
        fig, ax = plt.subplots(facecolor="white", edgecolor="white")

        if len(np.array(minval).shape) == 0 and len(np.array(maxval).shape) == 0:
            ax.set_xlim(minval, maxval)
            ax.set_ylim(minval, maxval)
        else:
            ax.set_xlim(minval[0], maxval[0])
            ax.set_ylim(minval[1], maxval[1])

        ax.set(adjustable="box", aspect="equal")

        # create the regions and vertices from centroids
        regions, vertices = get_voronoi_finite_polygons_2d(centroids)

        colour_dict = {
            ConfidenceLevels.GOLD: mcolors.TABLEAU_COLORS["tab:purple"],
            ConfidenceLevels.HIGH: mcolors.TABLEAU_COLORS["tab:green"],
            ConfidenceLevels.MEDIUM: mcolors.TABLEAU_COLORS["tab:orange"],
            ConfidenceLevels.LOW: mcolors.TABLEAU_COLORS["tab:red"],
            ConfidenceLevels.NO_MATCH: mcolors.TABLEAU_COLORS["tab:gray"],
        }

        # fill the plot with contours
        target_centroid_ids = np.array(self.reference_data.loc["centroid_id"].array)
        duplicate_centroid_indices = []
        if len(np.unique(np.array(plotting_matches.centroid_indices))) != len(
            plotting_matches.centroid_indices
        ):
            unique, counts = np.unique(
                plotting_matches.centroid_indices, return_counts=True
            )
            duplicate_centroid_indices = np.take(
                unique, np.argwhere(counts != 1)
            ).reshape(-1)

        for i, region in enumerate(regions):
            polygon = vertices[region]
            ax.fill(
                *zip(*polygon), alpha=0.05, edgecolor="black", facecolor="white", lw=1
            )
            if i in target_centroid_ids:
                ax.fill(*zip(*polygon), edgecolor="gray", facecolor="none", lw=2)

            if (
                (i in centroids_from_archive)
                and (i not in plotting_matches.centroid_indices)
                and plotting_matches.plotting_mode == PlottingMode.ARCHIVE_MATCHES_VIEW
            ):
                ax.fill(
                    *zip(*polygon),
                    facecolor=colour_dict[ConfidenceLevels.NO_MATCH],
                    alpha=0.3,
                    label=ConfidenceLevels.get_string(ConfidenceLevels.NO_MATCH),
                )

        for list_index, centroid_index in enumerate(plotting_matches.centroid_indices):
            region = regions[centroid_index]
            polygon = vertices[region]

            if centroid_index in duplicate_centroid_indices:
                confidence_levels_ids = np.argwhere(
                    np.array(plotting_matches.centroid_indices) == centroid_index
                ).reshape(-1)
                confidence_scores = [
                    plotting_matches.confidence_levels[int(id)].value
                    for id in confidence_levels_ids
                ]
                max_confidence_id = np.argwhere(
                    np.array(confidence_scores) == max(confidence_scores)
                ).reshape(-1)

                colour = colour_dict[
                    plotting_matches.confidence_levels[
                        confidence_levels_ids[int(max_confidence_id[0])]
                    ]
                ]
                confidence_level = plotting_matches.confidence_levels[
                    confidence_levels_ids[int(max_confidence_id[0])]
                ]
            else:
                colour = colour_dict[plotting_matches.confidence_levels[list_index]]
                confidence_level = plotting_matches.confidence_levels[list_index]

            ax.fill(
                *zip(*polygon),
                alpha=0.8,
                color=colour,
                label=ConfidenceLevels.get_string(confidence_level),
            )
            if annotate:
                ax.annotate(
                    plotting_matches.mp_references[list_index],
                    (centroids[centroid_index, 0], centroids[centroid_index, 1]),
                    fontsize=4,
                )

        if plotting_matches.plotting_mode == PlottingMode.MP_REFERENCE_VIEW:
            descriptor_matches = np.array(plotting_matches.descriptors)
            ref_band_gaps = [
                self.reference_data.loc["band_gap"][ref]
                for ref in plotting_matches.mp_references
            ]
            ref_shear_moduli = [
                self.reference_data.loc["shear_modulus"][ref]
                for ref in plotting_matches.mp_references
            ]
            ax.scatter(
                descriptor_matches[:, 0], descriptor_matches[:, 1], color="b", s=5
            )
            ax.scatter(ref_band_gaps, ref_shear_moduli, color="b", s=5)
            for match_id in range(len(descriptor_matches)):
                ax.plot(
                    [descriptor_matches[match_id][0], ref_band_gaps[match_id]],
                    [descriptor_matches[match_id][1], ref_shear_moduli[match_id]],
                    linestyle="--",
                    color="b",
                )

        if x_axis_limits is not None and y_axis_limits is not None:
            x_tick_labels = np.linspace(x_axis_limits[0], x_axis_limits[1], 6)
            y_tick_labels = np.linspace(y_axis_limits[0], y_axis_limits[1], 6)
            ax.set_xticklabels([np.around(el, 1) for el in x_tick_labels])
            ax.set_yticklabels([np.around(el, 1) for el in y_tick_labels])

        ax.set_xlabel(f"{axis_labels[0]}")
        ax.set_ylabel(f"{axis_labels[1]}")

        if plotting_matches.plotting_mode == PlottingMode.ARCHIVE_MATCHES_VIEW:
            title = "Matches - Archive View"
        elif plotting_matches.plotting_mode == PlottingMode.MP_REFERENCE_VIEW:
            title = "Matches - Reference View"
        else:
            title = "MAP-Elites Grid"

        ax.set_title(title)
        self.legend_without_duplicate_labels(fig, ax)

        ax.set_aspect("equal")

        if directory_string is None:
            fig.show()
        else:
            fig.savefig(
                f"{directory_string}/{filename}_{plotting_matches.plotting_mode.value}.png",
                format="png",
                bbox_inches="tight",
            )
        plt.clf()
        return fig, ax

    def plot_matches_energy_difference(
        self,
        archive: Archive,
        plotting_matches: PlottingMatches,
        centroids: np.ndarray,
        minval: np.ndarray,
        maxval: np.ndarray,
        centroids_from_archive: Optional[np.ndarray] = None,
        directory_string: Optional[str] = None,
        filename: Optional[str] = "cvt_energy_diff_matches_from_archive",
        axis_labels: List[str] = ["Band Gap, eV", "Shear Modulus, GPa"],
        annotate: bool = True,
        fitness_limits: Optional[Tuple[float, float]] = None,
        x_axis_limits: Optional[Tuple[float, float]] = None,
        y_axis_limits: Optional[Tuple[float, float]] = None,
    ):
        """Adapted from wdac plot 2d cvt centroids function"""
        # create the plot object
        fig, ax = plt.subplots(facecolor="white", edgecolor="white")

        if len(np.array(minval).shape) == 0 and len(np.array(maxval).shape) == 0:
            ax.set_xlim(minval, maxval)
            ax.set_ylim(minval, maxval)
        else:
            ax.set_xlim(minval[0], maxval[0])
            ax.set_ylim(minval[1], maxval[1])

        ax.set(adjustable="box", aspect="equal")

        # create the regions and vertices from centroids
        regions, vertices = get_voronoi_finite_polygons_2d(centroids)
        # fill the plot with contours
        target_centroid_ids = np.array(self.reference_data.loc["centroid_id"].array)
        target_centroid_energies = np.array(self.reference_data.loc["energy"].array)
        duplicate_centroid_indices = []
        if len(np.unique(np.array(plotting_matches.centroid_indices))) != len(
            plotting_matches.centroid_indices
        ):
            unique, counts = np.unique(
                plotting_matches.centroid_indices, return_counts=True
            )
            duplicate_centroid_indices = np.take(
                unique, np.argwhere(counts != 1)
            ).reshape(-1)

        colour_dict = {
            "no_match": mcolors.CSS4_COLORS["silver"],
            "energy_above_reference": mcolors.CSS4_COLORS["rosybrown"],
            "energy_below_reference": mcolors.CSS4_COLORS["mediumaquamarine"],
            ConfidenceLevels.GOLD.value: mcolors.CSS4_COLORS["mediumpurple"],
        }

        label_dict = {
            ConfidenceLevels.GOLD.value: "Gold Standard",
            "energy_below_reference": "Energy Below Reference",
            "energy_above_reference": "Energy Above Reference",
            "no_match": "Not Accessed",
        }

        for i, region in enumerate(regions):
            polygon = vertices[region]
            ax.fill(
                *zip(*polygon), alpha=0.05, edgecolor="black", facecolor="white", lw=1
            )
            if i in target_centroid_ids:
                ax.fill(*zip(*polygon), edgecolor="gray", facecolor="none", lw=2)
                target_centroid_energy = target_centroid_energies[
                    np.argwhere(target_centroid_ids == i)
                ].reshape(-1)[0]
                if i in archive.centroid_ids:
                    centroid_energy = archive.fitnesses[
                        np.argwhere(np.array(archive.centroid_ids) == i)
                    ].reshape(-1)[0]
                    if centroid_energy < target_centroid_energy:
                        ax.fill(
                            *zip(*polygon),
                            facecolor=colour_dict["energy_below_reference"],
                            label=label_dict["energy_below_reference"],
                        )
                    elif (
                        i in archive.centroid_ids
                        and centroid_energy >= target_centroid_energy
                    ):
                        ax.fill(
                            *zip(*polygon),
                            facecolor=colour_dict["energy_above_reference"],
                            label=label_dict["energy_above_reference"],
                        )
                else:
                    # continue
                    ax.fill(
                        *zip(*polygon),
                        facecolor=colour_dict["no_match"],
                        label=label_dict["no_match"],
                    )

        for list_index, centroid_index in enumerate(plotting_matches.centroid_indices):
            region = regions[centroid_index]
            polygon = vertices[region]

            if centroid_index in duplicate_centroid_indices:
                confidence_levels_ids = np.argwhere(
                    np.array(plotting_matches.centroid_indices) == centroid_index
                ).reshape(-1)
                confidence_scores = [
                    plotting_matches.confidence_levels[int(id)].value
                    for id in confidence_levels_ids
                ]
                max_confidence_id = np.argwhere(
                    np.array(confidence_scores) == max(confidence_scores)
                ).reshape(-1)
                confidence_level = plotting_matches.confidence_levels[
                    int(max_confidence_id[0])
                ]
            else:
                confidence_level = plotting_matches.confidence_levels[list_index]

            if confidence_level == ConfidenceLevels.GOLD:
                ax.fill(
                    *zip(*polygon),
                    alpha=0.8,
                    color=colour_dict[ConfidenceLevels.GOLD.value],
                    label=label_dict[ConfidenceLevels.GOLD.value],
                )
            if annotate:
                ax.annotate(
                    plotting_matches.mp_references[list_index],
                    (centroids[centroid_index, 0], centroids[centroid_index, 1]),
                    fontsize=4,
                )

        self.legend_without_duplicate_labels(fig, ax, list(label_dict.values()))
        ax.set_xlabel(f"{axis_labels[0]}")
        ax.set_ylabel(f"{axis_labels[1]}")

        title = "Energy Comparison"
        ax.set_title(title)
        ax.set_aspect("equal")

        if x_axis_limits is not None and y_axis_limits is not None:
            x_tick_labels = np.linspace(x_axis_limits[0], x_axis_limits[1], 6)
            y_tick_labels = np.linspace(y_axis_limits[0], y_axis_limits[1], 6)
            ax.set_xticklabels([np.around(el, 1) for el in x_tick_labels])
            ax.set_yticklabels([np.around(el, 1) for el in y_tick_labels])

        if directory_string is None:
            fig.show()
        else:
            fig.savefig(
                f"{directory_string}/{filename}_{plotting_matches.plotting_mode.value}.png",
                format="png",
                bbox_inches="tight",
            )
        plt.clf()
        return fig, ax

    def legend_without_duplicate_labels(
        self, fig, ax, sorting_match_list: Optional[List[str]] = None
    ):
        """https://stackoverflow.com/questions/19385639/duplicate-items-in-legend-in-matplotlib"""

        try:
            handles, labels = ax.get_legend_handles_labels()
            unique = [
                (h, l)
                for i, (h, l) in enumerate(zip(handles, labels))
                if l not in labels[:i]
            ]
            if sorting_match_list is None:
                sorting_match_list = np.array(
                    [ConfidenceLevels.get_string(el) for el in list(ConfidenceLevels)]
                )  # todo: get this from ConfidenceLevels Enum
            unique = sorted(
                unique,
                key=lambda x: np.argwhere(np.array(sorting_match_list) == x[1]).reshape(
                    -1
                )[0],
            )
            ax.legend(
                *zip(*unique),
                loc="upper center",
                bbox_to_anchor=(0.5, -0.2),
                fontsize="small",
                ncols=2,
            )
        except Exception as e:
            print("legend error")
            pass
