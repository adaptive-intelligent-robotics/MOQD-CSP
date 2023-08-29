import json
import os
import pathlib
from typing import Union

import numpy as np
from ase.ga.utilities import CellBounds

from csp_elites.crystal.materials_data_model import MaterialProperties, StartGenerators
from csp_elites.crystal.symmetry_evaluator import SymmetryEvaluation
from csp_elites.map_elites.archive import Archive
from csp_elites.utils import experiment_parameters
from csp_elites.utils.archive_evolution_gif import plot_gif
from csp_elites.utils.asign_target_values_to_centroids import \
    reassign_data_from_pkl_to_new_centroids
from csp_elites.utils.experiment_parameters import ExperimentParameters
from csp_elites.utils.get_mpi_structures import get_all_materials_with_formula
from csp_elites.utils.plot import load_centroids, load_archive_from_pickle, \
    plot_all_maps_in_archive, plot_all_statistics_from_file
from retrieve_results.experiment_organiser import ExperimentOrganiser


class ExperimentProcessor:
    def __init__(self,
        experiment_label: str,
        centroid_filename: str,
        config_filepath: Union[pathlib.Path, ExperimentParameters],
        fitness_limits=(6.5, 10),
        save_structure_images: bool = True,
        filter_for_experimental_structures: bool = False,
        experiment_location: pathlib.Path = pathlib.Path(__file__).parent.parent / ".experiment.nosync"
    ):
        self.repo_location = pathlib.Path(__file__).parent.parent
        self.experiment_label = experiment_label
        self.experiment_location = experiment_location
        self.experiment_directory_path = self.experiment_location / "experiments" / experiment_label
        self.centroid_directory_path = self.experiment_location / "experiments" / centroid_filename[1:]
        self.all_centroids = load_centroids(str(self.centroid_directory_path))
        self.experiment_parameters = self._load_experiment_parameters(config_filepath)
        self.fitness_limits = fitness_limits
        self.save_structure_images = save_structure_images
        self.filter_for_experimental_structures = filter_for_experimental_structures
        self.formula = experiment_label[15:].split("_")[0]


    @staticmethod
    def _load_experiment_parameters(file_location: Union[pathlib.Path, ExperimentParameters]):
        if isinstance(file_location, pathlib.Path):
            with open(file_location, "r") as file:
                experiment_parameters = json.load(file)
            experiment_parameters = ExperimentParameters(**experiment_parameters)
            experiment_parameters.cellbounds = CellBounds(
                bounds={'phi': [20, 160], 'chi': [20, 160], 'psi': [20, 160], 'a': [2, 40],
                        'b': [2, 40],
                        'c': [2, 40]}),
            experiment_parameters.splits = {(2,): 1, (4,): 1}
            experiment_parameters.cvt_run_parameters["behavioural_descriptors"] = \
                [MaterialProperties(value) for value in
                 experiment_parameters.cvt_run_parameters["behavioural_descriptors"]]

            experiment_parameters.start_generator = StartGenerators(
                experiment_parameters.start_generator)
        else:
            experiment_parameters = file_location

        return experiment_parameters

    def plot(self, annotate: bool = True):
        plot_all_maps_in_archive(
            experiment_directory_path=str(self.experiment_directory_path),
            experiment_parameters=self.experiment_parameters,
            all_centroids=self.all_centroids,
            target_centroids=self.compute_target_centroids(),
            annotate=annotate
        )
        plot_gif(experiment_directory_path=str(self.experiment_directory_path))
        plot_all_statistics_from_file(
            filename=f"{self.experiment_directory_path}/{self.experiment_label}.dat",
            save_location=f"{self.experiment_directory_path}/",
        )

    def _get_last_archive_number(self):
        return max([int(name.lstrip("archive_").rstrip(".pkl")) for name in os.listdir(self.experiment_directory_path) if ((not os.path.isdir(name)) and ("archive_" in name) and (".pkl" in name))])

    def compute_target_centroids(self):
        number_of_atoms = len(self.experiment_parameters.blocks)
        bd_tag = [bd.value for bd in self.experiment_parameters.cvt_run_parameters ["behavioural_descriptors"]]
        tag = ""
        for el in bd_tag:
            tag += f"{el}_"
        comparison_data_location = self.experiment_location / "mp_reference_analysis" / f"{self.formula}_{number_of_atoms}" / f"{self.formula}_{tag[:-1]}.pkl"

        comparison_data_packed = load_archive_from_pickle(str(comparison_data_location))

        normalise_bd_values = (self.experiment_parameters.cvt_run_parameters["bd_minimum_values"], \
                                                   self.experiment_parameters.cvt_run_parameters["bd_maximum_values"]) if self.experiment_parameters.cvt_run_parameters["normalise_bd"] else None

        target_centroids = reassign_data_from_pkl_to_new_centroids(
            centroids_file=str(self.centroid_directory_path),
            target_data=comparison_data_packed,
            filter_for_number_of_atoms=self.experiment_parameters.fitler_comparison_data_for_n_atoms,
            normalise_bd_values=normalise_bd_values
        )
        return target_centroids

    def get_material_project_info(self):
        structure_info, known_structures = get_all_materials_with_formula(experiment_parameters.system_name)
        return structure_info, known_structures

    def process_symmetry(self):
        archive_number = self._get_last_archive_number()
        unrelaxed_archive_location = self.experiment_directory_path / f"archive_{archive_number}.pkl"

        centroid_tag = str(self.centroid_directory_path.name).rstrip(".dat")
        # target_data_path = self.experiment_location / "experiments" / "target_data" / f"target_data_{centroid_tag}.csv"
        number_of_atoms = self.experiment_parameters.cvt_run_parameters["filter_starting_Structures"]
        target_data_path = self.experiment_location / "mp_reference_analysis" / f"{self.formula}_{number_of_atoms}" / f"{self.formula}_target_data_{centroid_tag}.csv"

        # todo: change to include formula
        if not os.path.isfile(target_data_path):
            target_data_path = None

        archive = Archive.from_archive(unrelaxed_archive_location, centroid_filepath=self.centroid_directory_path)


        normalise_bd_values = (self.experiment_parameters.cvt_run_parameters["bd_minimum_values"], \
                                                   self.experiment_parameters.cvt_run_parameters["bd_maximum_values"]) if self.experiment_parameters.cvt_run_parameters["normalise_bd"] else None



        tareget_archive = Archive.from_reference_csv_path(
            target_data_path,
            normalise_bd_values=normalise_bd_values,
            centroids_path=self.centroid_directory_path,
        )
        symmetry_evaluation = SymmetryEvaluation(
            formula=self.formula,
            filter_for_experimental_structures=self.filter_for_experimental_structures,
            reference_data_archive=tareget_archive
        )

        matched_space_group_dict, spacegroup_dictionary = symmetry_evaluation.find_individuals_with_reference_symmetries(
            individuals=archive.individuals, indices_to_check=None)

        energy_indices = symmetry_evaluation.save_best_structures_by_energy(
            archive=archive,
            fitness_range=(9.1, 9.5),
            top_n_individuals_to_save=10,
            directory_to_save=self.experiment_directory_path,
            save_primitive=False,
            save_visuals=self.save_structure_images,
        )

        symmetry_indices = symmetry_evaluation.save_best_structures_by_symmetry(
            archive=archive,
            matched_space_group_dict=matched_space_group_dict,
            directory_to_save=self.experiment_directory_path,
            save_primitive=False,
            save_visuals=self.save_structure_images
        )

        all_individual_indices_to_check = np.unique(symmetry_indices + energy_indices)

        df, individuals_with_matches = symmetry_evaluation.executive_summary_csv(
            archive=archive,
            indices_to_compare=list(all_individual_indices_to_check),
            directory_to_save=self.experiment_directory_path,
        )

        symmetry_evaluation.group_structures_by_symmetry(
            archive=archive,
            experiment_directory_path=self.experiment_directory_path,
            centroid_full_path=self.centroid_directory_path,
        )
        if individuals_with_matches and target_data_path is not None:
            plotting_from_archive, plotting_from_mp = symmetry_evaluation.matches_for_plotting(
                individuals_with_matches)

            symmetry_evaluation.plot_matches_mapped_to_references(
                plotting_matches=plotting_from_archive,
                centroids=self.all_centroids,
                centroids_from_archive=archive.centroid_ids,
                minval=[0, 0] if self.experiment_parameters.cvt_run_parameters["normalise_bd"] else self.experiment_parameters.cvt_run_parameters["bd_minimum_values"],
                maxval=[1, 1] if self.experiment_parameters.cvt_run_parameters["normalise_bd"] else self.experiment_parameters.cvt_run_parameters["bd_maximum_values"],
                directory_string=str(self.experiment_directory_path),
            )
            symmetry_evaluation.plot_matches_mapped_to_references(
                plotting_matches=plotting_from_mp,
                centroids=self.all_centroids,
                centroids_from_archive=archive.centroid_ids,
                minval=[0, 0] if self.experiment_parameters.cvt_run_parameters["normalise_bd"] else self.experiment_parameters.cvt_run_parameters["bd_minimum_values"],
                maxval=[1, 1] if self.experiment_parameters.cvt_run_parameters["normalise_bd"] else self.experiment_parameters.cvt_run_parameters["bd_maximum_values"],
                directory_string=str(self.experiment_directory_path),
            )
            print("I was here")



if __name__ == '__main__':
    experiment_date = "0822"
    save_structure_images = True
    filter_for_experimental_structures = False

    experiment_organiser = ExperimentOrganiser()
    folder_list = experiment_organiser.get_all_folders_with_date(experiment_date)
    config_mapping, config_dict_csv = experiment_organiser.get_config_data(experiment_date)
    config_names = list(config_mapping.keys())
    experiment_tags_list = list(config_mapping.values())
    experiment_organiser.map_config_data_to_experiment(folder_list, config_dict_csv, experiment_date)
