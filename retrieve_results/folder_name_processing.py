import json
import os
import pathlib
import pickle
from datetime import date
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from ase.ga.utilities import CellBounds

from csp_elites.crystal.materials_data_model import MaterialProperties, StartGenerators
from csp_elites.crystal.symmetry_evaluator import SymmetryEvaluation
from csp_elites.map_elites.archive import Archive
from csp_elites.map_elites.elites_utils import __centroids_filename as make_centroid_filename
from csp_elites.utils import experiment_parameters
from csp_elites.utils.archive_evolution_gif import plot_gif
from csp_elites.utils.asign_target_values_to_centroids import \
    reassign_data_from_pkl_to_new_centroids
from csp_elites.utils.experiment_parameters import ExperimentParameters
from csp_elites.utils.get_mpi_structures import get_all_materials_with_formula
from csp_elites.utils.plot import load_centroids, load_archive_from_pickle, \
    plot_all_maps_in_archive, plot_all_statistics_from_file


class ExperimentOrganiser:
    def __init__(self):
        self.repo_location = pathlib.Path(__file__).parent.parent
        self.experiment_location = pathlib.Path(__file__).parent.parent / ".experiment.nosync"
        self.experiment_directory_path = self.experiment_location / "experiments"

    def get_all_folders_with_date(self, date: str) -> List[str]:
        list_of_files = [name for name in os.listdir(f"{self.experiment_directory_path}")
                         if os.path.isdir(self.experiment_directory_path / name)]

        folders_to_check = []
        for folder_name in list_of_files:
            if date in folder_name:
                folders_to_check.append(folder_name)
        return folders_to_check

    def is_experiment_valid(self, folder_name: str):
        list_of_archives = [name for name in os.listdir(f"{self.experiment_directory_path / folder_name}")
                            if not os.path.isdir(self.experiment_directory_path / folder_name / name) and ("archive_" in name) and (".pkl" in name)]
        if not list_of_archives:
            return False
        else:
            return True

    def get_centroid_name(self, folder_name: str):
        with open(self.experiment_directory_path / folder_name / "experiment_parameters.pkl", "rb") as file:
            experiment_parameters = pickle.load(file)

        centroid_filename = make_centroid_filename(
            k=200, # make number of niches dynamic
            dim=len(experiment_parameters["behavioural_descriptors"]),
            bd_names=experiment_parameters["behavioural_descriptors"],
            bd_minimum_values=experiment_parameters["bd_minimum_values"],
            bd_maximum_values=experiment_parameters["bd_maximum_values"],
            formula="TiO2" # todo: make this dynamic
        )
        return centroid_filename

    def get_config_data(self, date: str):
        # date = self.get_date_from_folder_name(folder_name)
        config_folder_1 = self.repo_location / "configs" / date
        config_files_1 = [name for name in os.listdir(config_folder_1)
                         if not os.path.isdir(config_folder_1 / name)]
        next_day = date[:2] + str(int(date[-2:]) + 1)
        if os.path.isdir(self.repo_location / "configs" / next_day):
            config_folder_2 = self.repo_location / "configs" / next_day

            config_files_2 = [name for name in os.listdir(config_folder_2)
                             if not os.path.isdir(config_folder_2 / name)]
        else:
            config_folder_2 = []
            config_files_2 = []
        days = [date, next_day]
        config_files = [config_files_1, config_files_2]
        config_folders = [config_folder_1, config_folder_2]
        config_mapping_dict = {}
        config_for_csv = {}
        for id, configs_for_day in enumerate(config_files):
            for config in configs_for_day:
                with open(config_folders[id] / config, "r") as file:
                    config_data = json.load(file)
                config_tag = config_data["experiment_tag"]
                config_mapping_dict[f"{days[id]}/{config}"] = config_tag
                config_data["config_filename"] = f"{days[id]}/{config}"
                config_for_csv[config_tag] = config_data

        return config_mapping_dict, config_for_csv

    def map_config_data_to_experiment(self, experiments: List[str], config_for_csv: Dict[str, Any], tag):
        mapped_configs = {}
        for experiment_folder in experiments:
            experiment_tag = experiment_folder.split("TiO2_")[1]
            config_information = config_for_csv[experiment_tag]
            mapped_configs[experiment_folder] = config_information

        df = pd.DataFrame(mapped_configs)
        df = df.transpose()
        df = pd.concat([df, df["cvt_run_parameters"].apply(pd.Series)], axis=1)
        df.drop(columns="cvt_run_parameters", inplace=True)
        temp_cols = df.columns.tolist()
        index = df.columns.get_loc("config_filename")
        new_cols = temp_cols[index:index + 1] + temp_cols[:index] + temp_cols[index+1:]
        df = df[new_cols]

        # today = date.today()
        df.to_csv(pathlib.Path(__file__).parent.parent / ".experiment.nosync" / f"{tag}_list_of_configs.csv")

        pass

    @staticmethod
    def get_date_from_folder_name(folder_name: str):
        return folder_name[4:8]

    def is_done(self, folder_name: str):
        files = [name for name in
                            os.listdir(f"{self.experiment_directory_path / folder_name}")
                            if not os.path.isdir(
                self.experiment_directory_path / folder_name / name)]
        plot_done = "cvt_plot_gif.gif" in files
        symmetry_done = "ind_executive_summary.csv" in files
        return plot_done, symmetry_done


class ExperimentProcessor:
    def __init__(self,
        experiment_label: str,
        centroid_filename: str,
        config_filepath: pathlib.Path,
        fitness_limits=(6.5, 10),
        save_structure_images: bool = True,
        filter_for_experimental_structures: bool = False,
    ):
        self.repo_location = pathlib.Path(__file__).parent.parent
        self.experiment_label = experiment_label
        self.experiment_location = pathlib.Path(__file__).parent.parent / ".experiment.nosync"
        self.experiment_directory_path = self.experiment_location / "experiments" / experiment_label
        self.centroid_directory_path = self.experiment_location / "experiments" / centroid_filename[1:]
        self.all_centroids = load_centroids(str(self.centroid_directory_path))
        self.experiment_parameters = self._load_experiment_parameters(config_filepath)
        self.fitness_limits = fitness_limits
        self.save_structure_images = save_structure_images
        self.filter_for_experimental_structures = filter_for_experimental_structures

    @staticmethod
    def _load_experiment_parameters(file_location: pathlib.Path):
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

        return experiment_parameters

    def plot(self):
        plot_all_maps_in_archive(
            experiment_directory_path=str(self.experiment_directory_path),
            experiment_parameters=self.experiment_parameters,
            all_centroids=self.all_centroids,
            target_centroids=self.compute_target_centroids(),
        )
        plot_gif(experiment_directory_path=str(self.experiment_directory_path))
        plot_all_statistics_from_file(
            filename=f"{self.experiment_directory_path}/{self.experiment_label}.dat",
            save_location=f"{self.experiment_directory_path}/",
        )

    def _get_last_archive_number(self):
        return max([int(name.lstrip("archive_").rstrip(".pkl")) for name in os.listdir(self.experiment_directory_path) if ((not os.path.isdir(name)) and ("archive_" in name))])

    def compute_target_centroids(self):
        comparison_data_location = self.experiment_location / "experiments" / "target_data/ti02_band_gap_shear_modulus.pkl"
        comparison_data_packed = load_archive_from_pickle(str(comparison_data_location))
        target_centroids = reassign_data_from_pkl_to_new_centroids(
            centroids_file=str(self.centroid_directory_path),
            target_data=comparison_data_packed,
            filter_for_number_of_atoms=self.experiment_parameters.fitler_comparison_data_for_n_atoms
        )
        return target_centroids

    def get_material_project_info(self):
        structure_info, known_structures = get_all_materials_with_formula(experiment_parameters.system_name)
        return structure_info, known_structures

    def process_symmetry(self):
        archive_number = self._get_last_archive_number()
        unrelaxed_archive_location = self.experiment_directory_path / f"archive_{archive_number}.pkl"

        archive = Archive.from_archive(unrelaxed_archive_location, centroid_filepath=self.centroid_directory_path)
        symmetry_evaluation = SymmetryEvaluation(
            filter_for_experimental_structures=self.filter_for_experimental_structures,
        )

        matched_space_group_dict, spacegroup_dictionary = symmetry_evaluation.find_individuals_with_reference_symmetries(
            individuals=archive.individuals, indices_to_check=None)

        symmetry_evaluation.plot_histogram(
            spacegroup_dictionary=spacegroup_dictionary,
            against_reference=False,
            save_directory=self.experiment_directory_path,
        )

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

        centroid_tag = str(self.centroid_directory_path.name).rstrip(".dat")
        target_data_path = self.experiment_location / "experiments" / "target_data" / f"target_data_{centroid_tag}.csv"
        if not os.path.isfile(target_data_path):
            target_data_path = None

        symmetry_evaluation.executive_summary_csv(
            archive=archive,
            indices_to_compare=list(all_individual_indices_to_check),
            directory_to_save=self.experiment_directory_path,
            reference_data_path=target_data_path,
        )



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

    # folders_done = []
    # manual_check = {}
    # for folder in folder_list:
    #     print(folder)
    #     if experiment_organiser.is_experiment_valid(folder):
    #         centroid_name = experiment_organiser.get_centroid_name(folder)
    #         experiment_tag = folder.split("TiO2_")[1]
    #         config_match_index = np.argwhere(np.array(experiment_tags_list) == experiment_tag).reshape(-1)
    #         if len(config_match_index) == 0:
    #             manual_check[folder] = "no matching experiment tag in configs"
    #             continue
    #         elif len(config_match_index) > 1:
    #             files_in_experiment = [name for name in os.listdir(f"{experiment_organiser.experiment_directory_path / folder}") if name == "config.json"]
    #             if files_in_experiment:
    #                 config_filepath = experiment_organiser.experiment_directory_path / folder / files_in_experiment[0]
    #             else:
    #                 manual_check[folder] = "multiple matching experiment tags in configs"
    #                 continue
    #         else:
    #             config_filepath = experiment_organiser.repo_location / "configs" / config_names[config_match_index[0]]
    #
    #         plotting_done, symmetry_summary_done = experiment_organiser.is_done(folder)
    #
    #         if plotting_done and symmetry_summary_done:
    #             continue
    #         else:
    #             print(folder)
    #             centroid_filename = experiment_organiser.get_centroid_name(folder)
    #             experiment_processor = ExperimentProcessor(
    #                 experiment_label=folder,
    #                 config_filepath=config_filepath,
    #                 centroid_filename=centroid_filename,
    #                 fitness_limits=(6.5, 10),
    #                 save_structure_images=save_structure_images,
    #                 filter_for_experimental_structures=filter_for_experimental_structures,
    #             )
    #             if not plotting_done:
    #                 experiment_processor.plot()
    #             if not symmetry_summary_done:
    #                 experiment_processor.process_symmetry()
    #
    # print(manual_check)
