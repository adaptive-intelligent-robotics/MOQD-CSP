import json
import os
import pathlib
import pickle
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from csp_elites.map_elites.elites_utils import __centroids_filename as make_centroid_filename


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

        if folder_name == "20230813_23_55_TiO2_constrained_qd_bg_shear":
            bd_minimum_values, bd_maximum_values = experiment_parameters["bd_minimum_values"], \
                                                   experiment_parameters["bd_maximum_values"]
        else:
            if experiment_parameters["normalise_bd"]:
                bd_minimum_values, bd_maximum_values = [0, 0], [1, 1]
            else:
                bd_minimum_values, bd_maximum_values = experiment_parameters["bd_minimum_values"], experiment_parameters["bd_maximum_values"]

        centroid_filename = make_centroid_filename(
            k=200, # todo: make number of niches dynamic
            dim=len(experiment_parameters["behavioural_descriptors"]),
            bd_names=experiment_parameters["behavioural_descriptors"],
            bd_minimum_values=bd_minimum_values,
            bd_maximum_values=bd_maximum_values,
            formula=self.get_formula_from_folder_name(folder_name=folder_name)
        )
        return centroid_filename

    @staticmethod
    def get_formula_from_folder_name(folder_name: str):
        folder_name = folder_name.split("/")[-1]
        return folder_name[15:].split("_")[0]

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
                try:
                    with open(config_folders[id] / config, "r") as file:
                        config_data = json.load(file)
                except UnicodeDecodeError:
                    continue
                config_tag = config_data["experiment_tag"]
                config_mapping_dict[f"{days[id]}/{config}"] = config_tag
                config_data["config_filename"] = f"{days[id]}/{config}"
                config_for_csv[config_tag] = config_data

        return config_mapping_dict, config_for_csv

    def map_config_data_to_experiment(self, experiments: List[str], config_for_csv: Dict[str, Any], tag):
        mapped_configs = {}
        for experiment_folder in experiments:
            formula = self.get_formula_from_folder_name(experiment_folder)
            # experiment_tag = experiment_folder.split(f"{formula}_")[1]
            experiment_tag = experiment_folder[15 +len(formula) + 1:]
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


    def csv_with_archive_count(self, dates: List[str]):
        all_folders = []
        for date in dates:
            all_folders += self.get_all_folders_with_date(date)

        experiment_tags = []
        maximum_archives = []
        for folder in all_folders:
            exp_id = folder.split("_")[-1]

            experiment_tags.append(folder[15:].rstrip(exp_id))
            if self.is_experiment_valid(folder):
                max_archive = max([int(name.lstrip("archive_").rstrip(".pkl")) for name in os.listdir(self.experiment_directory_path / folder) if ((not os.path.isdir(name)) and ("archive_" in name) and (".pkl" in name))])
            else:
                max_archive = 0
            maximum_archives.append(max_archive)

        all_data = np.vstack([all_folders, experiment_tags, maximum_archives]).T
        df = pd.DataFrame(all_data)
        df.columns = ["folder", "tag", "archives"]
        df.to_csv(self.experiment_directory_path.parent / "evaluations_per_folder.csv")
        pass

def load_config_list_into_csv(date: str):
    organiser = ExperimentOrganiser()
    _, config_for_csv = organiser.get_config_data(date=date)

    df = pd.DataFrame(config_for_csv)
    df = df.transpose()
    df = pd.concat([df, df["cvt_run_parameters"].apply(pd.Series)], axis=1)
    df.drop(columns="cvt_run_parameters", inplace=True)
    temp_cols = df.columns.tolist()
    index = df.columns.get_loc("config_filename")
    new_cols = temp_cols[index:index + 1] + temp_cols[:index] + temp_cols[index + 1:]
    df = df[new_cols]

    # today = date.today()
    df.to_csv(
        pathlib.Path(__file__).parent.parent / ".experiment.nosync" / f"{date}_list_of_configs_no_exp_mapping.csv")
