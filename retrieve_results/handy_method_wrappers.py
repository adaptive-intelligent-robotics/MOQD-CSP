import json
import os
import pathlib
from json import JSONDecodeError
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from retrieve_results.experiment_organiser import ExperimentOrganiser
from retrieve_results.experiment_processing import ExperimentProcessor


def plot_metrics_for_one_folder(folder_name: str, annotate: bool = True,
                                override_fitness_values: Optional[Tuple[int, int]] = None) :

    experiment_organiser = ExperimentOrganiser()
    date = experiment_organiser.get_date_from_folder_name(folder_name)
    config_mapping, config_dict_csv = experiment_organiser.get_config_data(date)
    config_names = list(config_mapping.keys())
    experiment_tags_list = list(config_mapping.values())

    centroid_name = experiment_organiser.get_centroid_name(folder_name)

    formula = experiment_organiser.get_formula_from_folder_name(folder_name)
    experiment_tag = folder_name[15 + len(formula) + 1:]

    config_match_index = np.argwhere(np.array(experiment_tags_list) == experiment_tag).reshape(-1)
    config_filepath = experiment_organiser.repo_location / "configs" / config_names[config_match_index[0]]
    experiment_processor = ExperimentProcessor(
        experiment_label=folder_name,
        config_filepath=config_filepath,
        centroid_filename=centroid_name,
        fitness_limits=(6.5, 10),
        save_structure_images=False,
        filter_for_experimental_structures=False,
    )
    if override_fitness_values is not None:
        experiment_processor.experiment_parameters.fitness_min_max_values = override_fitness_values
    experiment_processor.plot(annotate=annotate)
    # experiment_processor.process_symmetry()



def update_configs_csv(path_to_strip: str = "/Users/marta/Documents/MSc Artificial Intelligence/Thesis/csp-elites/configs/"):
    path_to_configs = pathlib.Path(__file__).parent.parent / "configs"

    sub_folders = [name for name in os.listdir(f"{path_to_configs}")
                         if os.path.isdir(path_to_configs / name)]

    all_configs = []
    for sub_folder in sub_folders:
        path = path_to_configs / sub_folder
        new_configs = [os.path.join(path, o)
                            for o in os.listdir(path) if os.path.isfile(os.path.join(path,o)) and (".pbs" not in o)(".sh" not in o)]
        all_configs += new_configs

    config_names = [config.lstrip(path_to_strip).rstrip(".json") for config in all_configs]

    data = {}

    for i, config in enumerate(all_configs):
        try:
            with open(config, "r") as file:
                config_data = json.load(file)
        except (JSONDecodeError, UnicodeDecodeError) as e:
            continue
        data[config_names[i]] = config_data

    df = pd.DataFrame(data)
    df = df.transpose()
    df = pd.concat([df, df["cvt_run_parameters"].apply(pd.Series)], axis=1)
    df.drop(columns="cvt_run_parameters", inplace=True)
    df.to_csv(path_to_configs / "list_of_configs.csv")
    print()


if __name__ == '__main__':
    update_configs_csv()
