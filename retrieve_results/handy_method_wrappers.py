from typing import Optional, Tuple

import numpy as np

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
    # experiment_tag = experiment_folder.split(f"{formula}_")[1]
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
    experiment_processor.process_symmetry()
