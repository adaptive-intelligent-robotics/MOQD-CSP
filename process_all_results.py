import os

import numpy as np

from retrieve_results.folder_name_processing import ExperimentProcessor, ExperimentOrganiser

if __name__ == '__main__':
    date = "0822"
    save_structure_images = False
    filter_for_experimental_structures = False
    always_run_plotting = True

    experiment_organiser = ExperimentOrganiser()
    folder_list = experiment_organiser.get_all_folders_with_date(date)
    config_mapping, config_dict_csv = experiment_organiser.get_config_data(date)
    config_names = list(config_mapping.keys())
    experiment_tags_list = list(config_mapping.values())
    experiment_organiser.map_config_data_to_experiment(folder_list, config_dict_csv, date)
    folders_done = []
    manual_check = {}
    for folder in folder_list:
        print(folder)
        if experiment_organiser.is_experiment_valid(folder):
            centroid_name = experiment_organiser.get_centroid_name(folder)
            experiment_tag = folder.split("TiO2_")[1]
            config_match_index = np.argwhere(np.array(experiment_tags_list) == experiment_tag).reshape(-1)
            if len(config_match_index) == 0:
                manual_check[folder] = "no matching experiment tag in configs"
                continue
            elif len(config_match_index) > 1:
                files_in_experiment = [name for name in os.listdir(f"{experiment_organiser.experiment_directory_path / folder}") if name == "config.json"]
                if files_in_experiment:
                    config_filepath = experiment_organiser.experiment_directory_path / folder / files_in_experiment[0]
                else:
                    manual_check[folder] = "multiple matching experiment tags in configs"
                    continue
            else:
                config_filepath = experiment_organiser.repo_location / "configs" / config_names[config_match_index[0]]

            if always_run_plotting:
                plotting_done, symmetry_summary_done = False, False
            else:
                plotting_done, symmetry_summary_done = experiment_organiser.is_done(folder)

            if plotting_done and symmetry_summary_done:
                continue
            else:
                centroid_filename = experiment_organiser.get_centroid_name(folder)
                experiment_processor = ExperimentProcessor(
                    experiment_label=folder,
                    config_filepath=config_filepath,
                    centroid_filename=centroid_filename,
                    fitness_limits=(6.5, 10),
                    save_structure_images=save_structure_images,
                    filter_for_experimental_structures=filter_for_experimental_structures,
                )
                if not plotting_done:
                    try:
                        experiment_processor.plot()
                    except ValueError:
                        print(f"problem with plotting folder {folder}")
                        continue
                if not symmetry_summary_done:
                    try:
                        experiment_processor.process_symmetry()
                    except ValueError:
                        print(f"problem with plotting folder {folder}")

    print(manual_check)
