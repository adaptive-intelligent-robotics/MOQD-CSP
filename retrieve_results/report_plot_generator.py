import os
import pathlib
from typing import List, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from retrieve_results.experiment_organiser import ExperimentOrganiser

import scienceplots

from retrieve_results.experiment_processing import ExperimentProcessor

plt.style.use('science')
plt.rcParams['savefig.dpi'] = 500

class ReportPlotGenerator:
    def __init__(self, path_to_experiments: pathlib.Path, plot_labels: Optional[List[str]], title_tag: Optional[str]):
        self.experiment_organiser = ExperimentOrganiser()
        self.experiment_organiser.experiment_directory_path = path_to_experiments
        self._path_to_all_experiments = path_to_experiments
        self.experiment_list = [name for name in os.listdir(f"{self._path_to_all_experiments}")
                         if os.path.isdir(self._path_to_all_experiments / name) and (name != "all_plots")]
        self.sub_experiment_list, self.centroids_list, self.configs_list = self._list_all_individual_experiments()
        self.plot_labels = plot_labels
        self.title_tag = title_tag
        self.summary_plot_folder = self._path_to_all_experiments / "all_plots"
        self.summary_plot_folder.mkdir(exist_ok=True)


    def _list_all_individual_experiments(self):
        sub_experiments = []
        centroids_list = []
        configs_list = []
        all_configs = pd.read_csv(pathlib.Path(__file__).parent.parent / "configs/list_of_configs.csv")
        experiment_tag_column = all_configs.columns.get_loc("experiment_tag")
        config_address_column = all_configs.columns.get_loc("Unnamed: 0")
        all_configs = all_configs.transpose().to_numpy()
        for experiment in self.experiment_list:
            path_to_experiment = self._path_to_all_experiments / experiment
            sub_experiments_by_exp = [name for name in os.listdir(f"{path_to_experiment}")
                                      if os.path.isdir(path_to_experiment / name)]
            sub_experiments.append(sub_experiments_by_exp)
            centroid_list_by_exp = []
            config_list_by_exp = []
            for sub_experiment in sub_experiments_by_exp:
                centroid_list_by_exp.append(self.experiment_organiser.get_centroid_name(f"{experiment}/{sub_experiment}")[1:])
                formula = self.experiment_organiser.get_formula_from_folder_name(sub_experiment)
                # experiment_tag = experiment_folder.split(f"{formula}_")[1]
                experiment_tag = sub_experiment[15 + len(formula) + 1:]

                config_match_index = np.argwhere(all_configs[experiment_tag_column] == experiment_tag).reshape(-1)

                if len(config_match_index) > 1:
                    dates = [int(all_configs[config_address_column, match_id][2:4]) for match_id in config_match_index]
                    latest_config = np.argwhere(np.array(dates) == max(dates)).reshape(-1)
                    config_match_index = config_match_index[latest_config][0]
                else:
                    config_match_index = config_match_index[0]
                config_name = str(all_configs[config_address_column, config_match_index])
                config_list_by_exp.append(config_name)
            configs_list.append(config_list_by_exp)
            centroids_list.append(centroid_list_by_exp)

        return sub_experiments, centroids_list, configs_list

    def plot_mean_statistics(
        self, folder_names: Optional[List[str]] = None, labels: Optional[List[str]]=None,
        title_tag: Optional[str] = None, filename_tag: str = "", plot_individually: bool = True,
        reference_path: Optional[pathlib.Path] = None, reference_label: str = ""
    ):
        folders_to_plot = folder_names if folder_names is not None else self.experiment_list
        plot_labels = labels if labels is not None else self.plot_labels
        title_tag = title_tag if title_tag is not None else self.title_tag

        all_experiment_data = []

        for experiment in tqdm(folders_to_plot):
            i = np.argwhere(np.array(self.experiment_list) == experiment).reshape(-1)[0]
            data_for_1_experiment = []
            for experiment_name in self.sub_experiment_list[i]:
                with open(self._path_to_all_experiments/ experiment / experiment_name / f"{experiment_name}.dat",
                          "r") as file:
                    generation_data = np.loadtxt(file)

                generation_data = generation_data.T
                data_for_1_experiment.append(generation_data)

            all_experiment_data.append(data_for_1_experiment)

        metric_names = ["Evaluation number",
                        "Archive size",
                        "Maximum Fitness",
                        "Mean Fitness",
                        "Median Fitness",
                        "Fitness 5th Percentile",
                        "Fitness 95th Percentile",
                        "Coverage",
                        "QD score"
                        ]

        all_processed_data = []

        if reference_path is not None:
            ref_means, ref_quartile_25, ref_quartile_75 = self.load_reference_data(reference_path)
            ref_color = "#BA0079"


        for metric_id in tqdm(range(1, len(metric_names))):
            fig, ax = plt.subplots()
            reference_added = False
            for i, experiment in enumerate(all_experiment_data):
                if reference_path is not None and not reference_added:
                    ax.plot(ref_means[0], ref_means[metric_id], label=reference_label, color=ref_color)
                    ax.fill_between(ref_means[0], (ref_quartile_25[metric_id]),
                                    (ref_quartile_75[metric_id]), alpha=.1, color=ref_color)
                    reference_added = True

                processed_data = []
                minimum_number_of_datapoints = min([len(el[0]) for el in experiment])
                for el in experiment:
                    processed_data.append(el[:, :minimum_number_of_datapoints])
                processed_data = np.array(processed_data)
                all_processed_data.append(processed_data)

                quartile_25 = np.percentile(processed_data, 25, axis=0)
                quartile_75 = np.percentile(processed_data, 75, axis=0)
                means = np.mean(processed_data, axis=0)

                ax.plot(processed_data[0, 0], means[metric_id], label=plot_labels[i])
                ax.fill_between(processed_data[0, 0], (quartile_25[metric_id]),(quartile_75[metric_id]), alpha=.1)
                ax.set_xlabel("Evaluation Count")
                ax.set_ylabel(metric_names[metric_id])
                ax.set_title(f"{title_tag} - {metric_names[metric_id]}")

                if plot_individually:
                    fig_ind, ax_ind = plt.subplots()
                    if reference_path is not None:
                        ax_ind.plot(ref_means[0], ref_means[metric_id], label=reference_label,
                                color=ref_color)
                        ax_ind.fill_between(ref_means[0], (ref_quartile_25[metric_id]),
                                        (ref_quartile_75[metric_id]), alpha=.1, color=ref_color)

                    ax_ind.plot(processed_data[0, 0], means[metric_id], label=plot_labels[i])
                    ax_ind.fill_between(processed_data[0, 0], (quartile_25[metric_id]),(quartile_75[metric_id]), alpha=.1)

                    ax_ind.set_xlabel("Evaluation Count")
                    ax_ind.set_ylabel(metric_names[metric_id])
                    ax_ind.set_title(f"{plot_labels[i]} {metric_names[metric_id]}")
                    ax_ind.legend()
                    ax_ind.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2),
                              fontsize="x-small", ncols=1 + int(reference_path is not None))
                    fig_ind.tight_layout()
                    save_name = f"{plot_labels[i]}_{metric_names[metric_id]}".replace(" ", "_").lower().replace(".", "").replace("/", "")
                    fig_ind.savefig(
                        self.summary_plot_folder / f"{save_name}.png")
                    plt.close(fig_ind)

            plot_labels
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2),
                          fontsize="x-small", ncols=min(3, len(plot_labels)+ int(reference_added)))
            fig.tight_layout()
            save_name = f"{filename_tag}_comparison_{metric_names[metric_id]}".replace(" ", "_").lower().replace(".", "").replace("/", "")
            fig.savefig(self.summary_plot_folder / f"{save_name}.png")
        plt.close(fig)

    def plot_cvt_and_symmetry(self, override_fitness_values: Optional[List[float]] = None, annotate=False, force_replot=True, all_sub_experiments=True):
        for i, experiment in enumerate(self.experiment_list):
            sub_experiments = self.sub_experiment_list[i] if all_sub_experiments else [self.sub_experiment_list[i][0]]
            for j, sub_experiment in enumerate(sub_experiments):

                experiment_processor = ExperimentProcessor(
                    experiment_label=sub_experiment,
                    config_filepath=pathlib.Path(__file__).parent.parent / f"configs/{self.configs_list[i][j]}.json",
                    centroid_filename=None,
                    fitness_limits=(6.5, 10),
                    save_structure_images=False,
                    filter_for_experimental_structures=False,
                    centroid_directory_path=pathlib.Path(__file__).parent.parent / f".experiment.nosync/experiments/{self.centroids_list[i][j]}",
                  experiment_directory_path=self._path_to_all_experiments / experiment /sub_experiment
                )
                if override_fitness_values is not None:
                    experiment_processor.experiment_parameters.fitness_min_max_values = override_fitness_values
                experiment_processor.plot(annotate=annotate, force_replot=force_replot)
                experiment_processor.process_symmetry(annotate=False)


    def load_reference_data(self, path_to_reference: pathlib.Path):

        sub_experiments_by_exp = [name for name in os.listdir(f"{path_to_reference}")
                                  if os.path.isdir(path_to_reference / name)]
        reference_data = []
        for sub_experiment in sub_experiments_by_exp:
            with open(path_to_reference / sub_experiment /f"{sub_experiment}.dat", "r") as file:
                reference_data.append(np.loadtxt(file).T)

        all_processed_data = []
        minimum_number_of_datapoints = min([len(el[0]) for el in reference_data])
        for experiment in reference_data:
            processed_data = np.array(experiment[:, :minimum_number_of_datapoints])
            all_processed_data.append(processed_data)

        all_processed_data = np.array(all_processed_data)
        quartile_25 = np.percentile(all_processed_data, 25, axis=0)
        quartile_75 = np.percentile(all_processed_data, 75, axis=0)
        means = np.mean(all_processed_data, axis=0)

        return means, quartile_25, quartile_75

    def load_experiment_get_means_and_quartile(self, path_to_experiment):
        pass



if __name__ == '__main__':
    report_generator = ReportPlotGenerator(
        path_to_experiments=pathlib.Path(__file__).parent.parent / ".experiment.nosync/report_data/6_benchmark",
        plot_labels=["No Threshold", "With Threshold"],
        title_tag="Impact of Stability Threshold"
    )
    report_generator.plot_cvt_and_symmetry(override_fitness_values=[8.7, 9.7])
    # report_generator.plot_mean_statistics()
