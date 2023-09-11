import json
import os
import pathlib
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from csp_elites.map_elites.archive import Archive
from retrieve_results.experiment_organiser import ExperimentOrganiser

import scienceplots

from retrieve_results.experiment_processing import ExperimentProcessor

plt.style.use("science")
plt.rcParams["savefig.dpi"] = 500


class ReportPlotGenerator:
    def __init__(
        self,
        path_to_experiments: pathlib.Path,
        plot_labels: Optional[List[str]],
        title_tag: Optional[str],
    ):
        self.experiment_organiser = ExperimentOrganiser()
        self.experiment_organiser.experiment_directory_path = path_to_experiments
        self._path_to_all_experiments = path_to_experiments
        self.experiment_list = [
            name
            for name in os.listdir(f"{self._path_to_all_experiments}")
            if os.path.isdir(self._path_to_all_experiments / name)
            and (name != "all_plots")
        ]
        (
            self.sub_experiment_list,
            self.centroids_list,
            self.configs_list,
        ) = self._list_all_individual_experiments()
        self.plot_labels = plot_labels
        self.title_tag = title_tag
        self.summary_plot_folder = self._path_to_all_experiments / "all_plots"
        self.summary_plot_folder.mkdir(exist_ok=True)

    def _list_all_individual_experiments(self):
        sub_experiments = []
        centroids_list = []
        configs_list = []
        all_configs = pd.read_csv(
            pathlib.Path(__file__).parent.parent / "configs/list_of_configs.csv"
        )
        experiment_tag_column = all_configs.columns.get_loc("experiment_tag")
        config_address_column = all_configs.columns.get_loc("Unnamed: 0")
        all_configs = all_configs.transpose().to_numpy()
        for experiment in self.experiment_list:
            path_to_experiment = self._path_to_all_experiments / experiment
            sub_experiments_by_exp = [
                name
                for name in os.listdir(f"{path_to_experiment}")
                if os.path.isdir(path_to_experiment / name)
            ]
            sub_experiments.append(sub_experiments_by_exp)
            centroid_list_by_exp = []
            config_list_by_exp = []
            for sub_experiment in sub_experiments_by_exp:
                centroid_list_by_exp.append(
                    self.experiment_organiser.get_centroid_name(
                        f"{experiment}/{sub_experiment}"
                    )[1:]
                )
                formula = self.experiment_organiser.get_formula_from_folder_name(
                    sub_experiment
                )
                # experiment_tag = experiment_folder.split(f"{formula}_")[1]
                experiment_tag = sub_experiment[15 + len(formula) + 1 :]

                config_match_index = np.argwhere(
                    all_configs[experiment_tag_column] == experiment_tag
                ).reshape(-1)

                if len(config_match_index) > 1:
                    dates = [
                        int(all_configs[config_address_column, match_id][2:4])
                        for match_id in config_match_index
                    ]
                    latest_config = np.argwhere(np.array(dates) == max(dates)).reshape(
                        -1
                    )
                    config_match_index = config_match_index[latest_config][0]
                else:
                    config_match_index = config_match_index[0]
                config_name = str(
                    all_configs[config_address_column, config_match_index]
                )
                config_list_by_exp.append(config_name)
            configs_list.append(config_list_by_exp)
            centroids_list.append(centroid_list_by_exp)

        return sub_experiments, centroids_list, configs_list

    def plot_mean_statistics(
        self,
        folder_names: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        title_tag: Optional[str] = None,
        filename_tag: str = "",
        plot_individually: bool = True,
        reference_path: Optional[pathlib.Path] = None,
        reference_label: str = "",
        y_limits_dict: Optional[Dict[str, float]] = None,
        filter_top_value: Optional[int] = None,
        filter_coverage_for_valid_solutions_only: bool = True,
    ):
        folders_to_plot = (
            folder_names if folder_names is not None else self.experiment_list
        )
        plot_labels = labels if labels is not None else self.plot_labels
        title_tag = title_tag if title_tag is not None else self.title_tag

        all_experiment_data = []

        for experiment in tqdm(folders_to_plot):
            i = np.argwhere(np.array(self.experiment_list) == experiment).reshape(-1)[0]
            data_for_1_experiment = []
            for j, experiment_name in enumerate(self.sub_experiment_list[i]):
                # with open(self._path_to_all_experiments/ experiment / experiment_name / f"{experiment_name}.dat",
                #           "r") as file:
                generation_data = self.compute_metrics_on_experiment(
                    path_to_subexperiment=self._path_to_all_experiments
                    / experiment
                    / experiment_name,
                    number_of_niches=self._number_of_niches_from_centroid_file(
                        self.centroids_list[i][j]
                    ),
                    top_value=None,
                    filter_coverage_for_valid_solutions_only=filter_coverage_for_valid_solutions_only,
                )

                generation_data = generation_data.T
                data_for_1_experiment.append(generation_data)

            all_experiment_data.append(data_for_1_experiment)

        metric_names = [
            "Evaluation number",
            "Archive size",
            "Maximum Fitness",
            "Mean Fitness",
            "Median Fitness",
            "Fitness 5th Percentile",
            "Fitness 95th Percentile",
            "Coverage",
            "QD score",
        ]

        all_processed_data = []

        if reference_path is not None:
            ref_median, ref_quartile_25, ref_quartile_75 = self.load_reference_data(
                reference_path
            )
            ref_color = "#BA0079"

        for metric_id in tqdm(range(1, len(metric_names))):
            fig, ax = plt.subplots()
            reference_added = False
            for i, experiment in enumerate(all_experiment_data):
                if reference_path is not None and not reference_added:
                    ax.plot(
                        ref_median[0],
                        ref_median[metric_id],
                        label=reference_label,
                        color=ref_color,
                    )
                    ax.fill_between(
                        ref_median[0],
                        (ref_quartile_25[metric_id]),
                        (ref_quartile_75[metric_id]),
                        alpha=0.1,
                        color=ref_color,
                    )
                    reference_added = True

                processed_data = []
                minimum_number_of_datapoints = min([len(el[0]) for el in experiment])
                for el in experiment:
                    processed_data.append(el[:, :minimum_number_of_datapoints])
                processed_data = np.array(processed_data)
                all_processed_data.append(processed_data)

                quartile_25 = np.percentile(processed_data, 25, axis=0)
                quartile_75 = np.percentile(processed_data, 75, axis=0)
                median = np.median(processed_data, axis=0)

                ax.plot(processed_data[0, 0], median[metric_id], label=plot_labels[i])
                ax.fill_between(
                    processed_data[0, 0],
                    (quartile_25[metric_id]),
                    (quartile_75[metric_id]),
                    alpha=0.1,
                )
                ax.set_xlabel("Evaluation Count")
                ax.set_ylabel(metric_names[metric_id])
                ax.set_title(f"{title_tag} - {metric_names[metric_id]}")

                if plot_individually:
                    fig_ind, ax_ind = plt.subplots()
                    if reference_path is not None:
                        ax_ind.plot(
                            ref_median[0],
                            ref_median[metric_id],
                            label=reference_label,
                            color=ref_color,
                        )
                        ax_ind.fill_between(
                            ref_median[0],
                            (ref_quartile_25[metric_id]),
                            (ref_quartile_75[metric_id]),
                            alpha=0.1,
                            color=ref_color,
                        )

                    ax_ind.plot(
                        processed_data[0, 0], median[metric_id], label=plot_labels[i]
                    )
                    ax_ind.fill_between(
                        processed_data[0, 0],
                        (quartile_25[metric_id]),
                        (quartile_75[metric_id]),
                        alpha=0.1,
                    )

                    ax_ind.set_xlabel("Evaluation Count")
                    ax_ind.set_ylabel(metric_names[metric_id])
                    ax_ind.set_title(f"{plot_labels[i]} {metric_names[metric_id]}")
                    ax_ind.legend()
                    ax_ind.legend(
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.2),
                        fontsize="x-small",
                        ncols=1 + int(reference_path is not None),
                    )
                    fig_ind.tight_layout()
                    save_name = (
                        f"{plot_labels[i]}_{metric_names[metric_id]}".replace(" ", "_")
                        .lower()
                        .replace(".", "")
                        .replace("/", "")
                    )
                    fig_ind.savefig(self.summary_plot_folder / f"{save_name}.png")
                    plt.close(fig_ind)

            if (
                y_limits_dict is not None
                and metric_names[metric_id] in y_limits_dict.keys()
            ):
                ax.set_ylim(y_limits_dict[metric_names[metric_id]])
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.2),
                fontsize="small",
                ncols=min(2, len(plot_labels) + int(reference_added)),
            )
            fig.tight_layout()
            save_name = (
                f"{filename_tag}_comparison_{metric_names[metric_id]}".replace(" ", "_")
                .lower()
                .replace(".", "")
                .replace("/", "")
            )
            fig.savefig(self.summary_plot_folder / f"{save_name}.png")
        plt.close(fig)

    def plot_combined_figure_for_report(
        self,
        folder_names: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        title_tag: Optional[str] = None,
        filename_tag: str = "",
        plot_individually: bool = True,
        reference_path: Optional[pathlib.Path] = None,
        reference_label: str = "",
        y_limits_dict: Optional[Dict[str, float]] = None,
        filter_top_value: Optional[int] = None,
        filter_coverage_for_valid_solutions_only: bool = True,
    ):
        folders_to_plot = (
            folder_names if folder_names is not None else self.experiment_list
        )
        plot_labels = labels if labels is not None else self.plot_labels
        title_tag = title_tag if title_tag is not None else self.title_tag

        all_experiment_data = []

        for experiment in tqdm(folders_to_plot):
            i = np.argwhere(np.array(self.experiment_list) == experiment).reshape(-1)[0]
            data_for_1_experiment = []
            for j, experiment_name in enumerate(self.sub_experiment_list[i]):
                # with open(self._path_to_all_experiments/ experiment / experiment_name / f"{experiment_name}.dat",
                #           "r") as file:
                generation_data = self.compute_metrics_on_experiment(
                    path_to_subexperiment=self._path_to_all_experiments
                    / experiment
                    / experiment_name,
                    number_of_niches=self._number_of_niches_from_centroid_file(
                        self.centroids_list[i][j]
                    ),
                    top_value=None,
                    filter_coverage_for_valid_solutions_only=filter_coverage_for_valid_solutions_only,
                )

                generation_data = generation_data .T
                data_for_1_experiment.append(generation_data)

            all_experiment_data.append(data_for_1_experiment)

        metric_names = [
            "Evaluation number",
            "Archive size",
            "Maximum Fitness",
            "Mean Fitness",
            "Median Fitness",
            "Fitness 5th Percentile",
            "Fitness 95th Percentile",
            "Coverage",
            "QD score",
        ]

        all_processed_data = []

        if reference_path is not None:
            ref_median, ref_quartile_25, ref_quartile_75 = self.load_reference_data(
                reference_path
            )
            ref_color = "#BA0079"

        fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(8, 2.5))
        # for ax_id in range(len(ax)):
        # mpl.rcParams[]
        for ax_id, metric_id in enumerate([2, -1, -2]):
            for i, experiment in enumerate(all_experiment_data):
                if reference_path is not None:
                    ax[ax_id].plot(
                        ref_median[0],
                        ref_median[metric_id],
                        label=reference_label,
                        color=ref_color,
                    )
                    ax[ax_id].fill_between(
                        ref_median[0],
                        (ref_quartile_25[metric_id]),
                        (ref_quartile_75[metric_id]),
                        alpha=0.1,
                        color=ref_color,
                    )

                processed_data = []
                minimum_number_of_datapoints = min([len(el[0]) for el in experiment])
                for el in experiment:
                    processed_data.append(el[:, :minimum_number_of_datapoints])
                processed_data = np.array(processed_data)
                all_processed_data.append(processed_data)

                quartile_25 = np.percentile(processed_data, 25, axis=0)
                quartile_75 = np.percentile(processed_data, 75, axis=0)
                median = np.median(processed_data, axis=0)

                ax[ax_id].plot(
                    processed_data[0, 0], median[metric_id], label=plot_labels[i]
                )
                ax[ax_id].fill_between(
                    processed_data[0, 0],
                    (quartile_25[metric_id]),
                    (quartile_75[metric_id]),
                    alpha=0.1,
                )
                ax[ax_id].set_xlabel("Evaluation Count")
                ax[ax_id].set_ylabel(metric_names[metric_id])
                ax[ax_id].set_title(f"{metric_names[metric_id]}")
                if (
                    y_limits_dict is not None
                    and metric_names[metric_id] in y_limits_dict.keys()
                ):
                    ax[ax_id].set_ylim(y_limits_dict[metric_names[metric_id]])
        # fig.legend(loc="lower center")

        handles, labels = ax[1].get_legend_handles_labels()
        unique = [
            (h, l)
            for i, (h, l) in enumerate(zip(handles, labels))
            if l not in labels[:i]
        ]
        unique = sorted(
            unique, key=lambda x: np.argwhere(np.array(labels) == x[1]).reshape(-1)[0]
        )
        ax[1].legend(
            *zip(*unique),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.4),
            fontsize="x-small",
            ncols=min(4, len(labels) + int(reference_path is not None)),
        )

        fig.tight_layout()
        plt.subplots_adjust(wspace=0.3)
        # fig.savefig("test.png")
        save_name = (
            f"REPORT_{filename_tag}".replace(" ", "_")
            .lower()
            .replace(".", "")
            .replace("/", "")
        )
        fig.savefig(self.summary_plot_folder / f"{save_name}.png")

    def legend_without_duplicate_labels(
        self, ax, sorting_match_list: Optional[List[str]] = None
    ):
        """https://stackoverflow.com/questions/19385639/duplicate-items-in-legend-in-matplotlib"""
        try:
            handles, labels = ax.get_legend_handles_labels()
            unique = [
                (h, l)
                for i, (h, l) in enumerate(zip(handles, labels))
                if l not in labels[:i]
            ]
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
                fontsize="x-small",
                ncols=2,
            )
        except Exception as e:
            print("legend error")
            pass

    def plot_cvt_and_symmetry(
        self,
        override_fitness_values: Optional[List[float]] = None,
        experiment_list: Optional[List[str]] = None,
        annotate=False,
        force_replot=True,
        all_sub_experiments=True,
        plot_cvt=True,
    ):
        experiment_list = (
            experiment_list if experiment_list is not None else self.experiment_list
        )
        for experiment in experiment_list:
            i = np.argwhere(np.array(self.experiment_list) == experiment).reshape(-1)[0]
            sub_experiments = (
                self.sub_experiment_list[i]
                if all_sub_experiments
                else [self.sub_experiment_list[i][0]]
            )
            for j, sub_experiment in enumerate(sub_experiments):
                experiment_processor = ExperimentProcessor(
                    experiment_label=sub_experiment,
                    config_filepath=pathlib.Path(__file__).parent.parent
                    / f"configs/{self.configs_list[i][j]}.json",
                    centroid_filename=None,
                    fitness_limits=(6.5, 10),
                    save_structure_images=False,
                    filter_for_experimental_structures=False,
                    centroid_directory_path=pathlib.Path(__file__).parent.parent
                    / f".experiment.nosync/experiments/{self.centroids_list[i][j]}",
                    experiment_directory_path=self._path_to_all_experiments
                    / experiment
                    / sub_experiment,
                )
                if override_fitness_values is not None:
                    experiment_processor.experiment_parameters.fitness_min_max_values = (
                        override_fitness_values
                    )
                if plot_cvt:
                    experiment_processor.plot(
                        annotate=annotate, force_replot=force_replot
                    )

                path_to_group = (
                    self._path_to_all_experiments
                    / experiment
                    / sub_experiment
                    / "number_of_groups.json"
                )
                if path_to_group.exists():
                    continue
                else:
                    experiment_processor.process_symmetry(annotate=False)

    def list_all_archives(self, sub_experiment: pathlib.Path):
        list_of_files = [
            name for name in os.listdir(f"{sub_experiment}") if not os.path.isdir(name)
        ]
        list_of_archives = [
            filename
            for filename in list_of_files
            if ("archive_" in filename) and (".pkl" in filename)
        ]
        list_of_archive_ids = [
            int(filename.lstrip("archive_").rstrip(".pkl"))
            for filename in list_of_archives
        ]
        indices_to_sort = np.argsort(list_of_archive_ids)
        list_of_archives = np.take(list_of_archives, indices_to_sort)
        return list_of_archives

    def compute_metrics_on_experiment(
        self,
        path_to_subexperiment: pathlib.Path,
        number_of_niches: int = 200,
        top_value: Optional[int] = None,
        filter_coverage_for_valid_solutions_only: bool = True,
    ):
        archive_strings = self.list_all_archives(path_to_subexperiment)
        all_data = []
        for i, archive_string in enumerate(archive_strings):
            archive = Archive.from_archive(path_to_subexperiment / archive_string)
            evaluation_number = int(archive_string.lstrip("archive_").rstrip(".pkl"))
            number_of_individuals = len(archive.fitnesses)
            fitness_metrics = archive.compute_fitness_metrics(top_value)
            coverage = archive.compute_coverage(
                number_of_niches, top_value, filter_coverage_for_valid_solutions_only
            )
            qd_score = archive.compute_qd_score(top_value)
            one_row = np.hstack(
                [
                    evaluation_number,
                    number_of_individuals,
                    fitness_metrics,
                    coverage,
                    qd_score,
                ]
            )
            all_data.append(one_row)

        return np.array(all_data)

    def load_reference_data(self, path_to_reference: pathlib.Path):
        sub_experiments_by_exp = [
            name
            for name in os.listdir(f"{path_to_reference}")
            if os.path.isdir(path_to_reference / name)
        ]
        # reference_data = []
        # for sub_experiment in sub_experiments_by_exp:
        #     with open(path_to_reference / sub_experiment /f"{sub_experiment}.dat", "r") as file:
        #         reference_data.append(np.loadtxt(file).T)

        data_for_1_experiment = []
        for sub_experiment in sub_experiments_by_exp:
            # with open(self._path_to_all_experiments/ experiment / experiment_name / f"{experiment_name}.dat",
            #           "r") as file:
            generation_data = self.compute_metrics_on_experiment(
                path_to_subexperiment=path_to_reference / sub_experiment,
                number_of_niches=200,
                top_value=None,
                filter_coverage_for_valid_solutions_only=True,
            )

            generation_data = generation_data.T
            data_for_1_experiment.append(generation_data)

        all_processed_data = []
        minimum_number_of_datapoints = min([len(el[0]) for el in data_for_1_experiment])
        for experiment in data_for_1_experiment:
            processed_data = np.array(experiment[:, :minimum_number_of_datapoints])
            all_processed_data.append(processed_data)

        all_processed_data = np.array(all_processed_data)
        quartile_25 = np.percentile(all_processed_data, 25, axis=0)
        quartile_75 = np.percentile(all_processed_data, 75, axis=0)
        means = np.median(all_processed_data, axis=0)

        return means, quartile_25, quartile_75

    def _number_of_niches_from_centroid_file(self, centroid_file: str) -> int:
        return int(centroid_file.split("centroids_")[1].split("_")[0])

    def compute_average_match_metrics(
        self, experiemnts_list: Optional[List[str]] = None
    ):
        experiemnts_list = (
            experiemnts_list if experiemnts_list is not None else self.experiment_list
        )
        for experiment in tqdm(experiemnts_list):
            i = np.argwhere(np.array(self.experiment_list) == experiment).reshape(-1)[0]
            data_for_1_experiment = []
            for j, experiment_name in enumerate(self.sub_experiment_list[i]):
                path_to_stats_file = (
                    self._path_to_all_experiments
                    / experiment
                    / experiment_name
                    / "ind_report_summary.json"
                )
                if path_to_stats_file.exists():
                    with open(path_to_stats_file, "r") as file:
                        results_dict = json.load(file)
                        data_for_1_experiment.append(results_dict)

                else:
                    print(
                        f"{experiment}, {experiment_name} does not have a summary file"
                    )

            df2 = pd.DataFrame(data_for_1_experiment)

            df = pd.DataFrame(data_for_1_experiment)
            df.drop("fooled_ground_state_match", axis="columns", inplace=True)
            df.drop("ground_state_match", axis="columns", inplace=True)

            df3 = pd.DataFrame(
                [
                    df.mean(),
                    df.std(),
                    df2["ground_state_match"].value_counts(),
                    df2["fooled_ground_state_match"].value_counts(),
                    df2[
                        ["ground_state_match", "fooled_ground_state_match"]
                    ].value_counts(),
                ]
            )
            df3.rename(
                index={
                    "Unnamed 0": "mean",
                    "Unnamed 1": "std",
                    "count": "fooled_ground_state",
                    "count": "ground_state",
                }
            )

            df3.to_csv(
                self._path_to_all_experiments / experiment / "match_statistics.csv"
            )

    # def mean_matches_df(self):
    #     for i, experiment in enumerate(self.experiment_list):
    #         sub_experiments = self.sub_experiment_list[i] if all_sub_experiments else [self.sub_experiment_list[i][0]]
    #         for j, sub_experiment in enumerate(sub_experiments):


if __name__ == "__main__":
    report_generator = ReportPlotGenerator(
        path_to_experiments=pathlib.Path(__file__).parent.parent
        / ".experiment.nosync/report_data/6_benchmark",
        plot_labels=["No Threshold", "With Threshold"],
        title_tag="Impact of Stability Threshold",
    )
    report_generator.compute_average_match_metrics()

    # report_generator.plot_cvt_and_symmetry(override_fitness_values=[8.7, 9.7], all_sub_experiments=True)
    # report_generator.plot_mean_statistics()
    #
    # report_generator.compute_metrics_on_experiment(report_generator._path_to_all_experiments / report_generator.experiment_list[0] / report_generator.sub_experiment_list[0][0])
