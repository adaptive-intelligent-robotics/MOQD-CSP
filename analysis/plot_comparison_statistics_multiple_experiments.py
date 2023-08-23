import pathlib

import numpy as np
from matplotlib import pyplot as plt

from csp_elites.map_elites.archive import Archive

def plot_comparison_metrics(experiments_list, plot_labels,  normalise = False):
    plot_title_tag = "DQD OMG Operator vs Benchmark"


    # plot_labels = ["0.05 relax 100 steps","0.05 relax 100 threshold", "0.05 relax whole archive", "lr1", "lr-1", "standard"]

    path_to_experiments = pathlib.Path(__file__).parent.parent / ".experiment.nosync" / "experiments"

    all_experiment_data = []

    for experiment in experiments_list:
        data_for_1_experiment = []

        with open(path_to_experiments / experiment / f"{experiment}.dat", "r") as file:
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
    for metric_id in range(1, len(metric_names)):
        if metric_id in [2, 3, 4, 7, 8]:
            fig, ax = plt.subplots()
            for i, experiment in enumerate(all_experiment_data):
                processed_data = []
                minimum_number_of_datapoints = min([len(el[0]) for el in experiment])
                processed_data.append(experiment[:minimum_number_of_datapoints])
                processed_data = np.array(processed_data)
                all_processed_data.append(processed_data)

                # ci = 1.96 * np.std(processed_data, axis=0) / np.sqrt(len(processed_data[0]))
                # ci = np.std(processed_data, axis=0)
                # means = np.mean(processed_data, axis=0)
                #
                if normalise:
                    data = (processed_data[0][0][metric_id] - np.min(processed_data[0][0][metric_id])) / (np.max(processed_data[0][0][metric_id]) - np.min(processed_data[0][0][metric_id]))
                else:
                    data = processed_data[0][0][metric_id]
                ax.plot(processed_data[0][0][0], data, label=plot_labels[i])
                # ax.fill_between(processed_data[0, 0], (means[metric_id] - ci[metric_id]), (means[metric_id] + ci[metric_id]),
                #                  alpha=.1)
                ax.set_xlabel("Evaluation Count")
                ax.set_ylabel(metric_names[metric_id])
                ax.set_title(f"{metric_names[metric_id]} {plot_title_tag}")
            fig.legend(loc="lower right")
            fig.show()
        # fig.savefig(path_to_experiments / "20230813_00_00_200_niches_for_benchmark_100_relax_combined" / f"comp_with_threshold_{metric_names[metric_id]}.png")


if __name__ == '__main__':
    normalise = False
    experiments_list = [
        # "20230820_11_25_TiO2_500k_lr-5_10_relax_steps_always",
        # "20230820_14_36_TiO2_500k_lr-5_30_relax_steps_whole_archive_relax_10_steps_every_5",
        # "20230820_15_53_TiO2_500k_lr-5_100_relax_steps_always_more_init_smaller_batch",
        # "20230820_17_43_TiO2_5k_lr05_100_relax_steps_always_more_init_smaller_batch",
        # "20230821_00_58_TiO2_500k_lr05_100_relax_steps_always",
        # "20230821_11_30_TiO2_500k_lr05_100_relax_steps_always_with_force_threshold",
        # "20230820_18_20_TiO2_5k_dqd_lr05_10_steps_relax_and_50_every_5_generations",
        "20230821_18_36_TiO2_500k_lr1_100_relax_steps_always_new_normalisation",
        "20230821_23_48_TiO2_500k_lr1_100_relax_steps_always_new_normalisation_batch_10",
        "20230821_23_42_TiO2_500k_lr1_50_relax_steps_always_new_norm",
        "20230822_00_32_TiO2_500k_lr1_100_relax_steps_always_new_normalisation_cma_mega",
        "20230822_10_19_TiO2_500k_lr1_100_relax_steps_always_new_normalisation_cma_mega_batch_10",
        "20230813_01_48_TiO2_200_niches_for benchmark_100_relax_2",

            # "20230820_00_23_TiO2_500k_lr-3",
            # "20230820_00_23_TiO2_500k_lr-4",
            # "20230820_00_39_TiO2_500k_lr-5",
            # "20230820_00_39_TiO2_500k_lr-6",
    ]

    plot_labels = ["lr1 new norm", "new norm batch 10", "50 steps",
                   "cma mega att 1", "cma_mega batch 10", "standard"]
    plot_comparison_metrics(experiments_list, plot_labels)
