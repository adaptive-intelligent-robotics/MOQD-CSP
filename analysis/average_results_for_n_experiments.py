import pathlib

import numpy as np
from matplotlib import pyplot as plt

from csp_elites.map_elites.archive import Archive

if __name__ == '__main__':

    experiments_list = [
        [
            "20230813_01_48_TiO2_200_niches_for benchmark_100_relax_2",
            "20230813_01_49_TiO2_200_niches_for benchmark_100_relax_3",
            "20230813_01_49_TiO2_200_niches_for benchmark_100_relax_4",
            "20230813_01_49_TiO2_200_niches_for benchmark_100_relax_5",
            "20230813_12_14_TiO2_200_niches_for benchmark_100_relax_1",
        ],
        [
            "20230813_01_53_TiO2_200_niches_for benchmark_100_relax_1_with_threshold",
            "20230813_01_53_TiO2_200_niches_for benchmark_100_relax_2_with_threshold",
            "20230813_01_53_TiO2_200_niches_for benchmark_100_relax_5_with_threshold",
            "20230813_12_14_TiO2_200_niches_for benchmark_100_relax_3_with_threshold"
        ]

    ]

    plot_title_tag = "Baseline Comparison With and Without Threshold"
    plot_labels = ["No threshold", "With threshold"]

    centroids_list = [
        "centroids_200_2_band_gap_0_100_shear_modulus_0_120.dat",
    ] * len(experiments_list)

    path_to_experiments = pathlib.Path(__file__).parent.parent / ".experiment.nosync" / "experiments"

    all_experiment_data = []

    for experiment in experiments_list:
        data_for_1_experiment = []
        for experiment_name in experiment:
            with open(path_to_experiments / experiment_name / f"{experiment_name}.dat", "r") as file:
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
        fig, ax = plt.subplots()
        for i, experiment in enumerate(all_experiment_data):
            processed_data = []
            minimum_number_of_datapoints = min([len(el[0]) for el in experiment])
            for el in experiment:
                processed_data.append(el[:, :minimum_number_of_datapoints])
            processed_data = np.array(processed_data)
            all_processed_data.append(processed_data)

            # ci = 1.96 * np.std(processed_data, axis=0) / np.sqrt(len(processed_data[0]))
            ci = np.std(processed_data, axis=0)
            means = np.mean(processed_data, axis=0)

            ax.plot(processed_data[0, 0], means[metric_id], label=plot_labels[i])
            ax.fill_between(processed_data[0, 0], (means[metric_id] - ci[metric_id]), (means[metric_id] + ci[metric_id]),
                             alpha=.1)
            ax.set_xlabel("Evaluation Count")
            ax.set_ylabel(metric_names[metric_id])
            ax.set_title(f"{metric_names[metric_id]} {plot_title_tag}")
        fig.legend()
        fig.savefig(path_to_experiments / "20230813_00_00_200_niches_for_benchmark_100_relax_combined" / f"comp_with_threshold_{metric_names[metric_id]}.png")





    print()
