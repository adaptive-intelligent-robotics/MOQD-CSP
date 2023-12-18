import os
import pathlib

from tqdm import tqdm

from retrieve_results.report_plot_generator import ReportPlotGenerator

if __name__ == "__main__":
    path_to_all_experiments = (
        pathlib.Path(__file__).parent.parent.parent / ".experiment.nosync/report_data"
    )
    all_experiments = ["7_benchmark_other_materials"]
    fitness_values = [[7, 9.5]]

    for i, experiment in tqdm(enumerate(all_experiments)):
        report_generator = ReportPlotGenerator(
            path_to_experiments=path_to_all_experiments / experiment,
            plot_labels=None,
            title_tag=None,
        )
        report_generator.plot_cvt_and_symmetry(
            override_fitness_values=fitness_values[i],
            force_replot=False,
            all_sub_experiments=True,
            plot_cvt=False,
        )
