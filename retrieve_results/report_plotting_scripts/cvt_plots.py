import os
import pathlib

from tqdm import tqdm

from retrieve_results.report_plot_generator import ReportPlotGenerator

if __name__ == '__main__':
    path_to_all_experiments = pathlib.Path(
                __file__).parent.parent.parent / ".experiment.nosync/report_data"
    all_experiments = [name for name in os.listdir(f"{path_to_all_experiments }")
                         if os.path.isdir(path_to_all_experiments / name) and (name != "all_plots")]

    for experiment in tqdm(all_experiments):
        report_generator = ReportPlotGenerator(
            path_to_experiments=path_to_all_experiments/ experiment,
            plot_labels=None,
            title_tag=None,
        )
        report_generator.plot_cvt_and_symmetry(override_fitness_values=[8.7, 9.5], force_replot=True)
