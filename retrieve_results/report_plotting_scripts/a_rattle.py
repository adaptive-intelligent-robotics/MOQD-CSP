import pathlib

from tqdm import tqdm

from retrieve_results.report_plot_generator import ReportPlotGenerator


def no_relax(plot_individually=False, exp_1=True):
    report_generator = ReportPlotGenerator(
        path_to_experiments=pathlib.Path(
            __file__).parent.parent.parent / ".experiment.nosync/report_data/0_no_relax",
        plot_labels=None,
        title_tag=None
    )
    if exp_1:
        report_generator.plot_combined_figure_for_report(
            folder_names=[
                "no_relax_materials", "rattle_no_relax",
            ],
            labels=["Materials Mutations", "Rattle Mutation",],
            title_tag="Effect of Gaussian Noise and Large Number of evaluations",
            filename_tag="200k_evals",
            plot_individually=plot_individually,
            # reference_path=pathlib.Path(
            # __file__).parent.parent.parent / ".experiment.nosync/report_data/6_benchmark/benchmark",
            # reference_label="Benchmark",
        )




if __name__ == '__main__':
    plot_archives = True
    no_relax(plot_individually=False, exp_1=True)
    # if plot_archives:
    #     all_experiments = ["0_no_relax"]
    #     fitness_values = [[8.7, 9.5]]  # 6.5
    #     path_to_all_experiments = pathlib.Path(
    #         __file__).parent.parent.parent / ".experiment.nosync/report_data"
    #     for i, experiment in tqdm(enumerate(all_experiments)):
    #         report_generator = ReportPlotGenerator(
    #             path_to_experiments=path_to_all_experiments / experiment,
    #             plot_labels=None,
    #             title_tag=None,
    #         )
    #         report_generator.plot_cvt_and_symmetry(override_fitness_values=fitness_values[i],
    #                                                force_replot=False, all_sub_experiments=False)
