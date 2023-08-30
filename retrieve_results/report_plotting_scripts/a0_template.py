import pathlib

from retrieve_results.report_plot_generator import ReportPlotGenerator


def exp(plot_individually=True, exp_1=True):
    report_generator = ReportPlotGenerator(
        path_to_experiments=pathlib.Path(
            __file__).parent.parent.parent / ".experiment.nosync/report_data/3_local_optimisation",
        plot_labels=None,
        title_tag=None
    )
    if exp_1:
        report_generator.plot_mean_statistics(
            folder_names=[
                ""
            ],
            labels=[""],
            title_tag="",
            filename_tag="",
            plot_individually=plot_individually,
            reference_path=pathlib.Path(
            __file__).parent.parent.parent / ".experiment.nosync/report_data/",
            reference_label="",
        )




if __name__ == '__main__':
    exp(plot_individually=True, exp_1=True)
