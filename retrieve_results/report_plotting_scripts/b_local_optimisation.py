import pathlib

from retrieve_results.report_plot_generator import ReportPlotGenerator


def relaxation_exp(plot_individually=True, relax_archive_every_5=True):
    ### 3 - Local Optimisation
    ### 1a - Force Threshold 0.2, 0.4, 1
    report_generator = ReportPlotGenerator(
        path_to_experiments=pathlib.Path(
            __file__).parent.parent.parent / ".experiment.nosync/report_data/3_local_optimisation",
        plot_labels=None,
        title_tag=None
    )
    if relax_archive_every_5:
        report_generator.plot_mean_statistics(
            folder_names=[
                "relax_10_always_archive_20_every_5", "relax_10_always_archive_50_every_5",
            ],
            labels=["20 steps", "50 steps"],
            title_tag="Relaxing Whole Archive Every 5 Generations",
            filename_tag="archive_every_5",
            plot_individually=plot_individually,
            reference_path=pathlib.Path(
            __file__).parent.parent.parent / ".experiment.nosync/report_data/1_force_threshold/threshold_1",
            reference_label="100 steps",
        )


    if relax_archive_every_5:
        report_generator.plot_mean_statistics(
            folder_names=[
                "relax_10_always_archive_20_every_5", "relax_10_always_archive_50_every_5",
            ],
            labels=["20 steps", "50 steps"],
            title_tag="Relaxing Whole Archive Every 5 Generations",
            filename_tag="archive_every_5",
            plot_individually=plot_individually,
            reference_path=pathlib.Path(
            __file__).parent.parent.parent / ".experiment.nosync/report_data/1_force_threshold/threshold_1",
            reference_label="100 steps",
        )



if __name__ == '__main__':
    relaxation_exp(plot_individually=True, relax_archive_every_5=True)
