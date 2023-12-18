import pathlib

from retrieve_results.report_plot_generator import ReportPlotGenerator


def relaxation_exp(
    plot_individually=True,
    relax_archive_every_5=True,
    relax_steps=True,
    archive_relax_no_intermediate_relax=True,
):
    ### 3 - Local Optimisation
    ### 1a - Force Threshold 0.2, 0.4, 1
    report_generator = ReportPlotGenerator(
        path_to_experiments=pathlib.Path(__file__).parent.parent.parent
        / ".experiment.nosync/report_data/3_local_optimisation",
        plot_labels=None,
        title_tag=None,
    )
    if relax_archive_every_5:
        report_generator.plot_combined_figure_for_report(
            folder_names=[
                "relax_10_always_archive_20_every_5",
                "relax_10_always_archive_50_every_5",
                "relax_archive_50_every_5",
            ],
            labels=["20 steps", "50 steps", "50 steps - no intermittent relax"],
            title_tag="Relaxing Whole Archive Every 5 Generations",  # 10 steps always
            filename_tag="archive_every_5_step_10_always",
            plot_individually=plot_individually,
            reference_path=pathlib.Path(__file__).parent.parent.parent
            / ".experiment.nosync/report_data/1_force_threshold/threshold_1",
            reference_label="100 Steps Always",
            y_limits_dict={
                "Maximum Fitness": (9.3, 9.42),
            },
        )

    if relax_steps:
        report_generator.plot_combined_figure_for_report(
            folder_names=[
                "relax_10",
                "relax_50",
            ],
            labels=["10 Steps", "50 Steps"],
            title_tag="Number of Relaxation Steps",  #
            filename_tag="n_relax_steps",
            plot_individually=plot_individually,
            reference_path=pathlib.Path(__file__).parent.parent.parent
            / ".experiment.nosync/report_data/1_force_threshold/threshold_1",
            reference_label="100 Steps",
            y_limits_dict={
                "Maximum Fitness": (9.3, 9.42),
            },
        )

    if archive_relax_no_intermediate_relax:
        report_generator.plot_combined_figure_for_report(
            folder_names=[
                "relax_archive_10_every_5",
                "relax_archive_10_every_10",
                "relax_archive_50_every_5",
                "relax_archive_50_every_10",
            ],
            labels=[
                "10 Every 5 Generations",
                "10 Every 10 Generations",
                "50 Every 5 Generations",
                "50 Every 10 Generations",
            ],
            title_tag="Relaxing Whole Archive No Intermediate Relaxation",
            filename_tag="relax_archive_no_intermediate_relax",
            plot_individually=plot_individually,
            reference_path=pathlib.Path(__file__).parent.parent.parent
            / ".experiment.nosync/report_data/1_force_threshold/threshold_1",
            reference_label="100 Steps Always",
            y_limits_dict={
                "Maximum Fitness": (9.3, 9.42),
            },
        )


if __name__ == "__main__":
    relaxation_exp(
        plot_individually=False,
        relax_archive_every_5=False,
        relax_steps=False,
        archive_relax_no_intermediate_relax=True,
    )
