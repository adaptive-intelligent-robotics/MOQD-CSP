import pathlib

from retrieve_results.report_plot_generator import ReportPlotGenerator


def force_threshold_exp(plot_individually=True,
        force_thesh=True, relax_steps=True, relax_steps_no_10=True):
    ### 1 - Force Threshold
    ### 1a - Force Threshold 0.2, 0.4, 1
    report_generator = ReportPlotGenerator(
        path_to_experiments=pathlib.Path(
            __file__).parent.parent / ".experiment.nosync/report_data/1_force_threshold",
        plot_labels=None,
        title_tag=None
    )
    if force_thesh:
        report_generator.plot_mean_statistics(
            folder_names=[
                "threshold_02", "threshold_04", "threshold_1",
            ],
            labels=["0.2 eV/A", "0.4 eV/A", "1 eV/A"],
            title_tag="Impact of Force Threshold",
            filename_tag="value",
            plot_individually=plot_individually,
        )
    if relax_steps:
        report_generator.plot_mean_statistics(
            folder_names=[
                "threshold_1_10_relax", "threshold_1_50_relax", "threshold_1"
            ],
            labels=["10 steps", "50 steps", "100 steps"],
            title_tag="Impact of Force Threshold on Relaxation",
            filename_tag="steps",
            plot_individually=plot_individually,
        )
    if relax_steps_no_10:
        report_generator.plot_mean_statistics(
            folder_names=[
                "threshold_1_50_relax", "threshold_1"
            ],
            labels=["50 steps", "100 steps"],
            title_tag="Impact of Force Threshold on Relaxation",
            filename_tag="steps_no_10",
            plot_individually=plot_individually,
        )


if __name__ == '__main__':
    force_threshold_exp(plot_individually=False, force_thesh=False, relax_steps=False, relax_steps_no_10=True)
