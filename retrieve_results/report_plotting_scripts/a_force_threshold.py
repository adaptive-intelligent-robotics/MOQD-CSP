import pathlib

from retrieve_results.report_plot_generator import ReportPlotGenerator


def force_threshold_exp(plot_individually=True,
        force_thesh=True, relax_steps=True, relax_steps_no_10=True):
    ### 1 - Force Threshold
    ### 1a - Force Threshold 0.2, 0.4, 1
    report_generator = ReportPlotGenerator(
        path_to_experiments=pathlib.Path(
            __file__).parent.parent.parent / ".experiment.nosync/report_data/1_force_threshold",
        plot_labels=None,
        title_tag=None
    )
    if force_thesh:
        report_generator.plot_combined_figure_for_report(
            folder_names=[
                "threshold_02", "threshold_04", "threshold_1",
            ],
            labels=["0.2 eV/A", "0.4 eV/A", "1 eV/A"],
            title_tag="Impact of Force Threshold",
            filename_tag="force_threshold_value",
            plot_individually=plot_individually,
            y_limits_dict={
                "Maximum Fitness": [9.3, 9.42]
            }
        )
    if relax_steps:
        report_generator.plot_combined_figure_for_report(
            folder_names=[
                "threshold_1_10_relax", "threshold_1_50_relax", "threshold_1"
            ],
            labels=["10 steps", "50 steps", "100 steps"],
            title_tag="Number of Relaxation Steps with 1 eV Threshold",
            filename_tag="steps",
            plot_individually=plot_individually,
        )
    if relax_steps_no_10:
        report_generator.plot_combined_figure_for_report(
            folder_names=[
                "threshold_1_50_relax", "threshold_1"
            ],
            labels=["50 steps", "100 steps"],
            title_tag="Impact of Force Threshold on Relaxation",
            filename_tag="steps_no_10",
            plot_individually=plot_individually,
        )

def compute_match_stats():
    report_generator = ReportPlotGenerator(
        path_to_experiments=pathlib.Path(
            __file__).parent.parent.parent / ".experiment.nosync/report_data/1_force_threshold",
        plot_labels=None,
        title_tag=None
    )
    report_generator.compute_average_match_metrics(["threshold_02", "threshold_04", "threshold_1"])


if __name__ == '__main__':
    # force_threshold_exp(plot_individually=False, force_thesh=True, relax_steps=True, relax_steps_no_10=True)
    compute_match_stats()
