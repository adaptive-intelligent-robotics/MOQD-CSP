import pathlib

from retrieve_results.report_plot_generator import ReportPlotGenerator


def force_vs_dqd(plot_individually=True, exp_1=True):
    report_generator = ReportPlotGenerator(
        path_to_experiments=pathlib.Path(__file__).parent.parent.parent
        / ".experiment.nosync/report_data/9_omg_vs_force",
        plot_labels=None,
        title_tag=None,
    )
    if exp_1:
        report_generator.plot_combined_figure_for_report(
            folder_names=[
                "force_only",
                "force_only_batch_10",  # "force_only_simple_batch_10", "dqd_lr1_100_relax_batch_10",  #"dqd_lr_1_simple",  # "force_only_no_relax_batch_10",
            ],
            labels=[
                "Batch 100",
                "Batch 10",
            ],  # "No Relax Batch 10",
            title_tag="",
            filename_tag="dqd_vs_gaus_force",
            plot_individually=plot_individually,
            reference_path=pathlib.Path(__file__).parent.parent.parent
            / ".experiment.nosync/report_data/8_omg_mega_selection/lr1_100_relax_batch_10",
            reference_label="DQD 100 batch 10",
            y_limits_dict={
                "Maximum Fitness": [9.35, 9.42],
                # "Mean Fitness",
                # "Median Fitness",
                # "Fitness 5th Percentile",
                # "Fitness 95th Percentile",
                # "QD score"
            },
        )


if __name__ == "__main__":
    force_vs_dqd(plot_individually=False, exp_1=True)
