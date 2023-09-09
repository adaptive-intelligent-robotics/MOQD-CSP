import pathlib

from retrieve_results.report_plot_generator import ReportPlotGenerator


def dqd(
    plot_individually=True, lr=True, relax_steps=True, batch_size=True, dqd_rattle=True
):
    report_generator = ReportPlotGenerator(
        path_to_experiments=pathlib.Path(__file__).parent.parent.parent
        / ".experiment.nosync/report_data/8_omg_mega_selection",
        plot_labels=None,
        title_tag=None,
    )
    if lr:
        report_generator.plot_combined_figure_for_report(
            folder_names=[
                "lr-3",
                "lr_01",
                "lr_1",
            ],
            labels=["1e-3", "1e-1", "1"],
            title_tag="Learning Rate Impact",
            filename_tag="lr_dqd",
            plot_individually=plot_individually,
            reference_path=pathlib.Path(__file__).parent.parent.parent
            / ".experiment.nosync/report_data/6_benchmark/benchmark_with_threshold",
            reference_label="Baseline",
            y_limits_dict={"Maximum Fitness": [9.35, 9.42]},
        )

    if relax_steps:
        report_generator.plot_combined_figure_for_report(
            folder_names=[
                "lr1_50_relax",
                "lr_1",  # "lr1_10_relax",
            ],
            labels=["50 Steps", "100 Steps"],  # "10 Steps",
            title_tag="Relaxation Impact",
            filename_tag="relax_steps_dqd",
            plot_individually=plot_individually,
            reference_path=pathlib.Path(__file__).parent.parent.parent
            / ".experiment.nosync/report_data/6_benchmark/benchmark_with_threshold",
            reference_label="Baseline",
        )

    if batch_size:
        report_generator.plot_combined_figure_for_report(
            folder_names=[
                "lr1_100_relax_batch_10",
                "lr1_100_relax_batch_20",
                "lr_1",
            ],
            labels=["Batch Size 10", "Batch Size 20", "Batch Size 100"],
            title_tag="Batch Size Impact",
            filename_tag="batch_size_dqd",
            plot_individually=plot_individually,
            reference_path=pathlib.Path(__file__).parent.parent.parent
            / ".experiment.nosync/report_data/6_benchmark/benchmark_with_threshold",
            reference_label="Baseline",
            y_limits_dict={
                "Maximum Fitness": [9.35, 9.42],
                # "Mean Fitness",
                # "Median Fitness",
                # "Fitness 5th Percentile",
                # "Fitness 95th Percentile",
                # "QD score"
            },
        )

    if dqd_rattle:
        report_generator.plot_combined_figure_for_report(
            folder_names=["lr1_100_relax_batch_10", "5050_with_rattle"],
            labels=["OMG-MEGA Mutation", "5050 DQD / rattle mutations"],
            title_tag="Impact of Rattle Mutation",
            filename_tag="5050_dqd",
            plot_individually=plot_individually,
            reference_path=pathlib.Path(__file__).parent.parent.parent
            / ".experiment.nosync/report_data/6_benchmark/benchmark_with_threshold",
            reference_label="Baseline",
            y_limits_dict={
                "Maximum Fitness": [9.35, 9.42],
                # "Mean Fitness",
                # "Median Fitness",
                # "Fitness 5th Percentile",
                # "Fitness 95th Percentile",
                # "QD score"
            },
        )

    report_generator.plot_combined_figure_for_report(
        folder_names=[
            "lr_1",
        ],
        labels=["Batch size100"],
        title_tag="Batch Size Impact",
        filename_tag="test",
        plot_individually=plot_individually,
        reference_path=pathlib.Path(__file__).parent.parent.parent
        / ".experiment.nosync/report_data/6_benchmark/benchmark_with_threshold",
        reference_label="Baseline",
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
    dqd(
        plot_individually=False,
        lr=False,
        relax_steps=False,
        batch_size=True,
        dqd_rattle=False,
    )
