import pathlib

from retrieve_results.report_plot_generator import ReportPlotGenerator


def force_mut_only(
    plot_individually=True,
    gaussian=True,
    lr_step_only=True,
    lr_step_only_simple=True,
    compare=True,
):
    report_generator = ReportPlotGenerator(
        path_to_experiments=pathlib.Path(__file__).parent.parent.parent
        / ".experiment.nosync/report_data/4_force_mutation_only",
        plot_labels=None,
        title_tag=None,
    )
    # if gaussian:
    #     report_generator.plot_mean_statistics(
    #         folder_names=[
    #             "lr-3", "lr-1", "lr05", "lr1", "lr1_simple", "lr1_simple_batch_10",  #"lr1_simple_batch_10_no_relax"
    #         ],
    #         labels=["1-e3", "1e-1", "0.05", "1", "1 simple", "1 simple batch 10"],
    #         title_tag="OMG-MEGA only force",
    #         filename_tag="lr_force_only",
    #         plot_individually=plot_individually,
    #         reference_path=pathlib.Path(
    #         __file__).parent.parent.parent / ".experiment.nosync/report_data/6_benchmark/benchmark_with_threshold",
    #         reference_label="Benchmark",
    #         y_limits_dict={
    #             "Maximum Fitness": [9.3, 9.42]
    #         }
    #     )

    if lr_step_only_simple:
        report_generator.plot_combined_figure_for_report(
            folder_names=[
                "no_gaussian_lr-3_simple",
                "no_gaussian_lr-1_simple",
                "no_gaussian_lr1_simple"
                # "lr1_simple_batch_10_no_relax"
            ],
            labels=["1e-3", "1e-1", "1"],
            title_tag="Deterministic Steps Learning Rate Simple",
            filename_tag="no_gaussian_simple",
            plot_individually=plot_individually,
            reference_path=pathlib.Path(__file__).parent.parent.parent
            / ".experiment.nosync/report_data/6_benchmark/benchmark",
            reference_label="Baseline",
            y_limits_dict={"Maximum Fitness": [9.35, 9.42]},
        )

        report_generator.plot_combined_figure_for_report(
            folder_names=[
                "no_gaussian_lr1_simple",
                "no_gaussian_lr1"
                # "lr1_simple_batch_10_no_relax"
            ],
            labels=["simple", "normal"],
            title_tag="Deterministic LR Simple vs Normal",
            filename_tag="no_gaussian_lr1_compare",
            plot_individually=plot_individually,
            reference_path=pathlib.Path(__file__).parent.parent.parent
            / ".experiment.nosync/report_data/6_benchmark/benchmark",
            reference_label="Baseline",
            y_limits_dict={"Maximum Fitness": [9.37, 9.42]},
        )

    if lr_step_only:
        report_generator.plot_combined_figure_for_report(
            folder_names=[
                "no_gaussian_lr-3",
                "no_gaussian_lr-1",
                "no_gaussian_lr1"
                # "lr1_simple_batch_10_no_relax"
            ],
            labels=["1e-3", "1e-1", "1"],
            title_tag="Deterministic Steps Learning Rate Simple",
            filename_tag="no_gaussian",
            plot_individually=plot_individually,
            reference_path=pathlib.Path(__file__).parent.parent.parent
            / ".experiment.nosync/report_data/6_benchmark/benchmark",
            reference_label="Baseline",
            y_limits_dict={"Maximum Fitness": [9.37, 9.42]},
        )

    if compare:
        report_generator.plot_combined_figure_for_report(
            folder_names=[
                "no_gaussian_lr1",
                "lr1",
            ],
            labels=["Deterministic", "Probailistic"],
            title_tag="Deterministic vsa Probabilistic Force Mutation",
            filename_tag="gauss_vs_no_gauss",
            plot_individually=plot_individually,
            reference_path=pathlib.Path(__file__).parent.parent.parent
            / ".experiment.nosync/report_data/6_benchmark/benchmark",
            reference_label="Baseline",
            y_limits_dict={"Maximum Fitness": [9.37, 9.42]},
        )


if __name__ == "__main__":
    force_mut_only(
        plot_individually=False,
        gaussian=False,
        lr_step_only=False,
        lr_step_only_simple=False,
        compare=True,
    )
