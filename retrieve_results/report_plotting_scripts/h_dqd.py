import pathlib

from retrieve_results.report_plot_generator import ReportPlotGenerator


def dqd(plot_individually=True, lr=True, relax_steps=True, batch_size=True):
    report_generator = ReportPlotGenerator(
        path_to_experiments=pathlib.Path(
            __file__).parent.parent.parent / ".experiment.nosync/report_data/8_omg_mega_selection",
        plot_labels=None,
        title_tag=None
    )
    if lr:
        report_generator.plot_mean_statistics(
            folder_names=[
                "lr-5", "lr-3", "lr_01", "lr_05", "lr_1",
            ],
            labels=["1e-5", "1e-3", "1e-1", "5e-5", "1"],
            title_tag="Learning Rate Impact on OMG-MEGA Mutation",
            filename_tag="lr_dqd",
            plot_individually=plot_individually,
            reference_path=pathlib.Path(
            __file__).parent.parent.parent / ".experiment.nosync/report_data/6_benchmark/benchmark_with_threshold",
            reference_label="Default Mutations",
        )

    if relax_steps:
        report_generator.plot_mean_statistics(
            folder_names=[
                "lr1_10_relax", "lr1_50_relax", "lr_1",
            ],
            labels=["10 Steps", "50 Steps", "100 Steps"],
            title_tag="Relaxation Impact on OMG-MEGA Mutation",
            filename_tag="relax_steps_dqd",
            plot_individually=plot_individually,
            reference_path=pathlib.Path(
            __file__).parent.parent.parent / ".experiment.nosync/report_data/6_benchmark/benchmark_with_threshold",
            reference_label="Default Mutations",
        )

    if batch_size:
        report_generator.plot_mean_statistics(
            folder_names=[
                "lr1_50_relax_batch_10", "lr1_100_relax_batch_10", "lr1_100_relax_batch_20", "lr_1"
            ],
            labels=["50 Steps Batch Size 10", "100 Steps Batch Size 10", "100 Steps Batch Size 10",
                    "100 Steps Batch Size 100"],
            title_tag="Relaxation Impact on OMG-MEGA Mutation",
            filename_tag="batch_size_dqd",
            plot_individually=plot_individually,
            reference_path=pathlib.Path(
            __file__).parent.parent.parent / ".experiment.nosync/report_data/6_benchmark/benchmark_with_threshold",
            reference_label="Default Mutations",
        )




if __name__ == '__main__':
    dqd(plot_individually=True, lr=False, relax_steps=False, batch_size=True)
