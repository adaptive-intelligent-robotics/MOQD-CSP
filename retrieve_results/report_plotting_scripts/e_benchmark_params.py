import pathlib

from retrieve_results.report_plot_generator import ReportPlotGenerator


def benchmark_params(plot_individually=True, niches_fill=True, n_niches=True, structure_initialise=True,
                     batch_size=True):
    report_generator = ReportPlotGenerator(
        path_to_experiments=pathlib.Path(
            __file__).parent.parent.parent / ".experiment.nosync/report_data/5_benchmark_params",
        plot_labels=None,
        title_tag=None
    )
    if niches_fill:
        report_generator.plot_mean_statistics(
            folder_names=[
                "02_niches"
            ],
            labels=["0.2 Niches Filled"],
            title_tag="Niches Filled with Random Generation",
            filename_tag="niches_filled",
            plot_individually=plot_individually,
            reference_path=pathlib.Path(
            __file__).parent.parent.parent / ".experiment.nosync/report_data/6_benchmark/benchmark",
            reference_label="0.1 Niches Filled",
        )

    if n_niches:
        report_generator.plot_mean_statistics(
            folder_names=[
                "500_niches"
            ],
            labels=["500 Niches"],
            title_tag="Number of Niches Available",
            filename_tag="niches_available",
            plot_individually=plot_individually,
            reference_path=pathlib.Path(
                __file__).parent.parent.parent / ".experiment.nosync/report_data/6_benchmark/benchmark",
            reference_label="200 Niches",
        )

    if structure_initialise:
        report_generator.plot_mean_statistics(
            folder_names=[
                "structure_initialise_40", "structure_initialise_80"
            ],
            labels=["40 Individuals", "80 Individuals"],
            title_tag="Number of Individuals Initialised",
            filename_tag="init_individuals",
            plot_individually=plot_individually,
            reference_path=pathlib.Path(
                __file__).parent.parent.parent / ".experiment.nosync/report_data/6_benchmark/benchmark",
            reference_label="20 Individuals",
        )

    if batch_size:
        report_generator.plot_mean_statistics(
            folder_names=[
                "batch_size_20", "batch_size_50"
            ],
            labels=["Batch Size 20", "Batch Size 50"],
            title_tag="Number of Individuals Initialised",
            filename_tag="batch_size",
            plot_individually=plot_individually,
            reference_path=pathlib.Path(
                __file__).parent.parent.parent / ".experiment.nosync/report_data/6_benchmark/benchmark",
            reference_label="Batch Size 100",
        )


if __name__ == '__main__':
    benchmark_params(
        plot_individually=True,
        niches_fill=True,
        n_niches=True,
        structure_initialise=True,
        batch_size=True,
    )
