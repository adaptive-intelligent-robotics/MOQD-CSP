

import pathlib

from retrieve_results.report_plot_generator import ReportPlotGenerator


def benchmark(plot_individually=True, all_comp=True, compute_symmtery_stats=True):
    report_generator = ReportPlotGenerator(
        path_to_experiments=pathlib.Path(
            __file__).parent.parent.parent / ".experiment.nosync/report_data/6_benchmark",
        plot_labels=None,
        title_tag=None
    )
    if all_comp:
        report_generator.plot_combined_figure_for_report(
            folder_names=[
                "benchmark", "benchmark_with_threshold"
            ],
            labels=["No Threshold", "With Threshold"],
            title_tag="Impact of Stability Threshold",
            filename_tag="benchmark",
            plot_individually=plot_individually,
            y_limits_dict={
                "Maximum Fitness": [9.375, 9.42]
            }
        )

    if compute_symmtery_stats:
        report_generator.compute_average_match_metrics()


if __name__ == '__main__':
    benchmark(plot_individually=False, all_comp=True, compute_symmtery_stats=False)
