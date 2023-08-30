import pathlib

from retrieve_results.report_plot_generator import ReportPlotGenerator


def force_vs_dqd(plot_individually=True, exp_1=True):
    report_generator = ReportPlotGenerator(
        path_to_experiments=pathlib.Path(
            __file__).parent.parent.parent / ".experiment.nosync/report_data/9_omg_vs_force",
        plot_labels=None,
        title_tag=None
    )
    if exp_1:
        report_generator.plot_mean_statistics(
            folder_names=[
                "force_only", "force_only_batch_10", "force_only_no_relax_batch_10", "force_only_simple_batch_10",
            ],
            labels=["Force Only Batch 100", "Force Only Batch 10", "Force Only No Relax Batch 10", "Force Only Simple, Batch 10" ],
            title_tag="",
            filename_tag="",
            plot_individually=plot_individually,
            reference_path=pathlib.Path(
            __file__).parent.parent.parent / ".experiment.nosync/report_data/8_omg_mega_selection/lr1_100_relax_batch_10",
            reference_label="OMG-MEGA Mutation",
        )




if __name__ == '__main__':
    force_vs_dqd(plot_individually=True, exp_1=True)
