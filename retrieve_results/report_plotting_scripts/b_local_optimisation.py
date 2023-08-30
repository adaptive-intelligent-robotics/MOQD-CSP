import pathlib

from retrieve_results.report_plot_generator import ReportPlotGenerator


# def relaxation_exp(plot_individually=True):
#     ### 1 - Force Threshold
#     ### 1a - Force Threshold 0.2, 0.4, 1
#     report_generator = ReportPlotGenerator(
#         path_to_experiments=pathlib.Path(
#             __file__).parent.parent / ".experiment.nosync/report_data/2_local_optimisation",
#         plot_labels=None,
#         title_tag=None
#     )
#     if force_thesh:
#         report_generator.plot_mean_statistics(
#             folder_names=[
#                 "threshold_02", "threshold_04", "threshold_1",
#             ],
#             labels=["0.2 eV/A", "0.4 eV/A", "1 eV/A"],
#             title_tag="Impact of Force Threshold",
#             filename_tag="value",
#             plot_individually=plot_individually,
#         )
#
#
# if __name__ == '__main__':
