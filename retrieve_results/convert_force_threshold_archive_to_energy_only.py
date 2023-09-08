import pathlib

import numpy as np

from csp_elites.map_elites.archive import Archive
from csp_elites.utils.plot import load_centroids, plot_2d_map_elites_repertoire_marta, \
    load_archive_from_pickle

if __name__ == '__main__':
    shear_modulus = 1
    experiment_labels = [
        # "20230809_23_50_TiO2_rattle_10_steps_relax_every_5_20_init_batch"
        "20230828_12_18_TiO2_force_threshold_02_12"
    ]
    centroid_filepaths = [
        f"centroids_200_2_band_gap_0_1_shear_modulus_0_{shear_modulus}.dat",
    ]
    archive_number = 240

    path_to_main_folder = pathlib.Path(__file__).parent.parent / ".experiment.nosync"/ "report_data/1_force_threshold/threshold_02"
    path_to_experiment = path_to_main_folder  /experiment_labels[0]
    path_to_centroids = pathlib.Path(__file__).parent.parent / ".experiment.nosync" / "experiments" / "centroids" / centroid_filepaths[0]

    archive = Archive.from_archive(
        archive_path=path_to_experiment / f"archive_{archive_number}.pkl",
        centroid_filepath=path_to_centroids,
    )
    forces, energies, stresses = archive.compute_chgnet_metrics_on_archive()

    archive_with_energy_fitness = Archive.from_archive(
        archive_path=path_to_experiment / f"archive_{archive_number}.pkl",
        centroid_filepath=path_to_centroids,
    )
    archive_with_energy_fitness.fitnesses = -1 * np.array(energies)
    all_centroids = load_centroids(path_to_centroids)
    plot_fitness, plot_descriptors, labels_for_plotting = archive_with_energy_fitness.convert_fitness_and_descriptors_to_plotting_format(all_centroids)
    plot_fitness_normal, plot_descriptors_normal, labels_for_plotting_normal = archive.convert_fitness_and_descriptors_to_plotting_format(all_centroids)


    plot_2d_map_elites_repertoire_marta(
    centroids=all_centroids,
    repertoire_fitnesses=plot_fitness,
    minval=[0, 0],
    maxval=[1, shear_modulus],
    repertoire_descriptors=plot_descriptors,
    vmin=8.5,
    vmax=9.5,
    directory_string=path_to_experiment,
    filename=f"cvt_plot_{archive_number}_energy_only",
        annotate=False,
        x_axis_limits=[0, 4],
        y_axis_limits=[0,120]
    )

    plot_2d_map_elites_repertoire_marta(
    centroids=all_centroids,
    repertoire_fitnesses=plot_fitness_normal,
    minval=[0, 0],
    maxval=[1, shear_modulus],
    repertoire_descriptors=plot_descriptors_normal,
    vmin=8.5,
    vmax=9.5,
    directory_string=path_to_experiment,
    filename=f"cvt_plot_{archive_number}_no_target",
        annotate=False,
        x_axis_limits=[0, 4],
        y_axis_limits=[0,120]
    )
