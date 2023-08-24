import pathlib

import numpy as np

from csp_elites.map_elites.archive import Archive
from csp_elites.utils.plot import load_centroids, plot_2d_map_elites_repertoire_marta, \
    load_archive_from_pickle

if __name__ == '__main__':
    experiment_labels = [
        # "20230809_23_50_TiO2_rattle_10_steps_relax_every_5_20_init_batch"
        "20230807_05_22_TiO2_rattle_relax_every_1000"
    ]
    centroid_filepaths = [
        "centroids_200_2_band_gap_0_100_shear_modulus_0_120.dat",
    ]
    archive_number = 340000

    path_to_main_folder = pathlib.Path(__file__).parent.parent / ".experiment.nosync"/ "experiments"
    path_to_experiment = path_to_main_folder  /experiment_labels[0]
    path_to_centroids = path_to_main_folder / "centroids" / centroid_filepaths[0]

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

    plot_2d_map_elites_repertoire_marta(
    centroids=all_centroids,
    repertoire_fitnesses=plot_fitness,
    minval=[0, 0],
    maxval=[100, 120],
    repertoire_descriptors=plot_descriptors,
    vmin=6.5,
    vmax=10,
    )
