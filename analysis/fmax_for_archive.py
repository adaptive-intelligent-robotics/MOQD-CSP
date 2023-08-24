import pathlib

import numpy as np
from matplotlib import pyplot as plt

from csp_elites.map_elites.archive import Archive

if __name__ == '__main__':
    experiment_labels = [
        "20230813_01_48_TiO2_200_niches_for benchmark_100_relax_2"
    ]
    centroid_filepaths = [
        "centroids_200_2_band_gap_0_100_shear_modulus_0_120.dat",
    ]
    archive_number = 5079

    path_to_main_folder = pathlib.Path(__file__).parent.parent / ".experiment.nosync"/ "experiments"
    path_to_experiment = path_to_main_folder  /experiment_labels[0]
    path_to_centroids = path_to_main_folder / "centroids" / centroid_filepaths[0]

    archive = Archive.from_archive(
        archive_path=path_to_experiment / f"archive_{archive_number}.pkl",
        centroid_filepath=path_to_centroids,
    )
    forces, energies, stresses = archive.compute_chgnet_metrics_on_archive()

    fmax = np.max((forces ** 2).sum(axis=2), axis=1) ** 0.5
    fmax_per_atoms = (forces ** 2).sum(axis=2) ** 0.5

    plt.hist(fmax)
    plt.show()
    print(fmax)
