import os
import pickle

import numpy as np
from sklearn.neighbors import KDTree

from csp_elites.map_elites.elites_utils import make_hashable
from csp_elites.utils.asign_target_values_to_centroids import compute_centroids_for_target_solutions
from csp_elites.utils.plot import load_centroids, load_archive_from_pickle, \
    convert_fitness_and_ddescriptors_to_plotting_format, plot_2d_map_elites_repertoire_marta

if __name__ == '__main__':
    descriptor_minimum_values = np.array([-4, 0])
    descriptor_maximum_values = np.array([0, 4])
    centroid_filename = f"../experiments/centroids/centroids_200_2.dat"
    comparison_data = "../experiments/target_data/tio2_target_data.pkl"
    fitness_min_max_values = (4, 9)
    target_centroids = compute_centroids_for_target_solutions(
        centroids_file=centroid_filename,
        target_data_file=comparison_data,
        filter_for_number_of_atoms=24
    )

    all_centroids = load_centroids(centroid_filename)
    kdt = KDTree(all_centroids, leaf_size=30, metric='euclidean')

    list_of_directories = [
                           '400_evals', '600_evals']
    list_of_archive_names = []
    # for directory in [name for name in os.listdir("../experiments") if
    #                   os.path.isdir(f"../experiments/{name}")]:
    for directory in list_of_directories:
        # list_of_directories.append(directory)
        list_of_archives_per_directory = []
        for filename in [name for name in os.listdir(f"../experiments/{directory}") if
                         not os.path.isdir(name)]:
            if "archive_" in filename:
                list_of_archives_per_directory.append(filename)
                if "relaxed_archive" in filename:
                    with open(f"../experiments/{directory}/{filename}", "rb") as file:
                        fitnesses, centroids, descriptors, individuals = pickle.load(file)
                else:
                    fitnesses, centroids, descriptors, individuals = load_archive_from_pickle(f"../experiments/{directory}/{filename}")

                # reassign centroids
                reassigned_centroids = []
                for i in range(len(fitnesses)):
                    niche_index = kdt.query([(descriptors[i][0], descriptors[i][1])], k=1)[1][0][0]
                    niche = kdt.data[niche_index]
                    n = make_hashable(niche)
                    reassigned_centroids.append(n)

                fitnesses_for_plotting, descriptors_for_plotting = convert_fitness_and_ddescriptors_to_plotting_format(
                    all_centroids=all_centroids,
                    centroids_from_archive=reassigned_centroids,
                    fitnesses_from_archive=fitnesses,
                    descriptors_from_archive=descriptors,
                )

                archive_id = filename.lstrip("archive_").rstrip(".pkl")
                if "relaxed" in filename:
                    archive_id += "_relaxed"
                plot_2d_map_elites_repertoire_marta(
                    centroids=all_centroids,
                    repertoire_fitnesses=fitnesses_for_plotting,
                    minval=descriptor_minimum_values,
                    maxval=descriptor_maximum_values,
                    repertoire_descriptors=descriptors_for_plotting,
                    vmin=fitness_min_max_values[0],
                    vmax=fitness_min_max_values[1],
                    target_centroids=target_centroids,
                    directory_string=f"../experiments/{directory}/", # TODO: remove this slach and add in plotting loop
                    filename=f"cvt_plot_{archive_id}",
                )
                print()
