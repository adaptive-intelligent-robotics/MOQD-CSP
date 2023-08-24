import pickle
from typing import Optional

import numpy as np
# from scipy.spatial import cKDTree : TODO -- faster?
from sklearn.neighbors import KDTree

from csp_elites.map_elites.elites_utils import make_hashable
from csp_elites.utils.get_mpi_structures import get_all_materials_with_formula


# from csp_qdax.pymap_elites_csp.plot import convert_fitness_and_ddescriptors_to_plotting_format, \
#     plot_2d_map_elites_repertoire_marta
# from csp_qdax.pymap_elites_csp.plot import load_centroids, plot_2d_map_elites_repertoire_marta, \
#     convert_fitness_and_ddescriptors_to_plotting_format


def compute_centroids_for_target_solutions(centroids_file: str,
                                           target_data_file: str,
                                           filter_for_number_of_atoms: Optional[int]):
    with open(centroids_file, "r") as f:
        c = np.loadtxt(f)
    kdt = KDTree(c, leaf_size=30, metric='euclidean')

    with open(target_data_file, "rb") as file:
        list_of_properties = pickle.load(file)

    docs, atom_objects = get_all_materials_with_formula("TiO2")

    if filter_for_number_of_atoms is not None:
        fitnesses = []
        formation_energies = []
        band_gaps = []
        for i, atoms in enumerate(atom_objects):
            if len(atoms.get_positions()) == 24:
                fitnesses.append(list_of_properties[0][i])
                formation_energies.append(list_of_properties[2][i])
                band_gaps.append(list_of_properties[1][i])
    else:
        fitnesses = list_of_properties[0]
        formation_energies = list_of_properties[2]
        band_gaps = list_of_properties[1]
    centroids = []
    for i in range(len(fitnesses)):
        niche_index = kdt.query([(formation_energies[i], band_gaps[i])], k=1)[1][0][0]
        niche = kdt.data[niche_index]
        n = make_hashable(niche)
        centroids.append(n)

    return centroids

def reassign_data_from_pkl_to_new_centroids(centroids_file: str,
                                           target_data,
                                           filter_for_number_of_atoms: Optional[int]):
    with open(centroids_file, "r") as f:
        c = np.loadtxt(f)
    kdt = KDTree(c, leaf_size=30, metric='euclidean')

    fitnesses, centroids, descriptors, individuals = target_data
    fitnesses_to_enumerate = []
    formation_energies = []
    band_gaps = []

    if filter_for_number_of_atoms is not None:
        for i, atoms in enumerate(individuals):
            if len(atoms.get_positions()) == 24:
                fitnesses_to_enumerate.append(fitnesses[i])
                formation_energies.append(descriptors[i][0])
                band_gaps.append(descriptors[i][1])
    else:
        for i, atoms in enumerate(individuals):
            fitnesses_to_enumerate.append(fitnesses[i])
            formation_energies.append(descriptors[i][0])
            band_gaps.append(descriptors[i][1])
    new_centroids = []
    for i in range(len(fitnesses_to_enumerate)):
        niche_index = kdt.query([(formation_energies[i], band_gaps[i])], k=1)[1][0][0]
        niche = kdt.data[niche_index]
        n = make_hashable(niche)
        new_centroids.append(n)

    return new_centroids
if __name__ == '__main__':
    compute_centroids_for_target_solutions(
        "../experiments/20230706_09_55_TiO2/centroids_2000_2.dat",
        "../../experiments/target_data/tio2_target_data.pkl",
        filter_for_number_of_atoms=24,
    )
