import pickle

import numpy as np
from ase.ga.ofp_comparator import OFPComparator
from ase.ga.utilities import CellBounds
from sklearn.neighbors import KDTree
from tqdm import tqdm

from csp_elites.crystal.crystal_evaluator import CrystalEvaluator
from csp_elites.map_elites.elites_utils import make_hashable
from csp_elites.utils.asign_target_values_to_centroids import compute_centroids_for_target_solutions
from csp_elites.utils.plot import load_archive_from_pickle, \
    convert_fitness_and_ddescriptors_to_plotting_format, plot_2d_map_elites_repertoire_marta

if __name__ == '__main__':
    archive_filename = "experiments/20230707_22_04_TiO2_no_relaxation_20k_evals/archive_20000.pkl"
    centroids_file = "experiments/20230707_22_04_TiO2_no_relaxation_20k_evals/centroids_200_2.dat"
    fitnesses, centroids, descriptors, individuals = load_archive_from_pickle(archive_filename)

    relaxed_archive = []
    energies = []
    new_descriptors = []

    comparator = OFPComparator(n_top=24, dE=1.0,
                               cos_dist_max=1e-3, rcut=10., binwidth=0.05,
                               pbc=[True, True, True], sigma=0.05, nsigma=4,
                               recalculate=False)

    crystal_evaluator = CrystalEvaluator(comparator=comparator)


    for individual in tqdm(individuals):
        energy, bds, _ = crystal_evaluator.compute_fitness_and_bd(
            atoms=individual,
            cellbounds=CellBounds(
            bounds={'phi': [20, 160], 'chi': [20, 160], 'psi': [20, 160], 'a': [2, 40], 'b': [2, 40],
                    'c': [2, 40]}),
            population=individuals,
            really_relax=True,
        )
        energies.append(energy)
        new_descriptors.append(bds)
        relaxed_archive.append(individual)

    with open(centroids_file, "r") as f:
        all_centroids = np.loadtxt(f)
    kdt = KDTree(all_centroids, leaf_size=30, metric='euclidean')

    centroids = []
    for i in range(len(relaxed_archive)):
        niche_index = kdt.query([new_descriptors[i]], k=1)[1][0][0]
        niche = kdt.data[niche_index]
        n = make_hashable(niche)
        centroids.append(n)

    fitnesses_for_plotting, descriptors_for_plotting = convert_fitness_and_ddescriptors_to_plotting_format(
        all_centroids=all_centroids,
        centroids_from_archive=centroids,
        fitnesses_from_archive=energies,
        descriptors_from_archive=new_descriptors,
    )
    plot_2d_map_elites_repertoire_marta(
        centroids=all_centroids,
        repertoire_fitnesses=fitnesses_for_plotting,
        minval=[-4, 0],
        maxval=[0, 4],
        repertoire_descriptors=descriptors_for_plotting,
        vmin=6.5,
        vmax=9,
        target_centroids=None,
        directory_string="experiments/20230707_22_04_TiO2_no_relaxation_20k_evals/",
        filename="cvt_plot_relaxed"
    )

    with open("experiments/20230707_22_04_TiO2_no_relaxation_20k_evals/relaxed_archive_20k.pkl", "wb") as file:
        pickle.dump([energies, centroids, new_descriptors, relaxed_archive], file)
