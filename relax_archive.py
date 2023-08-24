import pathlib
import pickle
import sys

import numpy as np
from ase import Atoms
from ase.ga.ofp_comparator import OFPComparator
from ase.ga.utilities import CellBounds
from chgnet.model import StructOptimizer
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.neighbors import KDTree
from tqdm import tqdm

from csp_elites.crystal.crystal_evaluator import CrystalEvaluator
from csp_elites.map_elites.elites_utils import make_hashable
from csp_elites.utils.asign_target_values_to_centroids import compute_centroids_for_target_solutions
from csp_elites.utils.plot import load_archive_from_pickle, \
    convert_fitness_and_ddescriptors_to_plotting_format, plot_2d_map_elites_repertoire_marta

if __name__ == '__main__':
    try:
        experiment_tag = sys.argv[1]
        archive_number = sys.argv[2]
    except IndexError:
        experiment_tag = "20230727_03_43_TiO2_test"
        archive_number = 1000020
    print(experiment_tag)
    directory = pathlib.Path(__file__).resolve().parent  / "experiments"
    centroids_file = directory / "centroids" / "centroids_200_2_band_gap_0_100_shear_modulus_0_100.dat"
    archive_filename = directory / experiment_tag / f"archive_{archive_number}.pkl"

    # archive_filename = "experiments/20230707_22_04_TiO2_no_relaxation_20k_evals/archive_20000.pkl"
    # centroids_file = "experiments/20230707_22_04_TiO2_no_relaxation_20k_evals/centroids_200_2.dat"
    fitnesses, centroids, descriptors, individuals = load_archive_from_pickle(archive_filename)

    relaxed_archive = []
    energies = []
    new_descriptors = []

    comparator = OFPComparator(n_top=24, dE=1.0,
                               cos_dist_max=1e-3, rcut=10., binwidth=0.05,
                               pbc=[True, True, True], sigma=0.05, nsigma=4,
                               recalculate=False)

    crystal_evaluator = CrystalEvaluator(comparator=comparator)

    structure_optimizer = StructOptimizer()
    individuals = [Atoms.fromdict(individual) for individual in individuals]

    relaxed_archive = []

    for individual in tqdm(individuals):
        relaxation_results = structure_optimizer.relax(individual)
        relaxed_archive.append(relaxation_results)
        with open(directory / experiment_tag / f"backup_relaxed_archive_{archive_number}.pkl",
                  "wb") as file:
            pickle.dump(relaxed_archive, file)

    relaxed_structures_as_dict = [AseAtomsAdaptor.get_atoms(result["final_structure"]).todict() for result in relaxed_archive]
    _, new_atoms_dict, energy_batch, bds_batch, _, _ = crystal_evaluator.batch_compute_fitness_and_bd(
        list_of_atoms=relaxed_structures_as_dict,
        cellbounds=CellBounds(
            bounds={'phi': [20, 160], 'chi': [20, 160], 'psi': [20, 160], 'a': [2, 40],
                    'b': [2, 40],
                    'c': [2, 40]}),
        really_relax=None,
        behavioral_descriptor_names=None,
        n_relaxation_steps=0,
    )


    with open(centroids_file, "r") as f:
        all_centroids = np.loadtxt(f)
    kdt = KDTree(all_centroids, leaf_size=30, metric='euclidean')

    centroids = []
    for i in range(len(relaxed_archive)):
        niche_index = kdt.query([[bds_batch[0][i], bds_batch[1][i]]], k=1)[1][0][0]
        niche = kdt.data[niche_index]
        n = make_hashable(niche)
        centroids.append(n)

    new_descriptors = np.array(bds_batch).T

    with open(directory / experiment_tag / f"relaxed_archive_{archive_number}.pkl", "wb") as file:
        pickle.dump([energy_batch, centroids, new_descriptors, relaxed_structures_as_dict, relaxed_archive], file)
