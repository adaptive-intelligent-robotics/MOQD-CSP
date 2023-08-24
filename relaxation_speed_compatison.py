import pathlib
import pickle
import sys
import time

import numpy as np
from ase import Atoms
from ase.ga.ofp_comparator import OFPComparator
from ase.ga.utilities import CellBounds
from chgnet.model import StructOptimizer
from matplotlib import pyplot as plt
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.neighbors import KDTree
from tqdm import tqdm

from csp_elites.crystal.crystal_evaluator import CrystalEvaluator
from csp_elites.map_elites.elites_utils import make_hashable
from csp_elites.parallel_relaxation.structure_optimizer import MultiprocessOptimizer
from csp_elites.utils.asign_target_values_to_centroids import compute_centroids_for_target_solutions
from csp_elites.utils.plot import load_archive_from_pickle, \
    convert_fitness_and_ddescriptors_to_plotting_format, plot_2d_map_elites_repertoire_marta

# archive_number =
# directory_string = pathlib.Path(
#     __file__).parent / ".experiment.nosync" / "experiments" / "20230805_11_13_TiO2_200_niches_for benchmark"
if __name__ == '__main__':
    try:
        experiment_tag = sys.argv[1]
        archive_number = sys.argv[2]
    except IndexError:
        experiment_tag = "20230805_11_13_TiO2_200_niches_for benchmark"
        archive_number = 24830
    print(experiment_tag)
    directory = pathlib.Path(__file__).resolve().parent  / "experiments"
    # directory = pathlib.Path(__file__).resolve().parent / ".experiment.nosync" /"experiments"
    centroids_file = directory / "centroids" / "centroids_200_2_band_gap_0_100_shear_modulus_0_100_slurm.dat"
    archive_filename = directory / experiment_tag / f"archive_{archive_number}.pkl"

    timing_batches_end = np.array([1, 6, 16, 66, 166])
    timing_batches_start = np.array([0, 1, 6, 16, 66])

    # fitnesses, centroids, descriptors, individuals = load_archive_from_pickle(archive_filename)
    #
    # # timing_batches_end = np.array([1, 3, 6,])
    # # timing_batches_start = np.array([0, 1, 3,])
    #
    # relaxed_archive = []
    # energies = []
    # new_descriptors = []
    #
    comparator = OFPComparator(n_top=24, dE=1.0,
                               cos_dist_max=1e-3, rcut=10., binwidth=0.05,
                               pbc=[True, True, True], sigma=0.05, nsigma=4,
                               recalculate=False)

    crystal_evaluator = CrystalEvaluator(comparator=comparator)

    # structure_optimizer = StructOptimizer()
    # individuals = [Atoms.fromdict(individual) for individual in individuals]
    #
    # relaxed_archive = []
    #
    # individuals_in_batches = [individuals[timing_batches_start[i]:timing_batches_end[i]] for i in range(len(timing_batches_end))]
    # ase_relaxation_timing = []
    # for individual_batch in tqdm(individuals_in_batches):
    #     batch_timing = 0
    #     for individual in individual_batch:
    #         tic = time.time()
    #         relaxation_results = structure_optimizer.relax(individual, steps=500, verbose=False)
    #         batch_timing += time.time() - tic
    #         relaxed_archive.append(relaxation_results)
    #         with open(directory / experiment_tag / f"backup_relaxed_archive_{archive_number}.pkl",
    #                   "wb") as file:
    #             pickle.dump(relaxed_archive, file)
    #
    #     ase_relaxation_timing.append(batch_timing)
    # relaxed_structures_as_dict = [AseAtomsAdaptor.get_atoms(result["final_structure"]).todict() for result in relaxed_archive]
    # _, new_atoms_dict, energy_batch, bds_batch, _ = crystal_evaluator.batch_compute_fitness_and_bd(
    #     list_of_atoms=relaxed_structures_as_dict,
    #     cellbounds=CellBounds(
    #         bounds={'phi': [20, 160], 'chi': [20, 160], 'psi': [20, 160], 'a': [2, 40],
    #                 'b': [2, 40],
    #                 'c': [2, 40]}),
    #     really_relax=None,
    #     behavioral_descriptor_names=None,
    #     n_relaxation_steps=500,
    # )
    #
    # with open(centroids_file, "r") as f:
    #     all_centroids = np.loadtxt(f)
    # kdt = KDTree(all_centroids, leaf_size=30, metric='euclidean')
    #
    # centroids = []
    # for i in range(len(relaxed_archive)):
    #     niche_index = kdt.query([[bds_batch[0][i], bds_batch[1][i]]], k=1)[1][0][0]
    #     niche = kdt.data[niche_index]
    #     n = make_hashable(niche)
    #     centroids.append(n)
    #
    # new_descriptors = np.array(bds_batch).T
    #
    # with open(directory / experiment_tag / f"relaxed_archive_{archive_number}.pkl", "wb") as file:
    #     pickle.dump([energy_batch, centroids, new_descriptors, relaxed_structures_as_dict, relaxed_archive], file)
    #

    fitnesses, centroids, descriptors, individuals = load_archive_from_pickle(archive_filename)

    relaxed_archive = []
    energies = []
    new_descriptors = []

    structure_optimizer = MultiprocessOptimizer()
    individuals = [Atoms.fromdict(individual) for individual in individuals]
    individuals_in_batches = [individuals[timing_batches_start[i]:timing_batches_end[i]] for i in range(len(timing_batches_end))]
    relaxed_archive = []
    marta_timings = []
    for individual_batch in tqdm(individuals_in_batches):
        tic = time.time()
        relaxation_results, _ = structure_optimizer.relax(individual_batch, n_relaxation_steps=500)
        marta_timings.append(time.time() - tic)
        relaxed_archive += relaxation_results
        with open(directory / experiment_tag / f"backup_relaxed_archive_marta_method{archive_number}.pkl",
                  "wb") as file:
            pickle.dump(relaxed_archive, file)

    relaxed_structures_as_dict = [AseAtomsAdaptor.get_atoms(result["final_structure"]).todict() for
                                  result in relaxed_archive]
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

    with open(directory / experiment_tag / f"relaxed_archive_marta_method{archive_number}.pkl", "wb") as file:
        pickle.dump(
            [energy_batch, centroids, new_descriptors, relaxed_structures_as_dict, relaxed_archive],
            file)

    batch_sizes = timing_batches_end - timing_batches_start
    plt.plot(batch_sizes, [10, 100, 8000, 4000], label="ASE")
    plt.plot(batch_sizes, marta_timings, label="Marta")
    plt.xlabel("Number of individuals_relaxed")
    plt.xticks(batch_sizes)
    plt.ylabel("Time taken, s")
    plt.legend()
    plt.savefig(str(directory / experiment_tag / f"relaxation_timings.png"), format="png")
