from ase.ga.ofp_comparator import OFPComparator

from csp_elites.crystal.crystal_evaluator import CrystalEvaluator
from csp_elites.utils.plot import load_archive_from_pickle

if __name__ == '__main__':
    archive_filename = "../experiments/20230712_13_39_TiO2_dummy/archive_4.pkl"
    experiments_blocks = [22] * 8 + [8] * 16

    fitness, centroids, descriptors, individuals = load_archive_from_pickle(archive_filename)

    comparator = OFPComparator(n_top=len(experiments_blocks), dE=1.0,
                               cos_dist_max=1e-3, rcut=10., binwidth=0.05,
                               pbc=[True, True, True], sigma=0.05, nsigma=4,
                               recalculate=False)

    crystal_evaluator = CrystalEvaluator(comparator=comparator)

    for i, ind in enumerate(individuals):
        print(fitness[i])
        energy, _ = crystal_evaluator.compute_energy(ind, really_relax=True)
        print(energy)
    # print()
