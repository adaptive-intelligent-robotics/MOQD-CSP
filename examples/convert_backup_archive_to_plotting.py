import pathlib
import pickle

import numpy as np
from ase.ga.utilities import CellBounds
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.neighbors import KDTree

from csp_elites.crystal.crystal_evaluator import CrystalEvaluator
from csp_elites.map_elites.elites_utils import make_hashable



def convert_back_up_relaxed_archive_comparison_format(backup_archive, centroids_file):
    crystal_evaluator = CrystalEvaluator(comparator=None)
    with open(backup_archive, "rb") as file:
        relaxed_archive = pickle.load(file)

    relaxed_structures_as_dict = [AseAtomsAdaptor.get_atoms(result["final_structure"]).todict() for
                                 result in relaxed_archive]
    _,   new_atoms_dict, energy_batch, bds_batch, _, _= crystal_evaluator.batch_compute_fitness_and_bd(
        list_of_atoms=relaxed_structures_as_dict,
        cellbounds=CellBounds(
            bounds={'phi': [20, 160], 'chi': [20, 160], 'psi': [20, 160], 'a': [2, 40],
                    'b': [2, 40],
                    'c': [2, 40]}),
        behavioral_descriptor_names=None,
        really_relax=None,
        n_relaxation_steps=0    ,
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
    return energy_batch, centroids, new_descriptors, relaxed_structures_as_dict, relaxed_archive


if __name__ == '__main__':
    folder = pathlib.Path(__file__).parent.parent / ".experiment.nosync" / "experiments"
    centroids_file = folder / "centroids" / "centroids_200_2_band_gap_0_100_shear_modulus_0_120.dat"
    experiment_tag = "20230803_03_41_TiO2_200_niches_1M_force_threshold_pyxtal"
    backup_archive_number = 96040

    backup_archive_full = folder / experiment_tag / f"backup_relaxed_archive_{backup_archive_number}.pkl"

    data_to_save = convert_back_up_relaxed_archive_comparison_format(backup_archive_full, centroids_file)

    with open(folder / experiment_tag / f"relaxed_archive_{backup_archive_number}.pkl", "wb") as file:
        pickle.dump(data_to_save, file)
