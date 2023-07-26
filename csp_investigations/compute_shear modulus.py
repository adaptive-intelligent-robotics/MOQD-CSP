import pickle

import matplotlib.pyplot as plt
import numpy as np
from ase.ga.ofp_comparator import OFPComparator
from ase.ga.utilities import CellBounds
from megnet.utils.models import load_model as megnet_load_model, load_model
from mp_api.client import MPRester
from pymatgen.core import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.neighbors import KDTree

from csp_elites.crystal.crystal_evaluator import MaterialProperties, CrystalEvaluator
from csp_elites.crystal.crystal_system import CrystalSystem
from csp_elites.map_elites.elites_utils import make_hashable
from csp_elites.utils.get_mpi_structures import return_structure_information_from_mp_api, \
    get_all_materials_with_formula
from csp_elites.utils.plot import plot_2d_map_elites_repertoire_marta, load_centroids, \
    convert_fitness_and_ddescriptors_to_plotting_format, load_archive_from_pickle
from csp_elites.utils.experiment_parameters import ExperimentParameters

if __name__ == '__main__':
    formula = "TiO2"
    blocks =  [22] * 8 + [8] * 16,
    comparator = OFPComparator(n_top=len(blocks), dE=1.0,
                               cos_dist_max=1e-3, rcut=10., binwidth=0.05,
                               pbc=[True, True, True], sigma=0.05, nsigma=4,
                               recalculate=False)

    crystal_evaluator = CrystalEvaluator(comparator=comparator)

    with MPRester(api_key="4nB757V2Puue49BqPnP3bjRPksr4J9y0") as mpr:
        docs = mpr.summary.search(formula=formula,
                                  # band_gap=(0.5, 1.0),
                                  fields=["material_id",
                                          "band_gap",
                                          "volume", "lattice", "formation_energy_per_atom",
                                          "energy_above_hull",
                                          "structure",
                                          'k_voigt', 'k_reuss', 'k_vrh', 'g_voigt', 'g_reuss',
                                          'g_vrh'
                                          ])

    # load a model in megnet.utils.models.AVAILABLE_MODELS
    model = load_model("logG_MP_2018")

    energies = []
    predicted_gs = []
    predicted_band_gaps = []
    queried_gs = []
    all_individuals = []
    for mp_structure in docs:
        structure = mp_structure.structure
        predicted_G = crystal_evaluator.compute_shear_modulus(structure)
        predicted_band_gap = crystal_evaluator.compute_band_gap(structure)
        predicted_band_gaps.append(predicted_band_gap)

        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.calc = None
        all_individuals.append(atoms)
        energy = crystal_evaluator.compute_energy(atoms, really_relax=False)
        energies.append(energy[0])
        # predicted_G = 10 ** model.predict_structure(structure).ravel()[0]
        predicted_gs.append(predicted_G)
        print(f'The predicted K for {structure.composition.reduced_formula} is {predicted_G:.0f} GPa.')
        queried_shear_modulus = mp_structure.g_vrh

        if queried_shear_modulus is None:
            print(predicted_G)
        else:
            queried_gs.append(queried_shear_modulus)
            print(f"Error {(queried_shear_modulus - predicted_G) / queried_shear_modulus}")

    # plt.hist(predicted_gs)
    # plt.show()
    #
    # plt.hist(queried_gs)
    # plt.show()
    # print()

    all_centroids = load_centroids(
        "../csp_experiments/experiments/centroids/centroids_200_2_energy_formation_0_100_shear_modulus_0_100.dat")

    kdt = KDTree(all_centroids, leaf_size=30, metric='euclidean')

    centroids = []

    for i in range(len(energies)):
        niche_index = kdt.query([(predicted_band_gaps[i], predicted_gs[i])], k=1)[1][0][0]
        niche = kdt.data[niche_index]
        n = make_hashable(niche)
        centroids.append(n)

    # fitnesses_for_plotting, descriptors_for_plotting = convert_fitness_and_ddescriptors_to_plotting_format(
    #     all_centroids=all_centroids,
    #     centroids_from_archive=centroids,
    #     fitnesses_from_archive=energies,
    #     descriptors_from_archive=np.vstack([predicted_band_gaps, predicted_gs]).T,
    # )
    #
    # plot_2d_map_elites_repertoire_marta(
    #     centroids=all_centroids,
    #     repertoire_fitnesses=fitnesses_for_plotting,
    #     minval=[0, 0],
    #     maxval=[100, 100],
    #     repertoire_descriptors=descriptors_for_plotting,
    #     vmin=0,
    #     vmax=10,
    #     target_centroids=None,
    #     directory_string=None,
    #     filename=None,
    # )

    # reshape for pkl

    # all_data = []
    # for i in range(len(energies)):
    #     one_data_point = []
    #     one_data_point.append(energies[i])
    #     one_data_point.append(centroids[i])
    #     descriptor = (predicted_band_gaps[i], predicted_gs[i])
    #     one_data_point.append(descriptor)
    #     one_data_point.append(all_individuals[i])
    #
    #     all_data.append(one_data_point)

    # with open("../experiments/target_data/ti02_band_gap_shear_modulus.pkl", "wb") as file:
    #     pickle.dump(all_data, file)
    #
    # fitnesses, centroids, descriptors, individuals = \
    #     load_archive_from_pickle("../experiments/target_data/ti02_band_gap_shear_modulus.pkl")

    print()
