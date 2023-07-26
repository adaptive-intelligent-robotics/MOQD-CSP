import json
import os
import pathlib
import time
from dataclasses import asdict
from ase.ga.ofp_comparator import OFPComparator
from ase.ga.utilities import CellBounds

from csp_elites.utils.asign_target_values_to_centroids import \
    compute_centroids_for_target_solutions, reassign_data_from_pkl_to_new_centroids
from csp_elites.crystal.crystal_evaluator import CrystalEvaluator
from csp_elites.crystal.materials_data_model import MaterialProperties
from csp_elites.crystal.crystal_system import CrystalSystem
from csp_elites.map_elites.cvt_csp import CVT
from csp_elites.utils.experiment_parameters import ExperimentParameters
from csp_elites.utils.get_mpi_structures import get_all_materials_with_formula
from csp_elites.utils.plot import load_archive_from_pickle, load_centroids, \
    plot_2d_map_elites_repertoire_marta, convert_fitness_and_ddescriptors_to_plotting_format, \
    plot_all_statistics_from_file, plot_all_maps_in_archive
# from csp_elites.plot import load_centroids, load_archive_from_pickle
from csp_elites.map_elites.elites_utils import make_current_time_string, __centroids_filename

if __name__ == '__main__':

    experiment_directory_path = "/Users/marta/Documents/MSc Artificial Intelligence/Thesis/csp-elites/experiments/20230726_01_33_TiO2_local_115k"
    experiment_label = "20230726_01_33_TiO2_local_115k"
    archive_number = 68000

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    # archive = load_archive_from_pickle("../experiments/20230722_10_09_TiO2_100k_1000_niches/archive_61100.pkl")

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    experiment_parameters = ExperimentParameters(
        number_of_niches=200,
        maximum_evaluations=100000,
        experiment_tag="timing_1k",
        fitler_comparison_data_for_n_atoms=24,
        cvt_run_parameters= \
            {
                # more of this -> higher-quality CVT
                "cvt_samples": 25000,
                # we evaluate in batches to paralleliez
                "batch_size": 10,
                # proportion of niches to be filled before starting
                "random_init": 0.05,
                # batch for random initialization
                "random_init_batch": 3,
                # when to write results (one generation = one batch)
                "dump_period": 1000,
                # do we use several cores?
                "parallel": True,
                # do we cache the result of CVT and reuse?
                "cvt_use_cache": True,
                # min/max of parameters
                "bd_minimum_values": (0, 0),
                "bd_maximum_values": (100, 100),
                "relaxation_probability": 0,
                "behavioural_descriptors": [MaterialProperties.BAND_GAP,
                                            MaterialProperties.SHEAR_MODULUS],
                "number_of_relaxation_steps": 0,
                "curiosity_weights": True,
                "filter_starting_Structures": 24,
            },
        system_name="TiO2",
        blocks=[22] * 8 + [8] * 16,
        volume=450,
        ratio_of_covalent_radii=0.4,
        splits={(2,): 1, (4,): 1},
        cellbounds=CellBounds(
            bounds={'phi': [20, 160], 'chi': [20, 160], 'psi': [20, 160], 'a': [2, 40],
                    'b': [2, 40],
                    'c': [2, 40]}),
        operator_probabilities=[5., 0, 2., 3.],
        ### CVT PARAMETERS ###
        n_behavioural_descriptor_dimensions=2,
        fitness_min_max_values=[0, 10],
    )

    crystal_system = CrystalSystem(
        atom_numbers_to_optimise=experiment_parameters.blocks,
        volume=experiment_parameters.volume,
        ratio_of_covalent_radii=experiment_parameters.ratio_of_covalent_radii,
        splits=experiment_parameters.splits,
        operator_probabilities=experiment_parameters.operator_probabilities,
    )

    comparator = OFPComparator(n_top=len(experiment_parameters.blocks), dE=1.0,
                               cos_dist_max=1e-3, rcut=10., binwidth=0.05,
                               pbc=[True, True, True], sigma=0.05, nsigma=4,
                               recalculate=False)

    crystal_evaluator = CrystalEvaluator(comparator=comparator)

    cvt = CVT(
        number_of_bd_dimensions=experiment_parameters.n_behavioural_descriptor_dimensions,
        crystal_system=crystal_system,
        crystal_evaluator=crystal_evaluator,
    )

    cvt.start_experiment_from_archive(
        experiment_directory_path=experiment_directory_path,
        experiment_label=experiment_label,
        archive_number= archive_number,
        run_parameters=experiment_parameters.cvt_run_parameters,
        number_of_niches=experiment_parameters.number_of_niches,
        maximum_evaluations=experiment_parameters.maximum_evaluations,
    )


    print()
