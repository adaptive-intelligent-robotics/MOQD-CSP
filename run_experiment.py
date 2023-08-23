import json
import os
import pathlib
import time
import tracemalloc
from dataclasses import asdict

import psutil
from ase.ga.ofp_comparator import OFPComparator
from ase.ga.utilities import CellBounds

from csp_elites.utils.asign_target_values_to_centroids import \
    compute_centroids_for_target_solutions, reassign_data_from_pkl_to_new_centroids
from csp_elites.crystal.crystal_evaluator import CrystalEvaluator
from csp_elites.crystal.materials_data_model import MaterialProperties, StartGenerators
from csp_elites.crystal.crystal_system import CrystalSystem
from csp_elites.map_elites.cvt_csp import CVT
from csp_elites.utils.experiment_parameters import ExperimentParameters
from main import main

# from csp_elites.utils.get_mpi_structures import get_all_materials_with_formula
# from csp_elites.utils.plot import load_archive_from_pickle, load_centroids, \
#     plot_2d_map_elites_repertoire_marta, convert_fitness_and_ddescriptors_to_plotting_format, \
#     plot_all_statistics_from_file, plot_all_maps_in_archive
# # from csp_elites.plot import load_centroids, load_archive_from_pickle
# from csp_elites.map_elites.elites_utils import make_current_time_string, __centroids_filename

if __name__ == '__main__':
    import pyximport
    pyximport.install(setup_args={"script_args": ["--verbose"]})
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    tracemalloc.start()
    experiment_parameters = ExperimentParameters(
        number_of_niches=200,
        maximum_evaluations=100,
        experiment_tag="test_200_with_dels",
        fitler_comparison_data_for_n_atoms=24,
        start_generator=StartGenerators.PYXTAL,
        cvt_run_parameters= \
            {
                # more of this -> higher-quality CVT
                "cvt_samples": 25000,
                # we evaluate in batches to paralleliez
                "batch_size": 10,
                # proportion of niches to be filled before starting
                "random_init": 0,
                # batch for random initialization
                "random_init_batch": 20,
                # when to write results (one generation = one batch)
                "dump_period":  2,
                # do we use several cores?
                "parallel": True,
                # do we cache the result of CVT and reuse?
                "cvt_use_cache": True,
                # min/max of parameters
                "bd_minimum_values": (0, 0),
                "bd_maximum_values": (100, 100),
                "relaxation_probability": 0,
                "behavioural_descriptors": [MaterialProperties.BAND_GAP, MaterialProperties.SHEAR_MODULUS],
                "number_of_relaxation_steps": 4,
                "curiosity_weights": True,
                "filter_starting_Structures": 24,
                "seed": False,
                "profiling": False,

            },
        system_name="TiO2",
        blocks = [22] * 8 + [8] * 16,
        volume=450,
        ratio_of_covalent_radii = 0.4,
        splits={(2,): 1, (4,): 1},
        cellbounds = CellBounds(
            bounds={'phi': [20, 160], 'chi': [20, 160], 'psi': [20, 160], 'a': [2, 40], 'b': [2, 40],
                    'c': [2, 40]}),
        operator_probabilities=[5., 0, 2., 3.],
        ### CVT PARAMETERS ###
        n_behavioural_descriptor_dimensions=2,
        fitness_min_max_values=[0, 10],
    )

    main(experiment_parameters)
