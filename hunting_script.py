import json
import sys

from ase.ga.ofp_comparator import OFPComparator

import time

import psutil
from ase.ga.ofp_comparator import OFPComparator
from ase.ga.utilities import CellBounds

from csp_elites.crystal.materials_data_model import MaterialProperties, StartGenerators
from csp_elites.map_elites.elites_utils import make_current_time_string
from csp_elites.utils.experiment_parameters import ExperimentParameters
from memory_hunting.crystal.crystal_evaluator import CrystalEvaluatorHunting
from memory_hunting.crystal.crystal_system import CrystalSystemHunting
from memory_hunting.map_elites.cvt_csp import CVTHunting


def hunting_main(experiment_parameters: ExperimentParameters,
                 remove_energy_model=False,
                 remove_band_gap_model=False,
                 remove_shear_model=False,
                 no_check_population_to_kill=False,
                 no_mutation=False,
                 ):
    print(f"Memory before object creation {psutil.virtual_memory()[3] / 1000000000}")

    current_time_label = make_current_time_string(with_time=True)
    experiment_label = \
        f"{current_time_label}_{experiment_parameters.system_name}_{experiment_parameters.experiment_tag}"

    crystal_system = CrystalSystemHunting(
        atom_numbers_to_optimise=experiment_parameters.blocks,
        volume=experiment_parameters.volume,
        ratio_of_covalent_radii=experiment_parameters.ratio_of_covalent_radii,
        splits=experiment_parameters.splits,
        operator_probabilities=experiment_parameters.operator_probabilities,
        start_generator=experiment_parameters.start_generator,

    )

    comparator = OFPComparator(n_top=len(experiment_parameters.blocks), dE=1.0,
                               cos_dist_max=1e-3, rcut=10., binwidth=0.05,
                               pbc=[True, True, True], sigma=0.05, nsigma=4,
                               recalculate=False)

    crystal_evaluator = CrystalEvaluatorHunting(comparator=comparator,
                                                remove_energy_model=remove_energy_model,
                                                remove_band_gap_model=remove_band_gap_model,
                                                remove_shear_model=remove_shear_model,
                                                no_check_population_to_kill=no_check_population_to_kill,
                                                )

    cvt = CVTHunting(
        number_of_bd_dimensions=experiment_parameters.n_behavioural_descriptor_dimensions,
        crystal_system=crystal_system,
        crystal_evaluator=crystal_evaluator,
        remove_mutations=no_mutation
    )
    # snapshot = tracemalloc.take_snapshot()

    print(f"Memory after all object created, before compute loop {psutil.virtual_memory()[3]/1000000000}")
    tic = time.time()
    experiment_directory_path, archive = cvt.batch_compute_with_list_of_atoms(
        number_of_niches=experiment_parameters.number_of_niches,
        maximum_evaluations=experiment_parameters.maximum_evaluations,
        run_parameters=experiment_parameters.cvt_run_parameters,
        experiment_label=experiment_label,
    )
    print(f"time taken {time.time() - tic}")



if __name__ == '__main__':
    file_location = "configs/200_niches_100k_for_benchmark.json"

    with open(file_location, "r") as file:
        experiment_parameters = json.load(file)

    profiling_parameters_file = sys.argv[1]

    with open(profiling_parameters_file, "r") as file:
        profiling_paramters = json.load(file)

    remove_energy_model, remove_band_gap_model, remove_shear_model, no_check_population_to_kill, no_mutation, tag = list(profiling_paramters.values())

    experiment_parameters = ExperimentParameters(**experiment_parameters)
    experiment_parameters.cellbounds = CellBounds(
            bounds={'phi': [20, 160], 'chi': [20, 160], 'psi': [20, 160], 'a': [2, 40], 'b': [2, 40],
                    'c': [2, 40]}),
    experiment_parameters.splits = {(2,): 1, (4,): 1}
    experiment_parameters.cvt_run_parameters["behavioural_descriptors"] = \
        [MaterialProperties(value) for value in experiment_parameters.cvt_run_parameters["behavioural_descriptors"]]

    experiment_parameters.start_generator = StartGenerators(experiment_parameters.start_generator)
    experiment_parameters.experiment_tag = tag

    hunting_main(experiment_parameters, remove_energy_model=remove_energy_model,
    remove_band_gap_model=remove_band_gap_model,
    remove_shear_model=remove_shear_model,
    no_check_population_to_kill=no_check_population_to_kill,
                 no_mutation=no_mutation)
