import json
import os
import pathlib
import time
import tracemalloc
from dataclasses import asdict

import psutil
from ase.ga.ofp_comparator import OFPComparator

from csp_elites.crystal.crystal_evaluator import CrystalEvaluator
from csp_elites.crystal.crystal_system import CrystalSystem
from csp_elites.crystal.materials_data_model import MaterialProperties
from csp_elites.map_elites.cvt_csp import CVT
from csp_elites.utils.asign_target_values_to_centroids import \
    compute_centroids_for_target_solutions, reassign_data_from_pkl_to_new_centroids
from csp_elites.utils.experiment_parameters import ExperimentParameters
from csp_elites.utils.get_mpi_structures import get_all_materials_with_formula
from csp_elites.utils.plot import load_centroids, load_archive_from_pickle, \
    plot_all_maps_in_archive, plot_all_statistics_from_file
# from csp_elites.plot import load_centroids, load_archive_from_pickle
from csp_elites.map_elites.elites_utils import __centroids_filename, make_current_time_string
import json
import pathlib
import time
from dataclasses import asdict

import psutil


from csp_elites.map_elites.elites_utils import __centroids_filename
from csp_elites.utils.experiment_parameters import ExperimentParameters
from csp_elites.utils.plot import load_centroids


def main(experiment_parameters: ExperimentParameters):
    # print(experiment_parameters)
    ### CODE TO RUN
    print(f"Memory before object creation {psutil.virtual_memory()[3] / 1000000000}")

    current_time_label = make_current_time_string(with_time=True)
    experiment_label = \
        f"{current_time_label}_{experiment_parameters.system_name}_{experiment_parameters.experiment_tag}"

    crystal_system = CrystalSystem(
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

    crystal_evaluator = CrystalEvaluator(comparator=comparator)

    cvt = CVT(
        number_of_bd_dimensions=experiment_parameters.n_behavioural_descriptor_dimensions,
        crystal_system=crystal_system,
        crystal_evaluator=crystal_evaluator,
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

    experiment_parameters.splits = "DUMMY"
    experiment_parameters.cellbounds = "DUMMY"
    with open(f"{experiment_directory_path}/config.json", "w") as file:
        json.dump(asdict(experiment_parameters), file)

    # # Variables setting

    centroid_filename = __centroids_filename(
        k=experiment_parameters.number_of_niches,
        dim=experiment_parameters.n_behavioural_descriptor_dimensions,
        bd_names=experiment_parameters.cvt_run_parameters["behavioural_descriptors"],
        bd_minimum_values=experiment_parameters.cvt_run_parameters["bd_minimum_values"],
        bd_maximum_values=experiment_parameters.cvt_run_parameters["bd_maximum_values"],
    )
    centroid_filename = f"{pathlib.Path(experiment_directory_path).parent}{centroid_filename}"

    all_centroids = load_centroids(centroid_filename)

    # plot_fitness_from_file(fitness_plotting_filename)

    # ToDo: Pass target centroids in better
    if MaterialProperties.ENERGY_FORMATION in experiment_parameters.cvt_run_parameters["behavioural_descriptors"]:
        comparison_data = str(pathlib.Path(
            experiment_directory_path).parent.parent / "experiments/target_data/tio2_target_data.pkl")
        target_centroids = compute_centroids_for_target_solutions(
            centroids_file=centroid_filename,
            target_data_file=comparison_data,
            filter_for_number_of_atoms=experiment_parameters.fitler_comparison_data_for_n_atoms
        )
    else:
        comparison_data = str(pathlib.Path(
            experiment_directory_path).parent.parent / "experiments/target_data/ti02_band_gap_shear_modulus.pkl")
        comparison_data_packed = load_archive_from_pickle(comparison_data)
        target_centroids = reassign_data_from_pkl_to_new_centroids(
            centroids_file=centroid_filename,
            target_data=comparison_data_packed,
            filter_for_number_of_atoms=experiment_parameters.fitler_comparison_data_for_n_atoms
        )

    structure_info, known_structures = get_all_materials_with_formula(experiment_parameters.system_name)

    structures_for_comparison ={}
    for i, structure in enumerate(known_structures):
        if len(structure.get_atomic_numbers()) == experiment_parameters.fitler_comparison_data_for_n_atoms:
            structures_for_comparison[str(structure_info[i].material_id)] = structure

    plot_all_maps_in_archive(
        experiment_directory_path=experiment_directory_path,
        experiment_parameters=experiment_parameters,
        all_centroids=all_centroids,
        target_centroids=target_centroids,
    )

    plot_all_statistics_from_file(
        filename=f"{experiment_directory_path}/{experiment_label}.dat",
        save_location=f"{experiment_directory_path}/",
    )

    max_archive = max([int(name.lstrip("archive_").rstrip(".pkl")) for name in os.listdir(f"{experiment_directory_path}") if ((not os.path.isdir(name)) and ("archive_" in name))])

    archive_filename = f"{experiment_directory_path}/archive_{max_archive}.pkl"
    fitnesses, centroids, descriptors, individuals = load_archive_from_pickle(archive_filename)

    cvt.crystal_evaluator.compare_to_target_structures(
        generated_structures=individuals,
        target_structures=structures_for_comparison,
        directory_string=experiment_directory_path,
    )
