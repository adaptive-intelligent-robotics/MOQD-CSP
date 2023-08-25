import os
import json
import os
import pathlib
import sys
import time
from dataclasses import asdict

import psutil
import torch
from ase.ga.ofp_comparator import OFPComparator

from csp_elites.crystal.crystal_evaluator import CrystalEvaluator
from csp_elites.crystal.crystal_system import CrystalSystem
from csp_elites.map_elites.cvt_csp import CVT
from csp_elites.map_elites.elites_utils import __centroids_filename
from csp_elites.map_elites.elites_utils import make_current_time_string
from csp_elites.utils.experiment_parameters import ExperimentParameters
from csp_elites.utils.plot import load_centroids


class HiddenPrints:
    # from https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    def __init__(self, hide_prints: bool = True):
        self.hide_prints = hide_prints

    def __enter__(self):
        if self.hide_prints:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        else:
            sys.stdout = sys.__stdout__

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hide_prints:
            sys.stdout.close()
            sys.stdout = self._original_stdout
        else:
            sys.stdout = sys.__stdout__


def main(experiment_parameters: ExperimentParameters, hide_prints: bool=False):
    # print(experiment_parameters)
    ### CODE TO RUN
    print(torch.cuda.is_available())
    with HiddenPrints(hide_prints=hide_prints):
        print(f"Memory before object creation {psutil.virtual_memory()[3] / 1000000000}")

        current_time_label = make_current_time_string(with_time=True)
        experiment_label = \
            f"{current_time_label}_{experiment_parameters.system_name}_{experiment_parameters.experiment_tag}"

        print(experiment_label)
        alternative_operators = experiment_parameters.cvt_run_parameters["alternative_operators"] if "alternative_operators" in experiment_parameters.cvt_run_parameters.keys() else None
        learning_rate = experiment_parameters.cvt_run_parameters["dqd_learning_rate"] if "dqd_learning_rate" in experiment_parameters.cvt_run_parameters.keys() else 0.0001


        crystal_system = CrystalSystem(
            atom_numbers_to_optimise=experiment_parameters.blocks,
            volume=experiment_parameters.volume,
            ratio_of_covalent_radii=experiment_parameters.ratio_of_covalent_radii,
            splits=experiment_parameters.splits,
            compound_formula=experiment_parameters.system_name,
            operator_probabilities=experiment_parameters.operator_probabilities,
            start_generator=experiment_parameters.start_generator,
            alternative_operators=alternative_operators,
            learning_rate=learning_rate,
        )

        comparator = OFPComparator(n_top=len(experiment_parameters.blocks), dE=1.0,
                                   cos_dist_max=1e-3, rcut=10., binwidth=0.05,
                                   pbc=[True, True, True], sigma=0.05, nsigma=4,
                                   recalculate=False)

        force_threshold = experiment_parameters.cvt_run_parameters["force_threshold"] if "force_threshold" in experiment_parameters.cvt_run_parameters.keys() else False
        constrained_qd = experiment_parameters.cvt_run_parameters["constrained_qd"] if "constrained_qd" in experiment_parameters.cvt_run_parameters.keys() else False

        force_threshold_exp_fmax= experiment_parameters.cvt_run_parameters[
            "force_threshold_exp_fmax"] if "force_threshold_exp_fmax" in experiment_parameters.cvt_run_parameters.keys() else 1.0

        fmax_threshold = experiment_parameters.cvt_run_parameters[
            "fmax_threshold"] if "fmax_threshold" in experiment_parameters.cvt_run_parameters.keys() else 0.2


        compute_gradients = experiment_parameters.cvt_run_parameters["dqd"] if "dqd" in experiment_parameters.cvt_run_parameters.keys() else False
        crystal_evaluator = CrystalEvaluator(
            comparator=comparator,
            with_force_threshold=force_threshold,
            constrained_qd=constrained_qd,
            fmax_relaxation_convergence=fmax_threshold,
            force_threshold_fmax=force_threshold_exp_fmax,
            compute_gradients=compute_gradients,
        )

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
            formula=experiment_parameters.system_name,
        )
        centroid_filename = f"{pathlib.Path(experiment_directory_path).parent}{centroid_filename}"

        all_centroids = load_centroids(centroid_filename)

        # plot_fitness_from_file(fitness_plotting_filename)

        # ToDo: Pass target centroids in better
        # if MaterialProperties.ENERGY_FORMATION in experiment_parameters.cvt_run_parameters["behavioural_descriptors"]:
        #     comparison_data = str(pathlib.Path(
        #         experiment_directory_path).parent.parent / "experiments/target_data/tio2_target_data.pkl")
        #     target_centroids = compute_centroids_for_target_solutions(
        #         centroids_file=centroid_filename,
        #         target_data_file=comparison_data,
        #         filter_for_number_of_atoms=experiment_parameters.fitler_comparison_data_for_n_atoms
        #     )
        # else:
        #     comparison_data = str(pathlib.Path(
        #         experiment_directory_path).parent.parent / "experiments/target_data/ti02_band_gap_shear_modulus.pkl")
        #     comparison_data_packed = load_archive_from_pickle(comparison_data)
        #     target_centroids = reassign_data_from_pkl_to_new_centroids(
        #         centroids_file=centroid_filename,
        #         target_data=comparison_data_packed,
        #         filter_for_number_of_atoms=experiment_parameters.fitler_comparison_data_for_n_atoms
        #     )
        #
        # structure_info, known_structures = get_all_materials_with_formula(experiment_parameters.system_name)
        #
        # structures_for_comparison ={}
        # for i, structure in enumerate(known_structures):
        #     if len(structure.get_atomic_numbers()) == experiment_parameters.fitler_comparison_data_for_n_atoms:
        #         structures_for_comparison[str(structure_info[i].material_id)] = structure
        #
        # plot_all_maps_in_archive(
        #     experiment_directory_path=experiment_directory_path,
        #     experiment_parameters=experiment_parameters,
        #     all_centroids=all_centroids,
        #     target_centroids=target_centroids,
        # )
        #
        # plot_all_statistics_from_file(
        #     filename=f"{experiment_directory_path}/{experiment_label}.dat",
        #     save_location=f"{experiment_directory_path}/",
        # )
        #
        # max_archive = max([int(name.lstrip("archive_").rstrip(".pkl")) for name in os.listdir(f"{experiment_directory_path}") if ((not os.path.isdir(name)) and ("archive_" in name))])
        #
        # archive_filename = f"{experiment_directory_path}/archive_{max_archive}.pkl"
        # fitnesses, centroids, descriptors, individuals = load_archive_from_pickle(archive_filename)
        #
        # symmetry_evaluator = SymmetryEvaluation()




        # cvt.crystal_evaluator.compare_to_target_structures(
        #     generated_structures=[Atoms.fromdict(individual) for individual in individuals],
        #     target_structures=structures_for_comparison,
        #     directory_string=experiment_directory_path,
        # )
