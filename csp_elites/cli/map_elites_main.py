import json
import os
import pathlib
import sys
import time
from dataclasses import asdict
from typing import Optional

from csp_elites.crystal.crystal_evaluator import CrystalEvaluator
from csp_elites.crystal.crystal_system import CrystalSystem
from csp_elites.map_elites.cvt_csp import CVT
from csp_elites.map_elites.elites_utils import __centroids_filename
from csp_elites.map_elites.elites_utils import make_current_time_string
from csp_elites.utils.experiment_parameters import ExperimentParameters
from retrieve_results.experiment_processing import ExperimentProcessor


class HiddenPrints:
    # from https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    def __init__(self, hide_prints: bool = True):
        self.hide_prints = hide_prints

    def __enter__(self):
        if self.hide_prints:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
        else:
            sys.stdout = sys.__stdout__

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hide_prints:
            sys.stdout.close()
            sys.stdout = self._original_stdout
        else:
            sys.stdout = sys.__stdout__


def main(
    experiment_parameters: ExperimentParameters,
    hide_prints: bool = False,
    from_archive_path: Optional[pathlib.Path] = None,
):
    with HiddenPrints(hide_prints=hide_prints):
        current_time_label = make_current_time_string(with_time=True)
        experiment_label = f"{current_time_label}_{experiment_parameters.system_name}_{experiment_parameters.experiment_tag}"

        print(f"Experiment data is saved in {experiment_label}")
        alternative_operators = (
            experiment_parameters.cvt_run_parameters["alternative_operators"]
            if "alternative_operators"
            in experiment_parameters.cvt_run_parameters.keys()
            else None
        )
        learning_rate = (
            experiment_parameters.cvt_run_parameters["dqd_learning_rate"]
            if "dqd_learning_rate" in experiment_parameters.cvt_run_parameters.keys()
            else 0.0001
        )

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

        force_threshold = (
            experiment_parameters.cvt_run_parameters["force_threshold"]
            if "force_threshold" in experiment_parameters.cvt_run_parameters.keys()
            else False
        )
        constrained_qd = (
            experiment_parameters.cvt_run_parameters["constrained_qd"]
            if "constrained_qd" in experiment_parameters.cvt_run_parameters.keys()
            else False
        )

        force_threshold_exp_fmax = (
            experiment_parameters.cvt_run_parameters["force_threshold_exp_fmax"]
            if "force_threshold_exp_fmax"
            in experiment_parameters.cvt_run_parameters.keys()
            else 1.0
        )

        fmax_threshold = (
            experiment_parameters.cvt_run_parameters["fmax_threshold"]
            if "fmax_threshold" in experiment_parameters.cvt_run_parameters.keys()
            else 0.2
        )

        normalise_bd = (
            experiment_parameters.cvt_run_parameters["normalise_bd"]
            if "normalise_bd" in experiment_parameters.cvt_run_parameters.keys()
            else False
        )

        compute_gradients = (
            experiment_parameters.cvt_run_parameters["dqd"]
            if "dqd" in experiment_parameters.cvt_run_parameters.keys()
            else False
        )
        crystal_evaluator = CrystalEvaluator(
            with_force_threshold=force_threshold,
            constrained_qd=constrained_qd,
            fmax_relaxation_convergence=fmax_threshold,
            force_threshold_fmax=force_threshold_exp_fmax,
            compute_gradients=compute_gradients,
            bd_normalisation=(
                experiment_parameters.cvt_run_parameters["bd_minimum_values"],
                experiment_parameters.cvt_run_parameters["bd_maximum_values"],
            )
            if normalise_bd
            else None,
        )

        cvt = CVT(
            number_of_bd_dimensions=experiment_parameters.n_behavioural_descriptor_dimensions,
            crystal_system=crystal_system,
            crystal_evaluator=crystal_evaluator,
        )

        tic = time.time()

        if from_archive_path is not None:
            print("Running from archive")
            experiment_directory_path, archive = cvt.start_experiment_from_archive(
                experiment_to_load_directory_path=from_archive_path,
                number_of_niches=experiment_parameters.number_of_niches,
                maximum_evaluations=experiment_parameters.maximum_evaluations,
                run_parameters=experiment_parameters.cvt_run_parameters,
                experiment_label=experiment_label,
            )
        else:
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

        if experiment_parameters.cvt_run_parameters["normalise_bd"]:
            bd_minimum_values, bd_maximum_values = [0, 0], [1, 1]
        else:
            bd_minimum_values, bd_maximum_values = (
                experiment_parameters.cvt_run_parameters["bd_minimum_values"],
                experiment_parameters.cvt_run_parameters["bd_maximum_values"],
            )

        centroid_filename = __centroids_filename(
            k=experiment_parameters.number_of_niches,
            dim=experiment_parameters.n_behavioural_descriptor_dimensions,
            bd_names=experiment_parameters.cvt_run_parameters[
                "behavioural_descriptors"
            ],
            bd_minimum_values=bd_minimum_values,
            bd_maximum_values=bd_maximum_values,
            formula=experiment_parameters.system_name,
        )
        # centroid_filename = f"{pathlib.Path(experiment_directory_path).parent}{centroid_filename}"

        experiment_processor = ExperimentProcessor(
            experiment_label=experiment_label,
            config_filepath=experiment_parameters,
            centroid_filename=centroid_filename,
            fitness_limits=experiment_parameters.fitness_min_max_values,
            save_structure_images=False,
            filter_for_experimental_structures=False,
            experiment_location=pathlib.Path(__file__).parent.parent.parent,
        )

        experiment_processor.plot()
        experiment_processor.process_symmetry()
