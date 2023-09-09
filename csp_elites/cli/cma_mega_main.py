from csp_elites.cma_mega.cma_mega import CMAMEGA
from csp_elites.crystal.crystal_evaluator import CrystalEvaluator
from csp_elites.crystal.crystal_system import CrystalSystem
from csp_elites.utils.experiment_parameters import ExperimentParameters


def main_cma(experiment_parameters: ExperimentParameters):
    alternative_operators = (
        experiment_parameters.cvt_run_parameters["alternative_operators"]
        if "alternative_operators" in experiment_parameters.cvt_run_parameters.keys()
        else None
    )

    crystal_system = CrystalSystem(
        atom_numbers_to_optimise=experiment_parameters.blocks,
        volume=experiment_parameters.volume,
        ratio_of_covalent_radii=experiment_parameters.ratio_of_covalent_radii,
        splits=experiment_parameters.splits,
        operator_probabilities=experiment_parameters.operator_probabilities,
        start_generator=experiment_parameters.start_generator,
        alternative_operators=alternative_operators,
        learning_rate=None,
    )

    crystal_evaluator = CrystalEvaluator(
        with_force_threshold=False,
        constrained_qd=False,
        fmax_relaxation_convergence=0.2,
        force_threshold_fmax=False,
        compute_gradients=True,
    )

    cma = CMAMEGA(
        number_of_bd_dimensions=2,
        crystal_system=crystal_system,
        crystal_evaluator=crystal_evaluator,
        step_size_gradient_optimizer_niu=experiment_parameters.cvt_run_parameters[
            "cma_learning_rate"
        ],
        initial_cmaes_step_size_sigma_g=experiment_parameters.cvt_run_parameters[
            "cma_sigma_0"
        ],
    )

    cma.compute(
        experiment_parameters.number_of_niches,
        experiment_parameters.maximum_evaluations,
        experiment_parameters.cvt_run_parameters,
        experiment_label=experiment_parameters.experiment_tag,
    )
