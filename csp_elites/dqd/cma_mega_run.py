print("inside main cma")
from ase.ga.ofp_comparator import OFPComparator
from ase.ga.utilities import CellBounds

print("evaluator")
from csp_elites.crystal.crystal_evaluator import CrystalEvaluator
print("system")
from csp_elites.crystal.crystal_system import CrystalSystem
print("mega loop")
from csp_elites.dqd.cma_mega_loop import CMAMEGALOOP
print("exp parameters")
from csp_elites.map_elites.elites_utils import make_current_time_string
from csp_elites.utils.experiment_parameters import ExperimentParameters



def main_cma(experiment_parameters: ExperimentParameters):
    alternative_operators = experiment_parameters.cvt_run_parameters[
        "alternative_operators"] if "alternative_operators" in experiment_parameters.cvt_run_parameters.keys() else None

    comparator = OFPComparator(n_top=len(experiment_parameters.blocks), dE=1.0,
                               cos_dist_max=1e-3, rcut=10., binwidth=0.05,
                               pbc=[True, True, True], sigma=0.05, nsigma=4,
                               recalculate=False)

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
        comparator=comparator,
        with_force_threshold=False,
        constrained_qd=False,
        fmax_relaxation_convergence=0.2,
        force_threshold_fmax=False,
        compute_gradients=True,
    )


    cma = CMAMEGALOOP(
        number_of_bd_dimensions=2,
        crystal_system=crystal_system,
        crystal_evaluator=crystal_evaluator,
        step_size_gradient_optimizer_niu=experiment_parameters.cvt_run_parameters["cma_learning_rate"],
        initial_cmaes_step_size_sigma_g=experiment_parameters.cvt_run_parameters["cma_sigma_0"]
    )

    current_time_label = make_current_time_string(with_time=True)
    experiment_label = \
        f"{current_time_label}_{experiment_parameters.system_name}_{experiment_parameters.experiment_tag}"
    cma.compute(experiment_parameters.number_of_niches,
                experiment_parameters.maximum_evaluations,
                experiment_parameters.cvt_run_parameters,
                experiment_label=experiment_parameters.experiment_tag
                )
    pass


if __name__ == '__main__':
    experiment_parameters = ExperimentParameters(
        number_of_niches=20,
        maximum_evaluations=10,
        experiment_tag="test_n_generations",
        fitler_comparison_data_for_n_atoms=24,
        start_generator=StartGenerators.PYXTAL,
        cvt_run_parameters= \
            {
                # more of this -> higher-quality CVT
                "cvt_samples": 25000,
                # we evaluate in batches to paralleliez
                "batch_size": 1,
                # proportion of niches to be filled before starting
                "random_init": 0,
                # batch for random initialization
                "random_init_batch": 2,
                # when to write results (one generation = one batch)
                "dump_period":  2,
                # do we use several cores?
                "parallel": True,
                # do we cache the result of CVT and reuse?
                "cvt_use_cache": True,
                # min/max of parameters
                "bd_minimum_values": (0, 0),
                "bd_maximum_values": (100, 120),
                "relaxation_probability": 0,
                "behavioural_descriptors": [MaterialProperties.BAND_GAP, MaterialProperties.SHEAR_MODULUS],
                "number_of_relaxation_steps": 0,
                "curiosity_weights": True,
                "filter_starting_Structures": 24,
                "seed": False,
                "profiling": False,
                "force_threshold": True,
                "force_threshold_exp_fmax": 2.0,
                "constrained_qd": False,
                "relax_every_n_generations": 0,
                "alternative_operators": [("dqd", 10)],
                "relax_archive_every_n_generations": 2,
                "relax_archive_every_n_generations_n_relaxation_steps": 100,
                "fmax_threshold": 0.4,
                "dqd": True,
                "dqd_learning_rate": 0.0001,
                "cma_learning_rate": 1,
                "cma_sigma_0": 1,
            },
        system_name="TiO2",
        blocks = [22] * 8 + [8] * 16,
        volume=450,
        ratio_of_covalent_radii = 0.4,
        splits={(2,): 1, (4,): 1},
        cellbounds = CellBounds(
            bounds={'phi': [20, 160], 'chi': [20, 160], 'psi': [20, 160], 'a': [2, 40], 'b': [2, 40],
                    'c': [2, 40]}),
        operator_probabilities=[0., 0, 5., 5.],
        ### CVT PARAMETERS ###
        n_behavioural_descriptor_dimensions=2,
        fitness_min_max_values=[0, 10],
    )

    main_cma(experiment_parameters)
