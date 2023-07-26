import copy

import matgl
from ase.ga.soft_mutation import SoftMutation
from ase.ga.utilities import CellBounds, closest_distances_generator
from matgl.ext.ase import Relaxer

from csp_elites.crystal.crystal_system import CrystalSystem
from csp_elites.utils.experiment_parameters import ExperimentParameters

if __name__ == '__main__':
    experiment_parameters = ExperimentParameters(
        system_name="TiO2",
        blocks=[22] * 8 + [8] * 16,
        volume=453,
        ratio_of_covalent_radii=0.4,
        splits={(2,): 1, (4,): 1},
        cellbounds=CellBounds(
            bounds={'phi': [20, 160], 'chi': [20, 160], 'psi': [20, 160], 'a': [2, 40],
                    'b': [2, 40],
                    'c': [2, 40]}),
        operator_probabilities=[5., 2., 1., 2.],
        ### CVT PARAMETERS ###
        n_behavioural_descriptor_dimensions=2,
        number_of_niches=200,
        maximum_evaluations=100,
        cvt_run_parameters= \
            {
                # more of this -> higher-quality CVT
                "cvt_samples": 25000,
                # we evaluate in batches to paralleliez
                "batch_size": 1,
                # proportion of niches to be filled before starting
                "random_init": 0.001,
                # batch for random initialization
                "random_init_batch": 2,
                # when to write results (one generation = one batch)
                "dump_period": 1,
                # do we use several cores?
                "parallel": False,
                # do we cache the result of CVT and reuse?
                "cvt_use_cache": True,
                # min/max of parameters
                "bd_minimum_values": (-4, 0),
                "bd_maximum_values": (-1, 4),
            },
        experiment_tag=""
    )

    crystal_system = CrystalSystem(
        atom_numbers_to_optimise=experiment_parameters.blocks,
        volume=experiment_parameters.volume,
        ratio_of_covalent_radii=experiment_parameters.ratio_of_covalent_radii,
        splits=experiment_parameters.splits,
        operator_probabilities=experiment_parameters.operator_probabilities,
    )

    atoms = crystal_system.create_one_individual(0)
    atoms_relaxed = copy.deepcopy(atoms)

    relax_model = matgl.load_model("M3GNet-MP-2021.2.8-PES")
    relaxer = Relaxer(potential=relax_model)

    relaxer.relax(atoms_relaxed)
    closest_distances_soft_mutation = closest_distances_generator(experiment_parameters.blocks, 0.1)
    soft_mutation = SoftMutation(
            closest_distances_soft_mutation, bounds=[2., 5.], use_tags=False,
        )

    print()
