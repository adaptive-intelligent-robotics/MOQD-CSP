import pathlib
import pickle

import numpy as np
from ase import Atoms
from ase.ga.utilities import CellBounds, closest_distances_generator
from chgnet.model import CHGNet
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

from csp_elites.crystal.crystal_system import CrystalSystem
from csp_elites.crystal.force_mutation import GradientMutation
from csp_elites.crystal.materials_data_model import MaterialProperties, StartGenerators
from csp_elites.utils.experiment_parameters import ExperimentParameters



def benchmark_force_mutation():
    number_of_individuals = 100

    experiment_parameters = ExperimentParameters(
        number_of_niches=20,
        maximum_evaluations=20,
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
                "constrained_qd": False,
                "relax_every_n_generations": 2,
                "alternative_operators": [("rattle", 5), ("gradient", 5)],
                "relax_archive_every_n_generations": 4
            },
        system_name="TiO2",
        blocks=[22] * 8 + [8] * 16,
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
    crystal_system = CrystalSystem(
        atom_numbers_to_optimise=experiment_parameters.blocks,
        volume=experiment_parameters.volume,
        ratio_of_covalent_radii=experiment_parameters.ratio_of_covalent_radii,
        splits=experiment_parameters.splits,
        operator_probabilities=experiment_parameters.operator_probabilities,
        start_generator=experiment_parameters.start_generator,
        alternative_operators=None,
    )

    # starting_individuals = crystal_system.create_n_individuals(number_of_individuals)
    #
    # with open(pathlib.Path(__file__).parent / "data" / f"TiO2_{number_of_individuals}_structures.pkl", "wb") as file:
    #     pickle.dump(starting_individuals, file)

    closest_distances = closest_distances_generator(atom_numbers=np.unique(np.array(experiment_parameters.blocks)),
                                                    ratio_of_covalent_radii=experiment_parameters.ratio_of_covalent_radii)

    with open(pathlib.Path(__file__).parent / "data" / f"TiO2_{number_of_individuals}_structures.pkl", "rb") as file:
        starting_atom_dictionaries = pickle.load(file)

    starting_atoms = [Atoms.fromdict(atoms) for atoms in starting_atom_dictionaries]

    model = CHGNet.load()
    number_of_steps = 50

    for learning_rate in tqdm([0.001, 0.01, 0.1]):
        all_data = []
        list_of_atoms = starting_atoms
        force_mutation = GradientMutation(
            blmin=closest_distances, n_top=len(experiment_parameters.blocks),
            learning_rate=learning_rate
        )

        for i in tqdm(range(number_of_steps)):
            atoms_after_1_step = []
            incorrect_individual_indices = []
            for i, atoms in enumerate(list_of_atoms):
                if atoms is None:
                    incorrect_individual_indices.append(i)
                else:
                    new_atoms, _ = force_mutation.get_new_individual([atoms])
                    if new_atoms is None:
                        incorrect_individual_indices.append(i)
                    else:
                        atoms_after_1_step.append(new_atoms)

            graphs = [model.graph_converter(AseAtomsAdaptor.get_structure(atoms), on_isolated_atoms="warn") for atoms in atoms_after_1_step]
            for index in incorrect_individual_indices:
                graphs.insert(index, None)
                atoms_after_1_step.insert(index, None)

            hotfix_graphs = False
            if None in graphs:
                hotfix_graphs = True
                indices_to_update = []
                new_graphs = []
                for i in range(len(graphs)):
                    if graphs[i] is None:
                        indices_to_update.append(i)
                    else:
                        new_graphs.append(graphs[i])

                graphs = new_graphs
            predictions = model.predict_graph(
                graphs,
                task="efs",
                return_atom_feas=False,
                return_crystal_feas=False,
                batch_size=10,
            )
            forces = np.array([pred["f"] for pred in predictions])
            energies = np.array([pred["e"] for pred in predictions]) * -1
            stresses = np.array([pred["s"] for pred in predictions])

            if hotfix_graphs:
                # todo: make this dynamic
                for i in indices_to_update:
                    forces = np.insert(forces, i, np.full((24, 3), 100), axis=0)
                    energies = np.insert(energies, i, 10000)
                    stresses = np.insert(stresses, i, np.full((3, 3), 100), axis=0)
            all_data.append([energies, forces, stresses, atoms_after_1_step])
            list_of_atoms = atoms_after_1_step

            with open(pathlib.Path(__file__).parent / "data" / f"TiO2_{number_of_individuals}_steps_{number_of_steps}_lr_{learning_rate}.pkl", "wb") as file:
                pickle.dump(all_data, file)



if __name__ == '__main__':
    # benchmark_force_mutation()

    #
    # learning_rates = [0.1, 0.001]
    # number_of_steps = 2
    # number_of_structures = 10

    # for i in range(len(learning_rates)):
    #     filepath = pathlib.Path(__file__).parent / "data" / f"TiO2_{number_of_structures}_steps_{number_of_steps}_lr_{learning_rates[i]}.pkl"
    #     with open(filepath, "rb") as file:
    #         all_data = pickle.load(file)
    #         print(len(all_data))
    # for learning_rate in tqdm([0.0001, 0.001, 0.01, 0.1]):


    learning_rates = [0.01]
    number_of_steps = 50
    number_of_structures = 100

    for i in range(len(learning_rates)):
        filepath = pathlib.Path(__file__).parent.parent.parent / ".experiment.nosync" / "benchmarking" / "force_mutation" / "data" / f"TiO2_{number_of_structures}_steps_{number_of_steps}_lr_{learning_rates[i]}.pkl"
        with open(filepath, "rb") as file:
            all_data = pickle.load(file)
            print(len(all_data))


    fake_1 = all_data[:50]
    fake_2 = all_data[50:100]
    fake_3 = all_data[100:]

    print()

    # print()
