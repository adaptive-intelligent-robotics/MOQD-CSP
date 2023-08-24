import pathlib

from ase.ga.utilities import CellBounds
from tqdm import tqdm

from csp_elites.crystal.materials_data_model import MaterialProperties, StartGenerators
from csp_elites.utils.archive_evolution_gif import plot_gif
from csp_elites.utils.asign_target_values_to_centroids import \
    compute_centroids_for_target_solutions, reassign_data_from_pkl_to_new_centroids
from csp_elites.utils.get_mpi_structures import get_all_materials_with_formula
from csp_elites.utils.plot import plot_all_maps_in_archive, load_centroids, \
    plot_all_statistics_from_file, load_archive_from_pickle
from csp_experiments.run_experiment import ExperimentParameters

if __name__ == '__main__':

    experiment_labels = [
        "20230731_18_08_TiO2_200_niches_1M_force_threshold",
    ]
    centroid_list = [
        "centroids_200_2_band_gap_0_100_shear_modulus_0_100.dat",
    ]

    for i, experiment_label in tqdm(enumerate(experiment_labels)):
        print(experiment_label)
        experiment_directory = pathlib.Path(
            __file__).parent.parent / ".experiment.nosync" / "experiments" / experiment_label

        centroid_filename = pathlib.Path(__file__).parent.parent / ".experiment.nosync" / "experiments" / "centroids" / centroid_list[i]

        all_centroids = load_centroids(centroid_filename)

        experiment_parameters = ExperimentParameters(
            number_of_niches=200,
            maximum_evaluations=4,
            experiment_tag="test_plotting",
            fitler_comparison_data_for_n_atoms=0,
            start_generator=StartGenerators.PYXTAL,
            cvt_run_parameters= \
                {
                    # more of this -> higher-quality CVT
                    "cvt_samples": 25000,
                    # we evaluate in batches to paralleliez
                    "batch_size": 100,
                    # proportion of niches to be filled before starting
                    "random_init": 0.1,
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
                    "bd_maximum_values": (100, 100),
                    "relaxation_probability": 0,
                    "behavioural_descriptors": [MaterialProperties.BAND_GAP, MaterialProperties.SHEAR_MODULUS],
                    "number_of_relaxation_steps": 10,
                },
            system_name="TiO2",
            blocks= [22] * 8 + [8] * 16,
            volume=450,
            ratio_of_covalent_radii = 0.4,
            splits={(2,): 1, (4,): 1},
            cellbounds = CellBounds(
                bounds={'phi': [20, 160], 'chi': [20, 160], 'psi': [20, 160], 'a': [2, 40], 'b': [2, 40],
                        'c': [2, 40]}),
            operator_probabilities = [5., 0, 3., 2.],
            ### CVT PARAMETERS ###
            n_behavioural_descriptor_dimensions=2,
            fitness_min_max_values=[6.5, 10],
        )

        if MaterialProperties.ENERGY_FORMATION in experiment_parameters.cvt_run_parameters["behavioural_descriptors"]:
            comparison_data = str(pathlib.Path(
                experiment_directory).parent / "experiments/target_data/tio2_target_data.pkl")
            target_centroids = compute_centroids_for_target_solutions(
                centroids_file=centroid_filename,
                target_data_file=comparison_data,
                filter_for_number_of_atoms=experiment_parameters.fitler_comparison_data_for_n_atoms
            )
        else:
            comparison_data = str(pathlib.Path(
                experiment_directory).parent / "target_data/ti02_band_gap_shear_modulus.pkl")
            comparison_data_packed = load_archive_from_pickle(comparison_data)
            target_centroids = reassign_data_from_pkl_to_new_centroids(
                centroids_file=centroid_filename,
                target_data=comparison_data_packed,
                filter_for_number_of_atoms=experiment_parameters.fitler_comparison_data_for_n_atoms
            )


        plot_all_maps_in_archive(
            experiment_directory_path=experiment_directory, experiment_parameters=experiment_parameters,
            all_centroids=all_centroids,
            target_centroids=target_centroids,
        )

        plot_gif(experiment_directory_path=experiment_directory)

        plot_all_statistics_from_file(
            filename=f"{experiment_directory}/{experiment_label}.dat",
            save_location=f"{experiment_directory}/",
        )

        structure_info, known_structures = get_all_materials_with_formula(
            experiment_parameters.system_name)
