import os
import pathlib

import numpy as np
from ase import Atoms
from ase.ga.utilities import CellBounds

from csp_elites.crystal.materials_data_model import MaterialProperties
from csp_elites.map_elites.cvt_csp import CVT
from csp_elites.map_elites.elites_utils import add_to_archive, Species, save_archive
from csp_elites.utils.experiment_parameters import ExperimentParameters
from csp_elites.utils.plot import load_archive_from_pickle

if __name__ == '__main__':
    experiment_folder_name = "20230726_01_33_TiO2_local_115k"
    archive_number = 68000
    experiment_directory_path = pathlib.Path(__file__). resolve().parent.parent / "experiments" / experiment_folder_name
    archive_path = experiment_directory_path / f"archive_{archive_number}.pkl"
    folder_to_fix = experiment_directory_path / "to_fix"

    experiment_parameters = ExperimentParameters(
        number_of_niches=200,
        maximum_evaluations=100000,
        experiment_tag="timing_1k",
        fitler_comparison_data_for_n_atoms=24,
        cvt_run_parameters= \
            {
                # more of this -> higher-quality CVT
                "cvt_samples": 25000,
                # we evaluate in batches to paralleliez
                "batch_size": 10,
                # proportion of niches to be filled before starting
                "random_init": 0.05,
                # batch for random initialization
                "random_init_batch": 3,
                # when to write results (one generation = one batch)
                "dump_period": 1000,
                # do we use several cores?
                "parallel": True,
                # do we cache the result of CVT and reuse?
                "cvt_use_cache": True,
                # min/max of parameters
                "bd_minimum_values": (0, 0),
                "bd_maximum_values": (100, 100),
                "relaxation_probability": 0,
                "behavioural_descriptors": [MaterialProperties.BAND_GAP,
                                            MaterialProperties.SHEAR_MODULUS],
                "number_of_relaxation_steps": 0,
                "curiosity_weights": True,
                "filter_starting_Structures": 24,
            },
        system_name="TiO2",
        blocks=[22] * 8 + [8] * 16,
        volume=450,
        ratio_of_covalent_radii=0.4,
        splits={(2,): 1, (4,): 1},
        cellbounds=CellBounds(
            bounds={'phi': [20, 160], 'chi': [20, 160], 'psi': [20, 160], 'a': [2, 40],
                    'b': [2, 40],
                    'c': [2, 40]}),
        operator_probabilities=[5., 0, 2., 3.],
        ### CVT PARAMETERS ###
        n_behavioural_descriptor_dimensions=2,
        fitness_min_max_values=[0, 10],
    )

    fitnesses, centroids, descriptors, individuals = load_archive_from_pickle(
        filename=archive_path)
    dummy_cvt = CVT(2, None, None)
    refernce_archive = {}

    kdt, c = dummy_cvt._initialise_kdt_and_centroids(
        experiment_directory_path=experiment_directory_path,
        number_of_niches=200,
        run_parameters=experiment_parameters.cvt_run_parameters,
    )

    refernce_archive = dummy_cvt._convert_saved_archive_to_experiment_archive(
        experiment_directory_path=experiment_directory_path,
        experiment_label=None,
        archive_number=archive_number,
        kdt=kdt,
        archive=refernce_archive,
        individual_type="atoms"
    )

    list_of_files_in_directory = [name for name in os.listdir(f"{folder_to_fix}") if
                     not os.path.isdir(name)]
    for filename in list_of_files_in_directory:
        if "archive_" in filename:
            archive_id = filename.lstrip("archive_").rstrip(".pkl")
            archive_to_fix = {}
            archive_to_fix = dummy_cvt._convert_saved_archive_to_experiment_archive(
                experiment_directory_path=folder_to_fix,
                experiment_label=None,
                archive_number=archive_id,
                kdt=kdt,
                archive=archive_to_fix,
                individual_type="dict"
            )

            individual_matches_counter_post = 0
            for key, value in refernce_archive.items():
                individual_as_atoms = Atoms.fromdict(archive_to_fix[key].x)
                reference_individual_as_atoms = Atoms.fromdict(value.x)
                # print(reference_individual_as_atoms == individual_as_atoms)
                if reference_individual_as_atoms == individual_as_atoms:
                    archive_to_fix[key].desc = value.desc
                    individual_matches_counter_post += 1

            # descriptor_match_counter_post = 0
            # for key, value in refernce_archive.items():
            #
            #     if np.isclose(np.array(archive_to_fix[key].desc), np.array(value.desc)).all():
            #         descriptor_match_counter_post += 1
            #

            save_archive(archive_to_fix, archive_id, experiment_directory_path)

    print()
