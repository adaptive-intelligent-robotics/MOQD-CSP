import json
import pathlib
import sys

from ase.ga.utilities import CellBounds

from csp_elites.cli.main import main
from csp_elites.crystal.materials_data_model import MaterialProperties, StartGenerators
from csp_elites.utils.experiment_parameters import ExperimentParameters


if __name__ == "__main__":
    file_location = ""
    file_location = pathlib.Path(__file__).parent / "experiments" / "config.json"
    path_to_experiment = (
        pathlib.Path(__file__).parent
        / "experiments"
        / "20230828_11_29_TiO2_test_n_generations"
    )

    if file_location == "":
        file_location = sys.argv[1]
    with open(file_location, "r") as file:
        experiment_parameters = json.load(file)

    experiment_parameters = ExperimentParameters(**experiment_parameters)
    experiment_parameters.cellbounds = (
        CellBounds(
            bounds={
                "phi": [20, 160],
                "chi": [20, 160],
                "psi": [20, 160],
                "a": [2, 40],
                "b": [2, 40],
                "c": [2, 40],
            }
        ),
    )
    experiment_parameters.splits = {(2,): 1, (4,): 1}
    experiment_parameters.cvt_run_parameters["behavioural_descriptors"] = [
        MaterialProperties(value)
        for value in experiment_parameters.cvt_run_parameters["behavioural_descriptors"]
    ]

    experiment_parameters.start_generator = StartGenerators(
        experiment_parameters.start_generator
    )
    experiment_parameters.system_name = "TiO2"
    experiment_parameters.blocks = [22] * 8 + [8] * 16
    experiment_parameters.cvt_run_parameters["normalise_bd"] = True
    experiment_parameters.maximum_evaluations = 10
    experiment_parameters.cvt_run_parameters["alternative_operators"] = [("rattle", 10)]
    experiment_parameters.cvt_run_parameters["dump_period"] = 1
    main(experiment_parameters, hide_prints=False, from_archive_path=path_to_experiment)
