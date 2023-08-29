import json
import sys

from ase.ga.utilities import CellBounds

from csp_elites.crystal.materials_data_model import MaterialProperties, StartGenerators
from csp_elites.utils.experiment_parameters import ExperimentParameters
from main import main

if __name__ == '__main__':
    file_location = ""
    if file_location == "":
        file_location = sys.argv[1]
    with open(file_location, "r") as file:
        experiment_parameters = json.load(file)

    experiment_parameters = ExperimentParameters(**experiment_parameters)
    experiment_parameters.cellbounds = CellBounds(
            bounds={'phi': [20, 160], 'chi': [20, 160], 'psi': [20, 160], 'a': [2, 40], 'b': [2, 40],
                    'c': [2, 40]}),
    experiment_parameters.splits = {(2,): 1, (4,): 1}
    experiment_parameters.cvt_run_parameters["behavioural_descriptors"] = \
        [MaterialProperties(value) for value in experiment_parameters.cvt_run_parameters["behavioural_descriptors"]]

    experiment_parameters.start_generator = StartGenerators(experiment_parameters.start_generator)
    main(experiment_parameters, hide_prints=False)
