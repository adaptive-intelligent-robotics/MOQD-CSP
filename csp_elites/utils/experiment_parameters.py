import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Any, Optional

from ase.ga.utilities import CellBounds

from csp_elites.crystal.materials_data_model import StartGenerators, MaterialProperties


@dataclass
class ExperimentParameters:
    system_name: str
    blocks: List[int]
    volume: int
    ratio_of_covalent_radii: float
    splits: Dict[Tuple[int], int]
    cellbounds: CellBounds
    operator_probabilities: List[float]
    n_behavioural_descriptor_dimensions: int
    number_of_niches: int
    maximum_evaluations: int
    cvt_run_parameters: Dict[str, Any]
    experiment_tag: str
    fitness_min_max_values: []
    fitler_comparison_data_for_n_atoms: Optional[int]
    start_generator: StartGenerators

    @classmethod
    def generate_default_to_populate(cls):
        return cls(
            system_name="populate_here",
            blocks="populate_here",
            volume=450,
            ratio_of_covalent_radii=0.4,
            splits={(2,): 1, (4,): 1},
            cellbounds=CellBounds(
                bounds={'phi': [20, 160], 'chi': [20, 160], 'psi': [20, 160],
                        'a': [2, 40], 'b': [2, 40], 'c': [2, 40]
                        }
            ),
            operator_probabilities=[0, 0, 5, 5],
            n_behavioural_descriptor_dimensions=2,
            number_of_niches=200,
            maximum_evaluations="populate_here",
            cvt_run_parameters=cls.default_cvt_run_parameters(),
            experiment_tag="populate_here",
            fitness_min_max_values="populate_here",
            fitler_comparison_data_for_n_atoms=24,
            start_generator=StartGenerators.PYXTAL,
        )

    @staticmethod
    def default_cvt_run_parameters():
        return {
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
                # do we cache the result of CVT and reuse?
                "cvt_use_cache": True,
                # min/max of parameters
                "bd_minimum_values": (0, 0),
                "bd_maximum_values": (100, 120),
                "behavioural_descriptors": [MaterialProperties.BAND_GAP, MaterialProperties.SHEAR_MODULUS],
                "number_of_relaxation_steps": 0,
                "filter_starting_Structures": 24,
                "seed": False,
                "force_threshold": True,
                "force_threshold_exp_fmax": 2.0,
                "constrained_qd": False,
                "relax_every_n_generations": 0,
                "alternative_operators": [("dqd", 10)],
                "relax_archive_every_n_generations": 0,
                "relax_archive_every_n_generations_n_relaxation_steps": 0,
                "fmax_threshold": 0.4,
                "dqd": True,
                "dqd_learning_rate": 0.0001,
                "cma_learning_rate": 1,
                "cma_sigma_0": 1
            }

    def save_as_json(self, experiment_directory_path):
        self.splits = "DUMMY"
        self.cellbounds = "DUMMY"
        self.cvt_run_parameters["behavioural_descriptors"] = [descriptor.value for descriptor in self.cvt_run_parameters["behavioural_descriptors"]]
        self.start_generator = self.start_generator.value
        self.blocks = list(self.blocks)

        with open(f"{experiment_directory_path}/config.json", "w") as file:
            json.dump(asdict(self), file)
