from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional

from ase.ga.utilities import CellBounds


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
