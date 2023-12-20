from csp_elites.crystal.crystal_system import CrystalSystem
from csp_elites.crystal.materials_data_model import MaterialProperties, StartGenerators
from csp_elites.crystal.mo_crystal_evaluator import MOCrystalEvaluator
from csp_elites.map_elites.map_elites_csp import MapElites
from csp_elites.mome.mome_csp import MOME
from csp_elites.map_elites.elites_utils import __centroids_filename
from analysis_functions.experiment_processing import ExperimentProcessor

from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
import time
from typing import Tuple

import torch
import random
import numpy as np

@dataclass
class ExperimentConfig:
    
    number_of_niches: int
    maximum_evaluations: int

    ### CVT params
    cvt_samples: int
    batch_size: int
    random_init: int
    random_init_batch: int
    dump_period: int
    parallel: bool
    cvt_use_cache: bool
    bd_minimum_values: Tuple[float, ...]
    bd_maximum_values: Tuple[float, ...]
    relaxation_probability: float
    behavioural_descriptors: Tuple[str, ...]
    number_of_relaxation_steps: int
    filter_starting_Structures: int


    ## Algorithm params
    seed: bool
    profiling: bool
    force_threshold: bool
    force_threshold_exp_fmax: float
    fmax_threshold: float
    normalise_bd: bool
    
    alternative_operators: Tuple[float,...]
    compute_gradients: bool # use DQD or not
    from_archive_path: bool


@hydra.main(config_path="configs/", config_name="csp")
def main(config:ExperimentConfig) -> None:
    
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    
    torch.set_num_threads(config.number_of_threads)
        
    experiment_save_dir = f"output/{config.system.system_name}/{config.algo.algo_name}/{config.experiment_tag}"

    splits = {(2,): 1, (4,): 1}
    
    config.behavioural_descriptors = [
        MaterialProperties(value) for value in config.behavioural_descriptors
    ]

    start_generator = StartGenerators(
        config.system.start_generator
    )

    crystal_system = CrystalSystem(
            atom_numbers_to_optimise=config.system.blocks,
            volume=config.system.volume,
            ratio_of_covalent_radii=config.system.ratio_of_covalent_radii,
            splits=splits,
            compound_formula=config.system.system_name,
            operator_probabilities=config.system.operator_probabilities,
            start_generator=start_generator,
            alternative_operators=config.alternative_operators,
            learning_rate=config.dqd_learning_rate,
    )
    
    crystal_evaluator = MOCrystalEvaluator(
        with_force_threshold=config.force_threshold,
        fmax_relaxation_convergence=config.fmax_threshold,
        force_threshold_fmax=config.force_threshold_exp_fmax,
        compute_gradients=config.compute_gradients,
        bd_normalisation=(
            config.system.bd_minimum_values,
            config.system.bd_maximum_values,
        )
        if config.normalise_bd
        else None,
    )
        
    if config.algo.algo_name == "mome" or config.algo.algo_name == "biased_mome":
        main = MOME(
            crystal_system=crystal_system,
            crystal_evaluator=crystal_evaluator,
            number_of_niches=config.number_of_niches,
            number_of_bd_dimensions=config.system.n_behavioural_descriptor_dimensions,
            run_parameters=config,
            experiment_save_dir=experiment_save_dir,
            selection=config.algo.selection,
        )
    
    elif config.algo.algo_name == "map_elites_energy" or config.algo.algo_name == "map_elites_magmom":
        main = MapElites(
            crystal_system=crystal_system,
            crystal_evaluator=crystal_evaluator,
            number_of_niches=config.number_of_niches,
            number_of_bd_dimensions=config.system.n_behavioural_descriptor_dimensions,
            run_parameters=config,
            experiment_save_dir=experiment_save_dir,
            objective=config.algo.objective,
        )
        
    tic = time.time()

    archive = main.run(
        number_of_niches=config.number_of_niches,
        maximum_evaluations=config.maximum_evaluations,
        run_parameters=config,
    )
    
    print(f"time taken {time.time() - tic}")

    # Variables setting

    if config.normalise_bd:
        bd_minimum_values, bd_maximum_values = [0, 0], [1, 1]
    else:
        bd_minimum_values, bd_maximum_values = (
            config.system.bd_minimum_values,
            config.system.bd_maximum_values,
        )

    centroid_filename = __centroids_filename(
        k=config.number_of_niches,
        dim=config.system.n_behavioural_descriptor_dimensions,
        bd_names=config.behavioural_descriptors,
        bd_minimum_values=bd_minimum_values,
        bd_maximum_values=bd_maximum_values,
        formula=config.system.system_name,
    )

    experiment_processor = ExperimentProcessor(
        config=config,
        save_structure_images=False,
        filter_for_experimental_structures=False,
        centroid_filename=centroid_filename,
        centroids_load_dir=main.centroids_load_dir,
        experiment_save_dir=main.experiment_save_dir,
    )
    
    experiment_processor.plot_mome()
    experiment_processor.plot_mome(annotate=False)
    experiment_processor.process_symmetry()



if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="main", node=ExperimentConfig)
    main()

   