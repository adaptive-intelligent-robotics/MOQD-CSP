import wandb

from chgnet.graph import CrystalGraphConverter
from functools import partial
from omegaconf import OmegaConf

from csp_elites.crystal.mo_crystal_evaluator import MOCrystalEvaluator
from csp_elites.crystal.crystal_system import CrystalSystem
from csp_elites.map_elites.elites_utils import (
    make_experiment_folder,
)

from csp_elites.mome.mome_utils import (
    mome_add_to_niche,
    mome_uniform_selection_fn,
    mome_metrics_fn,
)

from csp_elites.map_elites.map_elites_csp import MapElites


class MOME(MapElites):
    
    def __init__(
        self,
        crystal_system: CrystalSystem,
        crystal_evaluator: MOCrystalEvaluator,
        number_of_niches: int,
        number_of_bd_dimensions: int,
        run_parameters: dict,
        experiment_save_dir: str,
        centroids_load_dir: str="./experiments/centroids/"
    ):
        # Initialise Crystal functions
        self.crystal_system = crystal_system
        self.crystal_evaluator = crystal_evaluator
        self.graph_converter = CrystalGraphConverter()
        
        # Set up lodding
        self.experiment_save_dir = make_experiment_folder(experiment_save_dir)
        self.centroids_load_dir = make_experiment_folder(centroids_load_dir)

        # Store parameters
        self.n_relaxation_steps = run_parameters.number_of_relaxation_steps
        self.number_of_niches = number_of_niches
        self.run_parameters = run_parameters

        self.relax_every_n_generations = (
            run_parameters.relax_every_n_generations
            if "relax_every_n_generations" in run_parameters.keys()
            else 0
        )
        self.relax_archive_every_n_generations = (
            run_parameters.relax_archive_every_n_generations
            if "relax_archive_every_n_generations" in run_parameters.keys()
            else 0
        )
               
        # Initialise archives and counters  
        self.archive = {}  # init archive (empty)
        self.n_evals = 0  # number of evaluations since the beginning
        self.b_evals = 0  # number evaluation since the last dump
        self.configuration_counter = 0
        self.generation_counter = 0
        self.number_of_bd_dimensions = number_of_bd_dimensions

        # Set up where to save centroids:
        if self.run_parameters.cvt_use_cache:
            self.centroids_save_dir = self.centroids_load_dir
        else:
            self.centroids_save_dir = self.experiment_save_dir
        
        # Initialise centroids
        self.kdt = self._initialise_kdt_and_centroids(
            experiment_directory_path=self.centroids_load_dir,
            number_of_niches=number_of_niches,
            run_parameters=run_parameters,
        )
    
        # Set up mome-specific functions
        self.add_to_niche_function = partial(mome_add_to_niche,
            max_front_size=run_parameters.max_front_size
        )
        self.selection_operator = partial(mome_uniform_selection_fn,
            batch_size=run_parameters.system.batch_size
        )

        self.metrics_function = mome_metrics_fn

        # Setup logging
        wandb.init(
            # set the wandb project where this run will be logged
            project=f"MOQD-CSP",
            name=f"{run_parameters.algo.algo_name}",
            # track hyperparameters and run metadata
            config=OmegaConf.to_container(run_parameters, resolve=True),
        )
        self.metrics_history = None
