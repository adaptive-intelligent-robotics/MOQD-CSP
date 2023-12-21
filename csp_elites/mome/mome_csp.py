import gc
import os
import pandas as pd
import wandb

import numpy as np
from chgnet.graph import CrystalGraphConverter
from functools import partial
from omegaconf import OmegaConf
from tqdm import tqdm

from csp_elites.crystal.mo_crystal_evaluator import MOCrystalEvaluator
from csp_elites.crystal.crystal_system import CrystalSystem
from csp_elites.map_elites.elites_utils import (
    save_archive,
    add_to_archive,
    make_experiment_folder,
)
from csp_elites.mome.mome_utils import (
    mome_add_to_niche,
    mome_uniform_selection_fn,
    mome_crowding_add_to_niche,
    mome_crowding_selection_fn,
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
        selection: str,
        addition: str,
        centroids_load_dir: str="./reference_data/centroids/"
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
        if selection == "uniform":
            self.selection_operator = partial(mome_uniform_selection_fn,
                batch_size=run_parameters.batch_size
            )
        elif selection == "biased":
            self.selection_operator = partial(mome_crowding_selection_fn,
                batch_size=run_parameters.batch_size
            )
        
        if addition == "uniform":
            self.add_to_niche_function = partial(mome_add_to_niche,
                max_front_size=run_parameters.max_front_size
            )
        elif addition == "biased":
            self.add_to_niche_function = partial(mome_crowding_add_to_niche,
                max_front_size=run_parameters.max_front_size
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

    def run(
        self,
        number_of_niches,
        maximum_evaluations,
        run_parameters,
    ):

        pbar = tqdm(desc="Number of evaluations", total=maximum_evaluations, position=2)
        while self.n_evals < maximum_evaluations:  ### NUMBER OF GENERATIONS
            self.generation_counter += 1
            
            population = self.get_population(
                run_parameters=run_parameters,
                number_of_niches=number_of_niches,    
            )
            
            n_relaxation_steps = self.set_number_of_relaxation_steps()

            (
                population_as_atoms,
                population,
                fitness_scores,
                descriptors,
                kill_list,
                gradients,
            ) = self.crystal_evaluator.batch_compute_fitness_and_bd(
                list_of_atoms=population,
                n_relaxation_steps=n_relaxation_steps
            )

            if population is not None:
                self.crystal_system.update_operator_scaling_volumes(
                    population=population_as_atoms
                )
                del population_as_atoms

            self.update_archive(
                population, fitness_scores, descriptors, kill_list, gradients
            )
            pbar.update(len(population))
            del population
            del fitness_scores
            del descriptors
            del kill_list
        
        save_archive(self.archive, self.n_evals, self.experiment_save_dir)
        
        # Save final metrics
        metrics_history_df = pd.DataFrame.from_dict(self.metrics_history,orient='index').transpose()
        metrics_history_df.to_csv(os.path.join(self.experiment_save_dir, "metrics_history.csv"), index=False)

        return self.archive
    
    def update_archive(
        self, population, fitness_scores, descriptors, kill_list, gradients
    ):
        s_list = self.crystal_evaluator.batch_create_species(
            population, fitness_scores, descriptors, kill_list, gradients
        )
        self.n_evals += self.run_parameters.batch_size
        self.b_evals += self.run_parameters.batch_size
        for s in s_list:
            if s is None:
                continue
            else:
                self.archive = add_to_archive(s, s.desc, self.archive, self.kdt, self.add_to_niche_function)

        if (
            self.b_evals >= self.run_parameters.dump_period
            and self.run_parameters.dump_period != -1
        ):
            print(
                "[{}/{}]".format(self.n_evals, int(self.run_parameters.maximum_evaluations)),
                end=" ",
                flush=True,
            )
            save_archive(self.archive, self.n_evals, self.experiment_save_dir)
            
            metrics_history_df = pd.DataFrame.from_dict(self.metrics_history,orient='index').transpose()
            metrics_history_df.to_csv(os.path.join(self.experiment_save_dir, "metrics_history.csv"), index=False)

            self.b_evals = 0
            
            
        # Calculate metrics and log
        metrics = self.metrics_function(
                self.archive,
                self.run_parameters,
                self.n_evals,
            )
        
        wandb.log(metrics)

        if self.metrics_history == None:
            self.metrics_history = {key: np.array(metrics[key]) for key in metrics}
            # for k, v in self.metrics_history.items():
            #     self.metrics_history[k] = np.expand_dims(v, axis=0)

        else:
            self.metrics_history = {key: np.append(self.metrics_history[key], metrics[key]) for key in metrics}
            
        gc.collect()