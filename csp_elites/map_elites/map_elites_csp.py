# | This file is based on the implementation map-elites implementation pymap_elites repo by resibots team https://github.com/resibots/pymap_elites
# | Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
# | Eloise Dalin , eloise.dalin@inria.fr
# | Pierre Desreumaux , pierre.desreumaux@inria.fr
# | **Main paper**: Mouret JB, Clune J. Illuminating search spaces by
# | mapping elites. arXiv preprint arXiv:1504.04909. 2015 Apr 20.

import gc
import os
import pandas as pd
import pickle
import wandb

import numpy as np
import psutil
from ase import Atoms
from chgnet.graph import CrystalGraphConverter
from functools import partial
from omegaconf import OmegaConf
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.neighbors import KDTree
from tqdm import tqdm

from csp_elites.crystal.crystal_evaluator import CrystalEvaluator
from csp_elites.crystal.crystal_system import CrystalSystem
from csp_elites.map_elites.elites_utils import (
    cvt,
    save_archive,
    add_to_archive,
    make_experiment_folder,
    map_elites_add_to_niche,
    map_elites_metrics_fn,
    map_elites_selection_fn,
    write_centroids,
    Species,
)
from csp_elites.utils.get_mpi_structures import get_all_materials_with_formula


class MapElites:
    def __init__(
        self,
        crystal_system: CrystalSystem,
        crystal_evaluator: CrystalEvaluator,
        number_of_niches: int,
        number_of_bd_dimensions: int,
        run_parameters: dict,
        experiment_save_dir: str,
        centroids_load_dir: str="./reference_data/centroids/"
    ):
        # Initialise Crystal functions
        self.crystal_system = crystal_system
        self.crystal_evaluator = crystal_evaluator
        self.graph_converter = CrystalGraphConverter()
        
        # Set up directories
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
        
        # Set up map-elites specific functions
        self.add_to_niche_function = map_elites_add_to_niche
        self.selection_operator = partial(map_elites_selection_fn,
            batch_size=run_parameters.batch_size
        )
        self.metrics_function = map_elites_metrics_fn
        
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
        
        
        # Save final archive
        save_archive(self.archive, self.n_evals, self.experiment_save_dir)
        
        # Save final metrics
        metrics_history_df = pd.DataFrame.from_dict(self.metrics_history,orient='index').transpose()
        metrics_history_df.to_csv(os.path.join(self.experiment_save_dir, "metrics_history.csv"), index=False)

        return self.archive

    def get_population(
        self,
        run_parameters,
        number_of_niches,
    ):
        # random initialization
        population = []
        
        # Make sure we have enough niches filled to start with
        if len(self.archive) <= run_parameters.random_init * number_of_niches:
            individuals = self.crystal_system.create_n_individuals(
                run_parameters.random_init_batch
            )
            if run_parameters.seed:
                individuals = self.initialise_known_atoms()
            population += individuals

            with open(
                f"{self.experiment_save_dir}/starting_population.pkl", "wb"
            ) as file:
                pickle.dump(population, file)

        # Get individuals for relaxation
        elif (
            (self.relax_archive_every_n_generations != 0)
            and (
                self.generation_counter % self.relax_archive_every_n_generations
                == 0
            )
            and (self.generation_counter != 0)
        ):
            population = [species.x for niche in self.archive.values() for species in niche]
            
        # Otherwise select indviduals from archive and mutate them
        else:  # variation/selection loop
            mutated_individuals = self.mutate_individuals(
                run_parameters.batch_size,
                selection_operator=self.selection_operator,
            )
            population += mutated_individuals

        return population
    
    
    def initialise_known_atoms(self):
        _, known_atoms = get_all_materials_with_formula(
            self.crystal_system.compound_formula
        )
        individuals = []
        for atoms in known_atoms:
            if (
                len(atoms.get_atomic_numbers())
                == self.run_parameters.filter_starting_Structures
            ):
                atoms.rattle()
                atoms.info = None
                atoms = atoms.todict()
                individuals.append(atoms)
        del known_atoms
        return individuals

    def update_archive(
        self, population, fitness_scores, descriptors, kill_list, gradients
    ):
        s_list = self.crystal_evaluator.batch_create_species(
            population, fitness_scores, descriptors, kill_list, gradients
        )
        evaluations_performed = len(population)
        self.n_evals += evaluations_performed
        self.b_evals += evaluations_performed
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

    def _initialise_kdt_and_centroids(
        self, experiment_directory_path, number_of_niches, run_parameters
    ):
        # create the CVT
        if run_parameters.normalise_bd:
            bd_minimum_values, bd_maximum_values = [0, 0], [1, 1]
        else:
            bd_minimum_values, bd_maximum_values = (
                run_parameters.system.bd_minimum_values,
                run_parameters.system.bd_maximum_values,
            )

        c = cvt(
            number_of_niches,
            self.number_of_bd_dimensions,
            run_parameters.cvt_samples,
            bd_minimum_values,
            bd_maximum_values,
            experiment_directory_path,
            run_parameters.behavioural_descriptors,
            run_parameters.cvt_use_cache,
            formula=self.crystal_system.compound_formula,
            centroids_load_dir=self.centroids_load_dir,
            centroids_save_dir=self.centroids_save_dir,
        )
        kdt = KDTree(c, leaf_size=30, metric="euclidean")
        write_centroids(
            c,
            experiment_folder=self.centroids_save_dir,
            bd_names=run_parameters.behavioural_descriptors,
            bd_minimum_values=run_parameters.system.bd_minimum_values,
            bd_maximum_values=run_parameters.system.bd_maximum_values,
            formula=self.crystal_system.compound_formula,
        )
        del c
        return kdt

    def mutate_individuals(
        self,
        batch_size,
        selection_operator,
    ):

        parents_x, parents_y = self.selection_operator(
            self.archive,
        )
        
        mutated_offsprings = []
        for n in range(0, batch_size):
            # copy & add variation
            z = self.crystal_system.mutate([parents_x[n], parents_y[n]])
            if z is None or (
                self.graph_converter(
                    AseAtomsAdaptor.get_structure(z), on_isolated_atoms="warn"
                )
                is None
            ):
                continue
            mutated_offsprings += [Atoms.todict(z)]
        return mutated_offsprings

    def set_number_of_relaxation_steps(self):
        if self.relax_every_n_generations != 0 and (
            self.relax_archive_every_n_generations == 0
        ):
            if self.generation_counter // self.relax_every_n_generations == 0:
                n_relaxation_steps = 100
            else:
                n_relaxation_steps = self.run_parameters.number_of_relaxation_steps
        elif (self.relax_archive_every_n_generations != 0) and (
            self.generation_counter % self.relax_archive_every_n_generations == 0
        ):
            n_relaxation_steps = (
                self.run_parameters[
                    "relax_archive_every_n_generations_n_relaxation_steps"
                ]
                if "relax_archive_every_n_generations_n_relaxation_steps"
                in self.run_parameters.keys()
                else 10
            )

        else:
            n_relaxation_steps = self.run_parameters.number_of_relaxation_steps

        return n_relaxation_steps