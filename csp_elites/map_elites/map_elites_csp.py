# | This file is based on the implementation map-elites implementation pymap_elites repo by resibots team https://github.com/resibots/pymap_elites
# | Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
# | Eloise Dalin , eloise.dalin@inria.fr
# | Pierre Desreumaux , pierre.desreumaux@inria.fr
# | **Main paper**: Mouret JB, Clune J. Illuminating search spaces by
# | mapping elites. arXiv preprint arXiv:1504.04909. 2015 Apr 20.

import gc
import os
import pickle

import numpy as np
import psutil
from ase import Atoms
from chgnet.graph import CrystalGraphConverter
from matplotlib import pyplot as plt
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
        centroids_load_dir: str="./experiments/centroids/"
    ):
        # Initialise Crystal functions
        self.crystal_system = crystal_system
        self.crystal_evaluator = crystal_evaluator
        self.graph_converter = CrystalGraphConverter()
        
        # Set up lodding
        self.experiment_save_dir = make_experiment_folder(experiment_save_dir)
        self.centroids_load_dir = make_experiment_folder(centroids_load_dir)
        self.log_file = open(f"{self.experiment_save_dir}/main_log.dat", "w")

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
        
        # Set up add to niche function
        self.add_to_niche_function = map_elites_add_to_niche
        self.selection_function = map_elites_selection_fn
        

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
                run_parameters.system.batch_size,
                selection_operator=self.selection_function,
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
                s.x["info"]["confid"] = self.configuration_counter
                self.configuration_counter += 1
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
            self.b_evals = 0
        # write log
        if self.log_file != None:
            fit_list = np.array([s.fitness for niche in self.archive.values() for s in niche])
            qd_score = np.sum(fit_list)
            coverage = 100 * len(fit_list) / self.number_of_niches

            self.log_file.write(
                "{} {} {} {} {} {} {} {} {}\n".format(
                    self.n_evals,
                    len(self.archive.keys()),
                    np.max(fit_list),
                    np.mean(fit_list),
                    np.median(fit_list),
                    np.percentile(fit_list, 5),
                    np.percentile(fit_list, 95),
                    coverage,
                    qd_score,
                )
            )
            self.log_file.flush()
            
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
            batch_size,
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