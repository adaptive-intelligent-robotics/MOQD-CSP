#| This file is a part of the pymap_elites framework.
#| Copyright 2019, INRIA
#| Main contributor(s):
#| Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
#| Eloise Dalin , eloise.dalin@inria.fr
#| Pierre Desreumaux , pierre.desreumaux@inria.fr
#|
#|
#| **Main paper**: Mouret JB, Clune J. Illuminating search spaces by
#| mapping elites. arXiv preprint arXiv:1504.04909. 2015 Apr 20.
#|
#| This software is governed by the CeCILL license under French law
#| and abiding by the rules of distribution of free software.  You
#| can use, modify and/ or redistribute the software under the terms
#| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
#| following URL "http://www.cecill.info".
#|
#| As a counterpart to the access to the source code and rights to
#| copy, modify and redistribute granted by the license, users are
#| provided only with a limited warranty and the software's author,
#| the holder of the economic rights, and the successive licensors
#| have only limited liability.
#|
#| In this respect, the user's attention is drawn to the risks
#| associated with loading, using, modifying and/or developing or
#| reproducing the software by the user in light of its specific
#| status of free software, that may mean that it is complicated to
#| manipulate, and that also therefore means that it is reserved for
#| developers and experienced professionals having in-depth computer
#| knowledge. Users are therefore encouraged to load and test the
#| software's suitability as regards their requirements in conditions
#| enabling the security of their systems and/or data to be ensured
#| and, more generally, to use and operate it in the same conditions
#| as regards security.
#|
#| The fact that you are presently reading this means that you have
#| had knowledge of the CeCILL license and that you accept its terms.
# import multiprocessing
import gc
import pickle
from typing import List, Dict, Union

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
from csp_elites.map_elites.elites_utils import cvt, save_archive, add_to_archive, \
    write_centroids, make_experiment_folder, Species, evaluate_parallel
from csp_elites.utils.get_mpi_structures import get_all_materials_with_formula
from csp_elites.utils.plot import load_archive_from_pickle


class CVT:
    def __init__(self,
        number_of_bd_dimensions: int,
        crystal_system: CrystalSystem,
        crystal_evaluator: CrystalEvaluator,

    ):
        self.number_of_bd_dimensions = number_of_bd_dimensions
        self.crystal_system = crystal_system
        self.crystal_evaluator = crystal_evaluator
        self.graph_converter = CrystalGraphConverter()
    def batch_compute_with_list_of_atoms(self,
        number_of_niches,
        maximum_evaluations,
        run_parameters,
        experiment_label,
        ):
        experiment_directory_path = make_experiment_folder(experiment_label)
        log_file = open(f'{experiment_directory_path}/{experiment_label}.dat', 'w')
        memory_log = open(f'{experiment_directory_path}/memory_log.dat', 'w')
        with open(f'{experiment_directory_path}/experiment_parameters.pkl', 'wb') as file:
            pickle.dump(run_parameters, file)

        # create the CVT
        kdt = self._initialise_kdt_and_centroids(
            experiment_directory_path=experiment_directory_path,
            number_of_niches=number_of_niches,
            run_parameters=run_parameters,
        )

        archive = {}  # init archive (empty)
        n_evals = 0  # number of evaluations since the beginning
        b_evals = 0  # number evaluation since the last dump

        # main loop
        configuration_counter = 0

        relax_every_n_generations = run_parameters["relax_every_n_generations"] if "relax_every_n_generations" in run_parameters.keys() else 0
        relax_archive_every_n_generations = run_parameters[
            "relax_archive_every_n_generations"] if "relax_archive_every_n_generations" in run_parameters.keys() else 0

        generation_counter = 0
        rambar = tqdm(total=100, desc='ram%', position=0)
        rambar.n = psutil.virtual_memory().percent
        rambar.refresh()
        print(f"Starting RAM usage Absolute {psutil.virtual_memory()[3]/1000000000}")

        ram_logging = []
        pbar = tqdm(desc="Number of evaluations", total=maximum_evaluations, position=2)
        while (n_evals < maximum_evaluations):  ### NUMBER OF GENERATIONS
            ram_logging.append(psutil.virtual_memory()[3]/1000000000)
            generation_counter += 1
            # random initialization
            population = []
            if len(archive) <= run_parameters['random_init'] * number_of_niches:
                individuals = self.crystal_system.create_n_individuals(
                    run_parameters['random_init_batch'])

                if run_parameters["seed"]:
                    _, known_atoms = get_all_materials_with_formula("TiO2")
                    for atoms in known_atoms:
                        if len(atoms.get_atomic_numbers()) == run_parameters["filter_starting_Structures"]:
                            atoms.rattle()
                            atoms.info = None
                            atoms = atoms.todict()
                            individuals.append(atoms)
                    del known_atoms
                population += individuals
                with open(f'{experiment_directory_path}/starting_population.pkl', 'wb') as file:
                    pickle.dump(population, file)

            elif (relax_archive_every_n_generations != 0) and (generation_counter % relax_archive_every_n_generations == 0) and (generation_counter != 0):
                population = [species.x for species in list(archive.values())]

            else:  # variation/selection loop
                keys = list(archive.keys())
                rand1 = np.random.randint(len(keys), size=run_parameters['batch_size'])
                rand2 = np.random.randint(len(keys), size=run_parameters['batch_size'])

                # print(rand1)
                # print(rand2)
                for n in range(0, run_parameters['batch_size']):
                    # parent selection
                    x = archive[keys[rand1[n]]]
                    y = archive[keys[rand2[n]]]
                    # copy & add variation
                    z = self.crystal_system.mutate([x, y])
                    # z, _ = self.crystal_system.operators.get_new_individual(
                    #     [Atoms.fromdict(x.x), Atoms.fromdict(y.x)])
                    if z is None:
                        print(" z is none bug")

                    elif self.graph_converter(AseAtomsAdaptor.get_structure(z), on_isolated_atoms="warn") is None:
                        print("graph is None")
                    else:
                        z = z.todict()
                        population += [z]

            # Check population for isolated atoms
            # population = [individual for individual in population if self.graph_converter( on_isolated_atoms="warn") is not None]

            if relax_every_n_generations != 0 and (relax_archive_every_n_generations == 0):
                if generation_counter // relax_every_n_generations == 0:
                    n_relaxation_steps = 100
                else:
                    n_relaxation_steps = run_parameters["number_of_relaxation_steps"]
            elif (relax_archive_every_n_generations != 0) and (generation_counter % relax_archive_every_n_generations == 0):
                n_relaxation_steps = run_parameters["relax_archive_every_n_generations_n_relaxation_steps"] if "relax_archive_every_n_generations_n_relaxation_steps" in run_parameters.keys() else 10

            else:
                n_relaxation_steps = run_parameters["number_of_relaxation_steps"]

            population_as_atoms, population, fitness_scores, descriptors, kill_list, gradients = self.crystal_evaluator.batch_compute_fitness_and_bd(
                list_of_atoms=population,
                cellbounds=self.crystal_system.cellbounds,
                really_relax=None,
                behavioral_descriptor_names=run_parameters["behavioural_descriptors"],
                n_relaxation_steps=n_relaxation_steps
            )

            if population is not None:
                self.crystal_system.update_operator_scaling_volumes(population=population_as_atoms)
                del population_as_atoms
            # todo: make sure population ok after relaxation
            s_list = self.crystal_evaluator.batch_create_species(population, fitness_scores, descriptors, kill_list, gradients)
            # count evals
            evaluations_performed = len(population)
            n_evals += evaluations_performed
            b_evals += evaluations_performed

            del population
            del fitness_scores
            del descriptors
            del kill_list
            # s_list = evaluate_parallel(to_evaluate)

            # natural selection

            for s in s_list:
                if s is None:
                    continue
                else:
                    s.x["info"]["confid"] = configuration_counter
                    configuration_counter += 1
                    add_to_archive(s, s.desc, archive, kdt)

            # write archive
            if b_evals >= run_parameters['dump_period'] and run_parameters['dump_period'] != -1:
                print("[{}/{}]".format(n_evals, int(maximum_evaluations)), end=" ", flush=True)
                save_archive(archive, n_evals, experiment_directory_path)
                b_evals = 0
            # write log
            if log_file != None:
                fit_list = np.array([x.fitness for x in archive.values()])
                qd_score = np.sum(fit_list)
                coverage = 100 * len(fit_list) / number_of_niches

                log_file.write("{} {} {} {} {} {} {} {} {}\n".format(n_evals,
                                                                     len(archive.keys()),
                                                                     np.max(fit_list),
                                                                     np.mean(fit_list),
                                                                     np.median(fit_list),
                                                                     np.percentile(fit_list, 5),
                                                                     np.percentile(fit_list, 95),
                                                                     coverage, qd_score))
                log_file.flush()
            pbar.update(evaluations_performed)
            rambar.n = psutil.virtual_memory().percent
            rambar.refresh()
            memory = psutil.virtual_memory()[3] / 1000000000
            memory_log.write("{} {}\n".format(n_evals, memory))
            memory_log.flush()
            gc.collect()

        save_archive(archive, n_evals, experiment_directory_path)
        plt.plot(range(len(ram_logging)), ram_logging)
        plt.xlabel("Number of Times Evaluation Loop Was Ran")
        plt.ylabel("Amount of RAM Used")
        plt.title("RAM over time")
        plt.savefig(f"{experiment_directory_path}/memory_over_time.png", format="png")

        print(f"End RAM usage {psutil.virtual_memory()[3] / 1000000000}")


        return experiment_directory_path, archive


    def get_all_individuals_from_archive(self, archive):
        pass

    def compute(self,
                number_of_niches,
                maximum_evaluations,
                run_parameters,
                experiment_label,
                ):
        """CVT MAP-Elites
           Vassiliades V, Chatzilygeroudis K, Mouret JB. Using centroidal voronoi
           tessellations to scale up the multidimensional archive of phenotypic elites algorithm.
           IEEE Transactions on Evolutionary Computation. 2017 Aug 3;22(4):623-30.

           Format of the logfile: evals archive_size max mean median 5%_percentile, 95%_percentile

        """
        # setup the parallel processing pool

        experiment_directory_path = make_experiment_folder(experiment_label)
        log_file = open(f'{experiment_directory_path}/{experiment_label}.dat', 'w')

        # setup the parallel processing pool
        # num_cores = multiprocessing.cpu_count()
        # pool = multiprocessing.Pool(num_cores)

        # create the CVT
        kdt = self._initialise_kdt_and_centroids(
            experiment_directory_path=experiment_directory_path,
            number_of_niches=number_of_niches,
            run_parameters=run_parameters,
        )
        archive = {} # init archive (empty)
        n_evals = 0 # number of evaluations since the beginning
        b_evals = 0 # number evaluation since the last dump

        # main loop
        configuration_counter = 0
        pbar = tqdm(desc="Number of evaluations", total=maximum_evaluations)
        random_initialisation = True
        while (n_evals < maximum_evaluations): ### NUMBER OF GENERATIONS
            to_evaluate = []
            # random initialization
            population = []
            if random_initialisation:
                individuals = self.crystal_system.create_n_individuals(
                    run_parameters['random_init_batch']) # individuals are dict
            #     population += individuals
            #     random_initialisation = False
                _, known_atoms = get_all_materials_with_formula("TiO2")
                # individuals = []
                for i, atoms in enumerate(known_atoms):
                    if len(atoms.get_atomic_numbers()) == run_parameters["filter_starting_Structures"]:
                        atoms.rattle()
                        atoms.info["confid"] = None
                        atoms = atoms.todict()
                        individuals.append(atoms)
                    del known_atoms
                # individuals = [AseAtomsAdaptor.get_atoms(structure) for structure in known_structures]
                population += individuals # all individuals are dictionary

            if random_initialisation:
                random_initialisation = False
            # if len(archive) <= run_parameters['random_init'] * number_of_niches:
                # individuals = self.crystal_system.create_n_individuals(
                #     run_parameters['random_init_batch'])
                #
                # population += individuals
                # for i in range(0, run_parameters['random_init_batch']):
                #     x = self.crystal_system.create_one_individual(individual_id=i)
                #     # individuals = self.crystal_system.create_n_individuals(run_parameters['random_init_batch'])
                #
                #     population += [x]
                # continue

            else:  # variation/selection loop
                keys = list(archive.keys())
                # we select all the parents at the same time because randint is slow
                # if run_parameters["curiosity_weights"]:
                #     weights = self.compute_curiosity_weights(archive=archive)
                # else:
                #     weights = np.full(len(keys), 1/len(keys))
                # rand1 = np.random.choice(np.arange(len(keys)), size=run_parameters['batch_size'], p=weights)
                # rand2 = np.random.choice(np.arange(len(keys)), size=run_parameters['batch_size'], p=weights)
                rand1 = np.random.randint(len(keys), size=run_parameters['batch_size'])
                rand2 = np.random.randint(len(keys), size=run_parameters['batch_size'])

                for n in range(0, run_parameters['batch_size']):
                    # parent selection
                    x = archive[keys[rand1[n]]]
                    y = archive[keys[rand2[n]]]
                    # copy & add variation
                    z, _ = self.crystal_system.operators.get_new_individual([Atoms.fromdict(x.x), Atoms.fromdict(y.x)])
                    z = z.todict()
                    # z.info["curiosity"] = 0
                    population += [z]

            # todo: update every 10 individuals tested
            # if population is not None:
            #     self.crystal_system.update_operator_scaling_volumes(population=population)

            for i in range(len(population)):
                x = population[i]
                really_relax = True

                to_evaluate += [(x, self.crystal_system.cellbounds,
                                 run_parameters["behavioural_descriptors"],
                                 run_parameters["number_of_relaxation_steps"],
                                 self.crystal_evaluator.compute_fitness_and_bd)]

            # chunks = chunkify_population(population)

            s_list = evaluate_parallel(to_evaluate)

            # s_list = pool.map(evaluate_parallel, to_evaluate, chunksize=2)
            # s_list = sum(s_list)
            # print()

            # to_evaluate = self.create_evaluate_list(
            #     population,
            #     self.crystal_system.cellbounds,
            #     run_parameters["behavioural_descriptors"],
            #     run_parameters["number_of_relaxation_steps"],
            #     # self.crystal_evaluator.compute_fitness_and_bd
            # )
            #
            # s_list = self.evaluate_parallel_old(
            #     to_evaluate
            # )

            # evaluation of the fitness for to_evaluate

            # s_list = self.evaluate_parallel_old(to_evaluate)
            # s_list = self.evaluate_parallel(
            #     population,
            #     self.crystal_system.cellbounds,
            #     run_parameters["behavioural_descriptors"],
            #     run_parameters["number_of_relaxation_steps"],
            #     self.crystal_evaluator.compute_fitness_and_bd
            # )
            # s_list = self.parallel_evaluator(
            #     population,
            #     self.crystal_system.cellbounds,
            #     run_parameters["behavioural_descriptors"],
            #     run_parameters["number_of_relaxation_steps"],
            # )

            # natural selection

            for s in s_list:
                if s is None:
                    continue
                else:
                    s.x["info"]["confid"] = configuration_counter
                    configuration_counter += 1
                    add_to_archive(s, s.desc, archive, kdt)
                    # individual_added, parent_id_list = add_to_archive(s, s.desc, archive, kdt)
                    # archive = self.update_parent_curiosity(archive, parent_id_list, individual_added)

            # count evals
            n_evals += len(to_evaluate)
            b_evals += len(to_evaluate)

            # write archive
            if b_evals >= run_parameters['dump_period'] and run_parameters['dump_period'] != -1:
                print("[{}/{}]".format(n_evals, int(maximum_evaluations)), end=" ", flush=True)
                save_archive(archive, n_evals, experiment_directory_path)
                b_evals = 0
            # write log
            if log_file != None:
                fit_list = np.array([x.fitness for x in archive.values()])
                qd_score = np.sum(fit_list)
                coverage = 100 * len(fit_list) / number_of_niches

                log_file.write("{} {} {} {} {} {} {} {} {}\n".format(n_evals, len(archive.keys()),
                        fit_list.max(), np.mean(fit_list), np.median(fit_list),
                        np.percentile(fit_list, 5), np.percentile(fit_list, 95), coverage, qd_score))
                log_file.flush()
            pbar.update(len(to_evaluate))

        save_archive(archive, n_evals, experiment_directory_path)
        return experiment_directory_path, archive


    def _initialise_kdt_and_centroids(self, experiment_directory_path, number_of_niches, run_parameters):
        # create the CVT
        c = cvt(number_of_niches, self.number_of_bd_dimensions,
                run_parameters['cvt_samples'],
                run_parameters["bd_minimum_values"],
                run_parameters["bd_maximum_values"],
                experiment_directory_path,
                run_parameters["behavioural_descriptors"],
                run_parameters['cvt_use_cache'],
                formula=self.crystal_system.compound_formula,
                )
        kdt = KDTree(c, leaf_size=30, metric='euclidean')
        write_centroids(
            c, experiment_folder=experiment_directory_path,
            bd_names=run_parameters["behavioural_descriptors"],
            bd_minimum_values=run_parameters["bd_minimum_values"],
            bd_maximum_values=run_parameters["bd_maximum_values"],
            formula=self.crystal_system.compound_formula
        )
        del c
        return kdt

    # @jit(parallel=True)
    def mutate_individuals(self, archive, run_parameters):
        keys = list(archive.keys())
        # we select all the parents at the same time because randint is slow
        rand1 = np.random.randint(len(keys), size=run_parameters['batch_size'])
        rand2 = np.random.randint(len(keys), size=run_parameters['batch_size'])
        mutated_offspring = []
        for n in range(0, run_parameters['batch_size']):
            # parent selection
            x = archive[keys[rand1[n]]]
            y = archive[keys[rand2[n]]]
            # copy & add variation
            z, _ = self.crystal_system.operators.get_new_individual([x.x, y.x])
            mutated_offspring += [z]
        return mutated_offspring


    def update_parent_curiosity(self, archive, parent_id_list: List[int], offspring_added_to_archive):
        # map id to niche
        id_niche_mapping = {value.x.info['confid']: key for key, value in archive.items()}
        for parent_id in parent_id_list:
            if parent_id in id_niche_mapping:
                niche = id_niche_mapping[parent_id]
                if offspring_added_to_archive:
                    archive[niche].x.info["curiosity"] += 1
                else:
                    archive[niche].x.info["curiosity"] -= 0.5
            else:
                continue
        return archive

    def compute_curiosity_weights(self, archive):
        all_scores = [individual.x.info["curiosity"] for individual in archive.values()]
        all_scores = np.array(all_scores)
        minimum_value = np.min(all_scores)
        if minimum_value < 0:
            all_scores = all_scores + (minimum_value + 0.5)
        elif minimum_value == 0:
            all_scores = all_scores + 0.5

        sum_of_scores = np.sum(all_scores)
        weights = all_scores / sum_of_scores
        return weights


    def start_experiment_from_archive(self, experiment_directory_path: str,
                                      experiment_label: str, archive_number:int,
                                      run_parameters,
                                      number_of_niches, maximum_evaluations):


        # experiment_directory_path = make_experiment_folder(experiment_label)
        log_file = open(f'{experiment_directory_path}/{experiment_label}_continued.dat', 'w')

        kdt, c = self._initialise_kdt_and_centroids(
            experiment_directory_path=experiment_directory_path,
            number_of_niches=number_of_niches,
            run_parameters=run_parameters
        )

        archive = {}  # init archive (empty)
        archive = self._convert_saved_archive_to_experiment_archive(
            experiment_directory_path=experiment_directory_path,
            archive_number=archive_number,
            experiment_label=experiment_label,
            kdt=kdt,
            archive=archive
        )

        n_evals = archive_number  # number of evaluations since the beginning
        b_evals = 0  # number evaluation since the last dump

        # main loop
        configuration_counter = 0 # todo: change configuration number to be alrger than the last one in the archive
        n_evals_to_do = maximum_evaluations - archive_number
        pbar = tqdm(desc="Number of evaluations", total=maximum_evaluations)
        random_initialisation = False
        snapshots = []
        # snapshots.append(tracemalloc.sna)
        while (n_evals < maximum_evaluations):  ### NUMBER OF GENERATIONS
            to_evaluate = []
            # random initialization
            population = []

            if random_initialisation:
                random_initialisation = False

            else:  # variation/selection loop
                keys = list(archive.keys())
                rand1 = np.random.randint(len(keys), size=run_parameters['batch_size'])
                rand2 = np.random.randint(len(keys), size=run_parameters['batch_size'])

                for n in range(0, run_parameters['batch_size']):
                    # parent selection
                    x = archive[keys[rand1[n]]]
                    y = archive[keys[rand2[n]]]
                    # copy & add variation
                    z, _ = self.crystal_system.operators.get_new_individual([Atoms.fromdict(x.x), Atoms.fromdict(y.x)])
                    z = z.todict()
                    population += [z]

            for i in range(len(population)):
                x = population[i]
                really_relax = True

                to_evaluate += [(x, self.crystal_system.cellbounds,
                                 run_parameters["behavioural_descriptors"],
                                 run_parameters["number_of_relaxation_steps"],
                                 self.crystal_evaluator.compute_fitness_and_bd)]

            s_list = evaluate_parallel(to_evaluate)

            # natural selection

            for s in s_list:
                if s is None:
                    continue
                else:
                    s.x["info"]["confid"] = configuration_counter
                    configuration_counter += 1
                    # s.calc = None
                    add_to_archive(s, s.desc, archive, kdt)


            # count evals
            n_evals += len(to_evaluate)
            b_evals += len(to_evaluate)

            # write archive
            if b_evals >= run_parameters['dump_period'] and run_parameters['dump_period'] != -1:
                print("[{}/{}]".format(n_evals, int(maximum_evaluations)), end=" ", flush=True)
                if n_evals == archive_number:
                    continue
                else:
                    save_archive(archive, n_evals, experiment_directory_path)
                b_evals = 0
            # write log
            if log_file != None:
                fit_list = np.array([x.fitness for x in archive.values()])
                qd_score = np.sum(fit_list)
                coverage = 100 * len(fit_list) / len(c)

                log_file.write("{} {} {} {} {} {} {} {} {}\n".format(n_evals, len(archive.keys()),
                                                                     fit_list.max(),
                                                                     np.mean(fit_list),
                                                                     np.median(fit_list),
                                                                     np.percentile(fit_list, 5),
                                                                     np.percentile(fit_list, 95),
                                                                     coverage, qd_score))
                log_file.flush()
            pbar.update(len(to_evaluate))

        save_archive(archive, n_evals, experiment_directory_path)
        return experiment_directory_path, archive

    def _convert_saved_archive_to_experiment_archive(self, experiment_directory_path,
                                                     experiment_label, archive_number, kdt, archive,
                                                     individual_type = "atoms"):
        fitnesses, centroids, descriptors, individuals = load_archive_from_pickle(
            filename=f"{experiment_directory_path}/archive_{archive_number}.pkl")

        if isinstance(individuals[0], Atoms):
            species_list = [
                Species(x=individuals[i].todict(), desc=descriptors[i], fitness=fitnesses[i], centroid=None)
                for i in range(len(individuals))
            ]
        elif isinstance(individuals[0], dict):
            species_list = [
                Species(x=individuals[i], desc=descriptors[i], fitness=fitnesses[i],
                        centroid=None)
                for i in range(len(individuals))
            ]
        for i in range(len(species_list)):
            add_to_archive(species_list[i], descriptors[i], archive=archive, kdt=kdt)

        return archive
