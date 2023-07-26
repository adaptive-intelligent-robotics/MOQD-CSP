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
from typing import List, Optional

import numpy as np
# from numba import jit, prange
from sklearn.neighbors import KDTree
from tqdm import tqdm

from csp_elites.crystal.crystal_evaluator import CrystalEvaluator
from csp_elites.crystal.crystal_system import CrystalSystem
from csp_elites.map_elites.elites_utils import cvt, save_archive, evaluate, add_to_archive, \
    write_centroids, make_experiment_folder, Species, evaluate_parallel


class CVT:
    def __init__(self,
        number_of_bd_dimensions: int,
        crystal_system: CrystalSystem,
        crystal_evaluator: CrystalEvaluator,

    ):
        self.number_of_bd_dimensions = number_of_bd_dimensions
        self.crystal_system = crystal_system
        self.crystal_evaluator = crystal_evaluator

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
        c = cvt(number_of_niches, self.number_of_bd_dimensions,
                run_parameters['cvt_samples'],
                run_parameters["bd_minimum_values"],
                run_parameters["bd_maximum_values"],
                experiment_directory_path,
                run_parameters["behavioural_descriptors"],
                run_parameters['cvt_use_cache'],
                )
        kdt = KDTree(c, leaf_size=30, metric='euclidean')
        write_centroids(
            c, experiment_folder=experiment_directory_path,
            bd_names=run_parameters["behavioural_descriptors"],
            bd_minimum_values=run_parameters["bd_minimum_values"],
            bd_maximum_values=run_parameters["bd_maximum_values"],
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
            # if random_initialisation:
            #     individuals = self.crystal_system.create_n_individuals(
            #         run_parameters['random_init_batch'])
            #     population += individuals
            #     random_initialisation = False

            if len(archive) <= run_parameters['random_init'] * number_of_niches:
                # individuals = self.crystal_system.create_n_individuals(
                #     run_parameters['random_init_batch'])
                #
                # population += individuals
                for i in range(0, run_parameters['random_init_batch']):
                    x = self.crystal_system.create_one_individual(individual_id=i)
                    # individuals = self.crystal_system.create_n_individuals(run_parameters['random_init_batch'])

                    population += [x]

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
                    z, _ = self.crystal_system.operators.get_new_individual([x.x, y.x])
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
                    s.x.info["confid"] = configuration_counter
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
                coverage = 100 * len(fit_list) / len(c)

                log_file.write("{} {} {} {} {} {} {} {} {}\n".format(n_evals, len(archive.keys()),
                        fit_list.max(), np.mean(fit_list), np.median(fit_list),
                        np.percentile(fit_list, 5), np.percentile(fit_list, 95), coverage, qd_score))
                log_file.flush()
            pbar.update(n_evals)

        save_archive(archive, n_evals, experiment_directory_path)
        return experiment_directory_path, archive

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

    # @jit(parallel=True)
    def create_evaluate_list(self, population, cellbounds, bd, nb_relaxation_steps): # fitness_function
        to_evaluate = []
        for i in range(len(population)):
            x = population[i]
            to_evaluate += [(x, cellbounds, bd, nb_relaxation_steps)] # fitness_function
        return to_evaluate

    # @jit(parallel=True)
    def evaluate_parallel_old(self, to_evaluate):
        s_list = []

        for i in range(len(to_evaluate)):
            z, cellbounds, behavioural_descriptors, n_relaxation_steps, f = to_evaluate[i]
            s = evaluate(
                z, cellbounds, behavioural_descriptors, n_relaxation_steps, f
            )
            s_list.append(s)
        return s_list

    # @jit(parallel=True)
    def evaluate_parallel(self, population, cellbounds, bd, nb_relaxation_steps, fitness_function) -> List[Optional[Species]]:
        s_list = []
        for i in range(len(population)):
            x = population[i]
            s = evaluate(
                x,
                cellbounds,
                bd,
                nb_relaxation_steps,
                fitness_function
            )
            s_list.append(s)
        return s_list

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
