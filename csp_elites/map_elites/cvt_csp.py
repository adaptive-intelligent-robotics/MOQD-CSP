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
import pathlib

from csp_elites.crystal.crystal_evaluator import CrystalEvaluator
from csp_elites.crystal.crystal_system import CrystalSystem

import numpy as np
# import multiprocessing
import pathos.multiprocessing as multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool

from sklearn.neighbors import KDTree
from tqdm import tqdm
from numba import jit

from csp_elites.crystal.crytsl_evaluator_parallel import parallel_fitness_func_and_bd_computation
from csp_elites.map_elites.elites_utils import cvt, save_archive, parallel_eval, evaluate, add_to_archive, \
    write_centroids, make_experiment_folder
from csp_elites.utils.plot import plot_all_statistics_from_file


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
           Vassiliades V, Chatzilygeroudis K, Mouret JB. Using centroidal voronoi tessellations to scale up the multidimensional archive of phenotypic elites algorithm. IEEE Transactions on Evolutionary Computation. 2017 Aug 3;22(4):623-30.

           Format of the logfile: evals archive_size max mean median 5%_percentile, 95%_percentile

        """
        # setup the parallel processing pool

        experiment_directory_path = make_experiment_folder(experiment_label)
        log_file = open(f'{experiment_directory_path}/{experiment_label}.dat', 'w')

        num_cores = multiprocessing.cpu_count()
        # pool = multiprocessing.Pool(num_cores)
        pool = Pool(num_cores)

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
        while (n_evals < maximum_evaluations): ### NUMBER OF GENERATIONS
            to_evaluate = []
            # random initialization
            population = []
            if len(archive) < run_parameters['random_init'] * number_of_niches:
                for i in range(0, run_parameters['random_init_batch']):
                    x = self.crystal_system.create_one_individual(individual_id=i)

                    population += [x]

            else:  # variation/selection loop
                keys = list(archive.keys())
                # we select all the parents at the same time because randint is slow
                rand1 = np.random.randint(len(keys), size=run_parameters['batch_size'])
                rand2 = np.random.randint(len(keys), size=run_parameters['batch_size'])
                for n in range(0, run_parameters['batch_size']):
                    # parent selection
                    x = archive[keys[rand1[n]]]
                    y = archive[keys[rand2[n]]]
                    # copy & add variation
                    z, _ = self.crystal_system.operators.get_new_individual([x.x, y.x])
                    population += [z]

            # todo: update every 10 individuals tested
            if population is not None:
                self.crystal_system.update_operator_scaling_volumes(population=population)

            for i in range(len(population)):
                x = population[i]

                if n_evals % 50 == 0 and n_evals >= 50:
                    really_relax = False
                else:
                    really_relax = False
                # self.crystal_evaluator.marta_relax_cells = np.random.choice([True, False], p=[1, 0])
                # really_relax = np.random.choice([True, False], p=[run_parameters["relaxation_probability"], 1 - run_parameters["relaxation_probability"]])
                if run_parameters["parallel"]:
                    to_evaluate += [(x, self.crystal_system.cellbounds, population, really_relax,
                                     run_parameters["behavioural_descriptors"],
                                     self.crystal_evaluator.compute_fitness_and_bd)]
                else:
                    to_evaluate += [(x, self.crystal_system.cellbounds, population, really_relax,
                                     run_parameters["behavioural_descriptors"],
                                     parallel_fitness_func_and_bd_computation)]


            # evaluation of the fitness for to_evaluate
            s_list = parallel_eval(evaluate, to_evaluate, pool, run_parameters) # TODO: what happens when individual killed by me?

            # natural selection

            for s in s_list:
                if s is None:
                    continue
                else:
                    s.x.info["confid"] = configuration_counter
                    configuration_counter += 1
                    add_to_archive(s, s.desc, archive, kdt)
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
