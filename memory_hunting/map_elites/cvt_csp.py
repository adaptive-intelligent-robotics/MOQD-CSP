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
from typing import List, Optional

import numpy as np
import psutil
from ase import Atoms
from matplotlib import pyplot as plt
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.neighbors import KDTree
from tqdm import tqdm

from csp_elites.map_elites.elites_utils import cvt, save_archive, evaluate, add_to_archive, \
    write_centroids, make_experiment_folder, Species, evaluate_parallel
from csp_elites.utils.get_mpi_structures import get_all_materials_with_formula
from csp_elites.utils.plot import load_archive_from_pickle
from memory_hunting.crystal.crystal_evaluator import CrystalEvaluatorHunting
from memory_hunting.crystal.crystal_system import CrystalSystemHunting


class CVTHunting:
    def __init__(self,
        number_of_bd_dimensions: int,
        crystal_system: CrystalSystemHunting,
        crystal_evaluator: CrystalEvaluatorHunting,
        remove_mutations,

    ):
        self.number_of_bd_dimensions = number_of_bd_dimensions
        self.crystal_system = crystal_system
        self.crystal_evaluator = crystal_evaluator
        self.remove_mutations = remove_mutations

    def batch_compute_with_list_of_atoms(self,
        number_of_niches,
        maximum_evaluations,
        run_parameters,
        experiment_label,
        ):
        experiment_directory_path = make_experiment_folder(experiment_label)
        log_file = open(f'{experiment_directory_path}/{experiment_label}.dat', 'w')
        memory_log = open(f'{experiment_directory_path}/memory_log.dat', 'w')

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

        ram_logging = []
        pbar = tqdm(desc="Number of evaluations", total=maximum_evaluations, position=2)
        while (n_evals < maximum_evaluations):  ### NUMBER OF GENERATIONS
            ram_logging.append(psutil.virtual_memory()[3]/1000000000)
            # random initialization
            population = []


            if len(archive) <= run_parameters['random_init'] * number_of_niches:
                individuals = self.crystal_system.create_n_individuals(
                    run_parameters['random_init_batch'])

                population += individuals

            else:  # variation/selection loop
                if self.remove_mutations:
                    individuals = self.crystal_system.create_n_individuals(
                        run_parameters['batch_size'])

                    population += individuals
                else:
                    keys = list(archive.keys())
                    rand1 = np.random.randint(len(keys), size=run_parameters['batch_size'])
                    rand2 = np.random.randint(len(keys), size=run_parameters['batch_size'])

                    for n in range(0, run_parameters['batch_size']):
                        # parent selection
                        x = archive[keys[rand1[n]]]
                        y = archive[keys[rand2[n]]]
                        # copy & add variation
                        z, _ = self.crystal_system.operators.get_new_individual(
                            [Atoms.fromdict(x.x), Atoms.fromdict(y.x)])
                        if z is None:
                            print(" z is none bug")
                        else:
                            z = z.todict()
                            population += [z]

            population_as_atoms, population, fitness_scores, descriptors, kill_list = \
                self.crystal_evaluator.batch_compute_fitness_and_bd(
                list_of_atoms=population,
                cellbounds=self.crystal_system.cellbounds,
                really_relax=None,
                behavioral_descriptor_names=run_parameters["behavioural_descriptors"],
                n_relaxation_steps=run_parameters["number_of_relaxation_steps"],
            )
            if population is not None:
                self.crystal_system.update_operator_scaling_volumes(population=population_as_atoms)
                del population_as_atoms

            s_list = self.crystal_evaluator.batch_create_species(population, fitness_scores, descriptors, kill_list)
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
                    if "info" not in s.x.keys():
                        print()
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

    def _initialise_kdt_and_centroids(self, experiment_directory_path, number_of_niches, run_parameters):
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
        del c
        return kdt
