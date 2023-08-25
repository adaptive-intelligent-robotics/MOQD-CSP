import pickle

import numpy as np
from ase import Atoms
from ase.ga.utilities import CellBounds
from chgnet.graph import CrystalGraphConverter
from ribs.emitters.opt import _get_es, _get_grad_opt
from ribs.emitters.rankers import _get_ranker
from sklearn.neighbors import KDTree

from csp_elites.crystal.crystal_evaluator import CrystalEvaluator
from csp_elites.crystal.crystal_system import CrystalSystem
from csp_elites.map_elites.elites_utils import cvt, write_centroids, make_experiment_folder, \
    Species, add_to_archive, save_archive


class CMAMEGALOOP:
    def __init__(self,
                 number_of_bd_dimensions: int,
                 crystal_system: CrystalSystem,
                 crystal_evaluator: CrystalEvaluator,
                 step_size_gradient_optimizer_niu: float = 1,
                 initial_cmaes_step_size_sigma_g: float = 0.05,
                 ):
        self.number_of_bd_dimensions = number_of_bd_dimensions
        self.crystal_system = crystal_system
        self.crystal_evaluator = crystal_evaluator
        self.graph_converter = CrystalGraphConverter()
        self.step_size_gradient_optimizer_niu = step_size_gradient_optimizer_niu
        self.initial_cmaes_step_size_sigma_g = initial_cmaes_step_size_sigma_g

        self.num_coefficeints = number_of_bd_dimensions + 1

        self.coef_optimizer = _get_es("cma_es",
                                      sigma0=initial_cmaes_step_size_sigma_g,
                                      batch_size=None,
                                      solution_dim=self.num_coefficeints,
                                      seed=None,
                                      dtype=np.float64,
                                      )
        self.coef_optimizer.reset(np.zeros(3))

        self._coeff_lower_bounds = np.full(
            self.num_coefficeints,
            -np.inf,
            dtype=np.float64,
        )
        self._coeff_upper_bounds = np.full(
            self.num_coefficeints,
            np.inf,
            dtype=np.float64,
        )
        self.crystal_evaluator = CrystalEvaluator(
            compute_gradients=True,
            constrained_qd=False,
            force_threshold_fmax=1,
            with_force_threshold=False,
            relax_every_n_generations=False,
            fmax_relaxation_convergence=0.2,
        )
        self.cellbounds = CellBounds(
            bounds={'phi': [20, 160], 'chi': [20, 160], 'psi': [20, 160], 'a': [2, 40], 'b': [2, 40],
                    'c': [2, 40]})

        self._ranker = _get_ranker("imp")
        self._rng = np.random.default_rng(None)
        self._ranker.reset(self, None, self._rng)
        self._selection_rule = "filter"
        self._grad_opt = _get_grad_opt(
            "gradient_ascent",
            theta0=np.zeros(self.num_coefficeints),
            lr=self.step_size_gradient_optimizer_niu
        )
        self._restart_rule = 'no_improvement'
    def normalize_all_gradients_at_once(self, gradients: np.ndarray):
        norms = np.linalg.norm(gradients, axis=2, keepdims=True)
        gradients /= norms
        return gradients

    def initialise_starting_parameters(self, number_of_niches, maximum_evaluations, run_parameters, experiment_label):
        self.experiment_directory_path = make_experiment_folder(experiment_label)
        self.log_file = open(f'{self.experiment_directory_path}/{experiment_label}.dat', 'w')
        self.memory_log = open(f'{self.experiment_directory_path}/memory_log.dat', 'w')
        with open(f'{self.experiment_directory_path}/experiment_parameters.pkl', 'wb') as file:
            pickle.dump(run_parameters, file)

        self.archive = {}  # init archive (empty)
        self.n_evals = 0  # number of evaluations since the beginning
        self.b_evals = 0  # number evaluation since the last dump
        self.n_relaxation_steps = run_parameters["number_of_relaxation_steps"]
        self.configuration_counter = 0
        # create the CVT
        self.kdt = self._initialise_kdt_and_centroids(
            experiment_directory_path=self.experiment_directory_path,
            number_of_niches=number_of_niches,
            run_parameters=run_parameters,
        )
        self.number_of_niches = number_of_niches
        self.run_parameters = run_parameters
        self.maximum_evaluations = maximum_evaluations


    def compute(self,
                      number_of_niches,
                      maximum_evaluations,
                      run_parameters,
                      experiment_label,
                      ):
        self.initialise_starting_parameters(number_of_niches,maximum_evaluations, run_parameters, experiment_label)
        print("hello cma")
        solution_theta = self.crystal_system.create_one_individual(self.configuration_counter)
        solution_theta = solution_theta.todict()
        self._grad_opt = _get_grad_opt(
            "gradient_ascent",
            theta0=solution_theta["positions"],
            lr=self.step_size_gradient_optimizer_niu
        )
        while self.n_evals < maximum_evaluations:
            #3: f, ∇f , m, ∇m ← evaluate(θ)
            _, population, fitness_scores, descriptors, kill_list, gradients = self.crystal_evaluator.batch_compute_fitness_and_bd(
                list_of_atoms=[solution_theta],
                n_relaxation_steps=self.n_relaxation_steps
            )
            solution_theta = population[0]

            # 4: ∇f ← normalize(∇f ), ∇m ← normalize(∇m)
            gradient_stack = np.stack([gradients[0][0][:len(solution_theta["positions"])],
                                         gradients[0][1],
                                         gradients[0][2]])
            all_gradients_normalised = self.normalize_all_gradients_at_once(gradient_stack)

            # 5: update_archive(θ, f, m)
            self.update_archive(population, fitness_scores, descriptors, kill_list, gradients)
            # 6: for i ← 1 to λ do
            # 7: c ∼ N (μ, Σ)
            coeficients = self.coef_optimizer.ask(
                self._coeff_lower_bounds,
                self._coeff_upper_bounds,
            )[:, :, None].reshape((-1, self.num_coefficeints, 1, 1))

            # 8: ∇i ← c0∇f + ∑k j=1 cj ∇mj
            gradient_mutation_amount = coeficients * all_gradients_normalised
            gradient_mutation_amount = gradient_mutation_amount.sum(axis=1)

            # 9: θ′ i ← θ + ∇i
            new_atoms = [Atoms.fromdict(solution_theta) for _ in range(self.coef_optimizer.batch_size)]
            for i, atom in enumerate(new_atoms):
                new_atoms[i].set_positions(new_atoms[i].get_positions() + gradient_mutation_amount[i])
            new_atoms = [el.todict() for el in new_atoms]

            # 10: f ′, ∗, m′, ∗ ← evaluate(θ′ i)
            updated_atoms, population, fitness_scores, descriptors, kill_list, all_gradients = self.crystal_evaluator.batch_compute_fitness_and_bd(
                new_atoms,
                n_relaxation_steps=self.n_relaxation_steps,
            )

            # 11: ∆i ← update_archive(θ′ i, f ′, m′)
            self.update_archive(population, fitness_scores, descriptors, kill_list, all_gradients)

            # 12: end loop
            # 13: rank ∇i by ∆i
            solution_batch = updated_atoms
            objective_batch = np.asarray(fitness_scores)
            measures_batch = np.asarray(descriptors)
            status_batch = None
            value_batch = np.asarray(fitness_scores)  # TODO: this needs to be fixed to be better
            batch_size = len(solution_batch)
            metadata_batch = np.empty(batch_size, dtype=object)

            indices, ranking_values = self._ranker.rank(
                self, None, self._rng, solution_batch, objective_batch,
                measures_batch, status_batch, value_batch, metadata_batch)

            # 14: ∇step ← ∑λ i=1 wi∇rank[i]
            num_parents = self.coef_optimizer.batch_size // 2  # todo this can be based on new solutions updated_atoms if self._selection_rule == "filter" else


            parents = [el.get_positions() for i, el in enumerate(updated_atoms) if i in indices]
            parents = parents[:num_parents]
            weights = (np.log(num_parents + 0.5) -
                       np.log(np.arange(1, num_parents + 1)))
            weights = weights / np.sum(weights)  # Normalize weights
            new_mean = (parents * weights).sum(axis=2).sum(axis=1)

            #15: θ ← θ + η∇step
            gradient_step = new_mean - self._grad_opt.theta
            self._grad_opt.step(gradient_step)

            new_positions = solution_theta["positions"] + self.step_size_gradient_optimizer_niu * new_mean
            solution_theta = solution_theta

            #16: Adapt CMA-ES parameters μ, Σ, p based on improvement ranking ∆i
            self.coef_optimizer.tell(indices, num_parents)

            # 17: if there is no change in the archive then
            if (self.coef_optimizer.check_stop(ranking_values[indices]) or
                    self._check_restart(new_positions)).all():
                print("restarting")
                # 19: Set θ to a randomly selected existing cell θi from the archive
                solution_theta = self.get_random_individual_from_archive()
                new_coeff = solution_theta["positions"]

                # 18: Restart CMA-ES with μ = 0, Σ = σgI.
                self._grad_opt.reset(new_coeff)
                self.coef_optimizer.reset(np.zeros(self.num_coefficeints))
                self._ranker.reset(self, self.archive, self._rng)

            # print(self.archive)


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
            formula=self.crystal_system.compound_formula,
        )
        del c
        return kdt

    def update_archive(self, population, fitness_scores, descriptors, kill_list, gradients):
        s_list = self.crystal_evaluator.batch_create_species(population, fitness_scores,
                                                             descriptors, kill_list, gradients)
        evaluations_performed = len(population)
        self.n_evals += evaluations_performed
        self.b_evals += evaluations_performed
        for s in s_list:
            if s is None:
                continue
            else:
                s.x["info"]["confid"] = self.configuration_counter
                self.configuration_counter += 1
                add_to_archive(s, s.desc, self.archive, self.kdt)
        if self.b_evals >= self.run_parameters['dump_period'] and self.run_parameters['dump_period'] != -1:
            print("[{}/{}]".format(self.n_evals, int(self.maximum_evaluations)), end=" ", flush=True)
            save_archive(self.archive, self.n_evals, self.experiment_directory_path)
            self.b_evals = 0
        # write log
        if self.log_file != None:
            fit_list = np.array([x.fitness for x in self.archive.values()])
            qd_score = np.sum(fit_list)
            coverage = 100 * len(fit_list) / self.number_of_niches

            self.log_file.write("{} {} {} {} {} {} {} {} {}\n".format(self.n_evals,
                                                                 len(self.archive.keys()),
                                                                 np.max(fit_list),
                                                                 np.mean(fit_list),
                                                                 np.median(fit_list),
                                                                 np.percentile(fit_list, 5),
                                                                 np.percentile(fit_list, 95),
                                                                 coverage, qd_score))
            self.log_file.flush()

    def get_random_individual_from_archive(self):
        individuals = [species.x for species in self.archive.values()]
        random_index = np.random.randint(0, len(individuals))
        return individuals[random_index]

    def _check_restart(self, num_parents):
        """Emitter-side checks for restarting the optimizer.

        The optimizer also has its own checks.

        Args:
            num_parents (int): The number of solution to propagate to the next
                generation from the solutions generated by CMA-ES.
        Raises:
          ValueError: If :attr:`restart_rule` is invalid.
        """
        if isinstance(self._restart_rule, (int, np.integer)):
            return self._itrs % self._restart_rule == 0
        if self._restart_rule == "no_improvement":
            return num_parents == 0
        if self._restart_rule == "basic":
            return False
        raise ValueError(f"Invalid restart_rule {self._restart_rule}")
