from typing import List

import numpy as np
from ase import Atoms
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.utilities import atoms_too_close, atoms_too_close_two_sets, closest_distances_generator, \
    CellBounds
from chgnet.model import CHGNet, CHGNetCalculator
from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor
from ribs.emitters.opt import CMAEvolutionStrategy, _get_es, _get_grad_opt
from ribs.emitters.rankers import _get_ranker

from csp_elites.crystal.crystal_evaluator import CrystalEvaluator
from csp_elites.map_elites.elites_utils import Species


class GradientMutation(OffspringCreator):
    """An implementation of the rattle mutation as described in:

    R.L. Johnston Dalton Transactions, Vol. 22,
    No. 22. (2003), pp. 4193-4207

    Parameters:

    blmin: Dictionary defining the minimum distance between atoms
        after the rattle.

    n_top: Number of atoms optimized by the GA.

    rattle_strength: Strength with which the atoms are moved.

    rattle_prop: The probability with which each atom is rattled.

    test_dist_to_slab: whether to also make sure that the distances
        between the atoms and the slab satisfy the blmin.

    use_tags: if True, the atomic tags will be used to preserve
        molecular identity. Same-tag atoms will then be
        displaced collectively, so that the internal
        geometry is preserved.

    rng: Random number generator
        By default numpy.random.
    """
    # def __init__(self, test_dist_to_slab=True, use_tags=False,
    #              verbose=False, rng=np.random):
    def __init__(self, blmin, n_top, learning_rate: float = 0.01, test_dist_to_slab=True, use_tags=False,
                 verbose=False, rng=np.random):
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.test_dist_to_slab = test_dist_to_slab
        self.use_tags = use_tags

        self.descriptor = 'GradientMutation'
        self.min_inputs = 1
        self.model = CHGNet.load()
        self.learning_rate = learning_rate

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.mutate(f)
        if indi is None:
            return indi, 'mutation: gradient rattle'

        indi = self.initialize_individual(f, indi)
        indi.info['data']['parents'] = [f.info['confid']]

        return self.finalize_individual(indi), 'mutation: gradient rattle'

    def mutate(self, atoms):
        """Does the actual mutation."""
        prediction = self.model.predict_structure(AseAtomsAdaptor.get_structure(atoms))
        forces = prediction["f"]

        # N = len(atoms) if self.n_top is None else self.n_top
        N = len(atoms)
        slab = atoms[:len(atoms) - N]
        atoms = atoms[-N:]
        tags = atoms.get_tags() if self.use_tags else np.arange(N)
        pos_ref = atoms.get_positions()
        num = atoms.get_atomic_numbers()
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        # st = 2. * self.rattle_strength

        count = 0
        maxcount = 1000
        too_close = True
        while too_close and count < maxcount:
            count += 1
            pos = pos_ref.copy()
            # ok = False
            # too_close = False
            pos += self.learning_rate * forces
            # for tag in np.unique(tags):
            #     select = np.where(tags == tag)
            #     # if self.rng.rand() < self.rattle_prop:
            #     #     ok = True
            #     #     r = self.rng.rand(3)
            #     #     pos[select] += st * (r - 0.5)
            #
            # if not ok:
            #     # Nothing got rattled
            #     continue

            top = Atoms(num, positions=pos, cell=cell, pbc=pbc, tags=tags)
            too_close = atoms_too_close(
                top, self.blmin, use_tags=self.use_tags)
            if not too_close and self.test_dist_to_slab:
                too_close = atoms_too_close_two_sets(top, slab, self.blmin)

        if count == maxcount:
            del prediction
            return None

        del prediction
        mutant = slab + top
        return mutant


class DQDMutationCMAMEGA(OffspringCreator):
    """

    Parameters:

    blmin: Dictionary defining the minimum distance between atoms
        after the rattle.

    n_top: Number of atoms optimized by the GA.

    rattle_strength: Strength with which the atoms are moved.

    rattle_prop: The probability with which each atom is rattled.

    test_dist_to_slab: whether to also make sure that the distances
        between the atoms and the slab satisfy the blmin.

    use_tags: if True, the atomic tags will be used to preserve
        molecular identity. Same-tag atoms will then be
        displaced collectively, so that the internal
        geometry is preserved.

    rng: Random number generator
        By default numpy.random.
    """
    # def __init__(self, test_dist_to_slab=True, use_tags=False,
    #              verbose=False, rng=np.random):
    def __init__(self, blmin, n_top, learning_rate: float = 1., test_dist_to_slab=True, use_tags=False,
                 rattle_prop=0.4,
                 verbose=False, rng=np.random):
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.test_dist_to_slab = test_dist_to_slab
        self.use_tags = use_tags

        self.descriptor = 'DQDMutation'
        self.min_inputs = 1
        # self.model = CHGNet.load()
        self.learning_rate = learning_rate
        self.rattle_prop = rattle_prop

        self.num_coefficeints = 2 + 1

        self.coef_optimizer = _get_es("cma_es",
                                      sigma0=0.05,
                                      batch_size=None,
                                      solution_dim=self.num_coefficeints,
                                      seed=None,
                                      dtype=np.float64,
                                      )
                            # **(es_kwargs if es_kwargs is not None else {}))
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
            comparator=None,
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
            "adam",
            theta0=np.zeros(self.num_coefficeints), # todo: what is x0???self._x0
        )
            # lr=lr,
            # **(grad_opt_kwargs if grad_opt_kwargs is not None else {}))


    def get_new_individual(self, parents: List[Species]):
        f = parents[0]

        indi = self.mutate(f)
        if indi is None:
            return indi, 'mutation: DQD rattle'

        indi = self.initialize_individual(f, indi)

        if "confid" in f.x["info"].keys():
            indi.info['data']['parents'] = [f.x["info"]['confid']]
        else:
            indi.info['data']['parents'] = [None]

        return self.finalize_individual(indi), 'mutation: DQD rattle'

    def mutate(self, species: Species):
        """Does the actual mutation."""
        atoms = Atoms.fromdict(species.x)
        # forces = self.normalize_gradient(species.fitness_gradient)
        # forces = forces[:len(species.x["positions"]), :]
        # descriptor_gradients = np.array([self.normalize_gradient(grad) for grad in species.descriptor_gradients])

        gradient_stack = np.vstack([[species.fitness_gradient[:len(species.x["positions"])], species.descriptor_gradients[0],
                                     species.descriptor_gradients[1]]])
        all_gradients_normalised = self.normalize_all_gradients_at_once(gradient_stack)
        # all_gradients_normalised = self.normalize_all_gradients_at_once(np.concatenate([forces, descriptor_gradients[0]]))
        # N = len(atoms) if self.n_top is None else self.n_top
        N = len(atoms)
        slab = atoms[:len(atoms) - N]
        atoms = atoms[-N:]
        tags = atoms.get_tags() if self.use_tags else np.arange(N)
        pos_ref = atoms.get_positions()
        num = atoms.get_atomic_numbers()
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        # st = 2. * self.rattle_strength

        count = 0
        maxcount = 1000
        too_close = True
        while too_close and count < maxcount:
            too_close = False
            count += 1
            pos = pos_ref.copy()
            # coeficients = np.random.default_rng().normal(loc=0.0, scale=self.learning_rate, size=(3)).reshape((3, 1, 1))
            coeficients = self.coef_optimizer.ask(
                self._coeff_lower_bounds,
                self._coeff_upper_bounds,
            )[:, :, None].reshape((-1, self.num_coefficeints, 1, 1))

            gradient_mutation_amount = coeficients * all_gradients_normalised
            gradient_mutation_amount = gradient_mutation_amount.sum(axis=1)

            new_atoms = [Atoms.fromdict(species.x) for _ in range(self.coef_optimizer.batch_size)]
            for i, atom in enumerate(new_atoms):
                new_atoms[i].set_positions(new_atoms[i].get_positions() + gradient_mutation_amount[i])
            new_atoms = [el.todict() for el in new_atoms]
            updated_atoms, _, fitness_scores, descriptors, _, all_gradients = self.crystal_evaluator.batch_compute_fitness_and_bd(
                new_atoms, cellbounds= self.cellbounds ,
                                     really_relax=None, behavioral_descriptor_names=None,
                                     n_relaxation_steps=0
            )

            solution_batch = updated_atoms
            objective_batch = np.asarray(fitness_scores)
            measures_batch = np.asarray(descriptors)
            status_batch = None
            value_batch = np.asarray(fitness_scores) # TODO: this needs to be fixed to be better
            batch_size = len(solution_batch)
            metadata_batch = np.empty(batch_size, dtype=object)
                #               if metadata_batch
                #                                                     is None else np.asarray(
                # metadata_batch, dtype=object))


            indices, ranking_values = self._ranker.rank(
                self, None, self._rng, solution_batch, objective_batch,
                measures_batch, status_batch, value_batch, metadata_batch)

            num_parents =  self.coef_optimizer.batch_size // 2 # todo this can be based on new solutions updated_atoms if self._selection_rule == "filter" else
            self.coef_optimizer.tell(indices, num_parents)

            parents = [el.get_positions() for i, el in enumerate(updated_atoms) if i in indices]
            parents = parents[:num_parents]
            weights = (np.log(num_parents + 0.5) -
                       np.log(np.arange(1, num_parents + 1)))
            weights = weights / np.sum(weights)  # Normalize weights
            # weights.reshape(-1, 1, 1, 1)
            # new_mean = np.sum(parents * weights, axis=1)
            new_mean = (parents * weights).sum(axis=2).sum(axis=1)

            gradient_step = new_mean - self._grad_opt.theta
            self._grad_opt.step(gradient_step)
            top = updated_atoms[indices[0]]
        #     ok = False
        #     for tag in np.unique(tags):
        #         select = np.where(tags == tag)
        #         if self.rng.rand() < self.rattle_prop:
        #             ok = True
        #             # r = self.rng.rand(3)
        #             pos[select] += gradient_mutation_amount[select]
        #     if not ok:
        #         # Nothing got rattled
        #         continue
        #
        #     top = Atoms(num, positions=pos, cell=cell, pbc=pbc, tags=tags)
        #     too_close = atoms_too_close(
        #         top, self.blmin, use_tags=self.use_tags)
        #     if not too_close and self.test_dist_to_slab:
        #         too_close = atoms_too_close_two_sets(top, slab, self.blmin)
        #
        # if count == maxcount:
        #     return None

        mutant = slab + top
        return mutant

    # def normalize_gradient(self, gradient: np.ndarray):
    #     return (gradient - np.min(gradient)) / (np.max(gradient) - np.min(gradient))

    def normalize_all_gradients_at_once(self, gradients: np.ndarray):
        norms = np.linalg.norm(gradients, axis=2, keepdims=True)
        gradients /= norms
        return gradients


class DQDMutationOMGMEGA(OffspringCreator):
    """

    Parameters:

    blmin: Dictionary defining the minimum distance between atoms
        after the rattle.

    n_top: Number of atoms optimized by the GA.

    rattle_strength: Strength with which the atoms are moved.

    rattle_prop: The probability with which each atom is rattled.

    test_dist_to_slab: whether to also make sure that the distances
        between the atoms and the slab satisfy the blmin.

    use_tags: if True, the atomic tags will be used to preserve
        molecular identity. Same-tag atoms will then be
        displaced collectively, so that the internal
        geometry is preserved.

    rng: Random number generator
        By default numpy.random.
    """
    # def __init__(self, test_dist_to_slab=True, use_tags=False,
    #              verbose=False, rng=np.random):
    def __init__(self, blmin, n_top, learning_rate: float = 0.01, test_dist_to_slab=True, use_tags=False,
                 rattle_prop=0.4,
                 verbose=False, rng=np.random):
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.test_dist_to_slab = test_dist_to_slab
        self.use_tags = use_tags

        self.descriptor = 'DQDMutation'
        self.min_inputs = 1
        # self.model = CHGNet.load()
        self.learning_rate = learning_rate
        self.rattle_prop = rattle_prop



    def get_new_individual(self, parents: List[Species]):
        f = parents[0]

        indi = self.mutate(f)
        if indi is None:
            return indi, 'mutation: DQD rattle'

        indi = self.initialize_individual(f, indi)

        if "confid" in f.x["info"].keys():
            indi.info['data']['parents'] = [f.x["info"]['confid']]
        else:
            indi.info['data']['parents'] = [None]

        return self.finalize_individual(indi), 'mutation: DQD rattle'

    def mutate(self, species: Species):
        """Does the actual mutation."""
        atoms = Atoms.fromdict(species.x)
        # forces = self.normalize_gradient(species.fitness_gradient)
        # forces = forces[:len(species.x["positions"]), :]
        # descriptor_gradients = np.array([self.normalize_gradient(grad) for grad in species.descriptor_gradients])

        gradient_stack = np.vstack([[species.fitness_gradient[:len(species.x["positions"])], species.descriptor_gradients[0],
                                     species.descriptor_gradients[1]]])
        all_gradients_normalised = self.normalize_all_gradients_at_once(gradient_stack)
        # all_gradients_normalised = self.normalize_all_gradients_at_once(np.concatenate([forces, descriptor_gradients[0]]))
        # N = len(atoms) if self.n_top is None else self.n_top
        N = len(atoms)
        slab = atoms[:len(atoms) - N]
        atoms = atoms[-N:]
        tags = atoms.get_tags() if self.use_tags else np.arange(N)
        pos_ref = atoms.get_positions()
        num = atoms.get_atomic_numbers()
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        # st = 2. * self.rattle_strength

        count = 0
        maxcount = 1000
        too_close = True
        while too_close and count < maxcount:
            count += 1
            pos = pos_ref.copy()
            coeficients = np.random.default_rng().normal(loc=0.0, scale=self.learning_rate, size=(3)).reshape((3, 1, 1))
            gradient_mutation_amount = coeficients * all_gradients_normalised
            gradient_mutation_amount = gradient_mutation_amount.sum(axis=0)

            ok = False
            for tag in np.unique(tags):
                select = np.where(tags == tag)
                if self.rng.rand() < self.rattle_prop:
                    ok = True
                    # r = self.rng.rand(3)
                    pos[select] += gradient_mutation_amount[select]
            if not ok:
                # Nothing got rattled
                continue

            top = Atoms(num, positions=pos, cell=cell, pbc=pbc, tags=tags)
            too_close = atoms_too_close(
                top, self.blmin, use_tags=self.use_tags)
            if not too_close and self.test_dist_to_slab:
                too_close = atoms_too_close_two_sets(top, slab, self.blmin)

        if count == maxcount:
            return None

        mutant = slab + top
        return mutant

    def normalize_all_gradients_at_once(self, gradients: np.ndarray):
        norms = np.linalg.norm(gradients, axis=2, keepdims=True)
        gradients /= norms
        return gradients



if __name__ == '__main__':
    with MPRester(api_key="4nB757V2Puue49BqPnP3bjRPksr4J9y0") as mpr:
        one_structure = mpr.get_structure_by_material_id("mp-1840", final=True)

    closest_distances = closest_distances_generator(atom_numbers=one_structure.atomic_numbers,
                                                    ratio_of_covalent_radii=0.4)

    target_atoms = AseAtomsAdaptor.get_atoms(one_structure)

    atoms_2 = AseAtomsAdaptor.get_atoms(one_structure)
    atoms_2.calc = CHGNetCalculator()
    atoms_2.rattle(0.1)


    initial_distance = (target_atoms.get_positions() - atoms_2.get_positions()).sum()
    # print((target_atoms.get_positions() - atoms_2.get_positions()).sum())

    gradient_mutator = DQDMutationOMGMEGA(blmin=closest_distances, n_top=len(one_structure.atomic_numbers))

    mutant = gradient_mutator.mutate(atoms_2.todict())

    print((target_atoms.get_positions() - mutant.get_positions()).sum() - initial_distance)
    a = 2
    print()
