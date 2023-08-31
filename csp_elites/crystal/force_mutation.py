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
    def __init__(self, blmin, n_top, simple: bool = False,
                 learning_rate: float = 0.01, test_dist_to_slab=True, use_tags=False,
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
        self.simple = simple
        self.counter = 0

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
        gradient_stack = np.vstack([species.fitness_gradient[:len(species.x["positions"])]]).reshape((1, len(species.x["positions"]), 3))
        all_gradients_normalised = self.normalize_all_gradients_at_once(gradient_stack).reshape((len(species.x["positions"]), 3))
        N = len(atoms)
        slab = atoms[:len(atoms) - N]
        atoms = atoms[-N:]
        tags = atoms.get_tags() if self.use_tags else np.arange(N)
        pos_ref = atoms.get_positions()
        num = atoms.get_atomic_numbers()
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        # st = 2. * self.rattle_strength
        self.counter += 1
        print(self.counter)
        if self.simple:
            print("simple")
            pos = pos_ref.copy()
            mutate_probability = self.rng.random(size=24)
            gradient_mutation_amount = self.learning_rate * all_gradients_normalised
            gradient_mutation_amount *= (mutate_probability > self.rattle_prop).reshape(-1, 1)
            pos += gradient_mutation_amount
            top = Atoms(num, positions=pos, cell=cell, pbc=pbc, tags=tags)
        else:
            print("full on")
            count = 0
            maxcount = 1000
            too_close = True
            while too_close and count < maxcount:
                count += 1
                pos = pos_ref.copy()
                gradient_mutation_amount = self.learning_rate * all_gradients_normalised
                # gradient_mutation_amount = gradient_mutation_amount.sum(axis=0)

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
    def __init__(self, blmin, n_top, force_only: bool = False, simple: bool = True, learning_rate: float = 0.01, test_dist_to_slab=True, use_tags=False,
                 rattle_prop=0.4,
                 verbose=False, rng=np.random):
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.test_dist_to_slab = test_dist_to_slab
        self.use_tags = use_tags

        self.descriptor = 'DQDMutation'
        self.min_inputs = 1
        self.learning_rate = learning_rate
        # self.rattle_prop = rattle_prop
        self.force_only = force_only
        self.simple = simple

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
        gradient_stack = np.vstack([[species.fitness_gradient[:len(species.x["positions"])], species.descriptor_gradients[0],
                                     species.descriptor_gradients[1]]])
        all_gradients_normalised = self.normalize_all_gradients_at_once(gradient_stack)

        if self.simple:
            print("simple")
            pos = atoms.get_positions()
            if self.force_only:
                print("force")
                coeficients = np.random.default_rng().normal(loc=0.0, scale=self.learning_rate,
                                                             size=(1)).reshape((1, 1, 1))
                gradient_mutation_amount = coeficients * all_gradients_normalised[0]
            else:
                coeficients = np.random.default_rng().normal(loc=0.0, scale=self.learning_rate,
                                                             size=(3)).reshape((3, 1, 1))
                gradient_mutation_amount = coeficients * all_gradients_normalised
            gradient_mutation_amount = gradient_mutation_amount.sum(axis=0)
            pos += gradient_mutation_amount
            atoms.set_positions(pos)
            mutant = atoms
        else:
            print("materials")
            N = len(atoms)
            slab = atoms[:len(atoms) - N]
            atoms = atoms[-N:]
            tags = atoms.get_tags() if self.use_tags else np.arange(N)
            pos_ref = atoms.get_positions()
            num = atoms.get_atomic_numbers()
            cell = atoms.get_cell()
            pbc = atoms.get_pbc()
            count = 0
            maxcount = 1000
            too_close = True
            while too_close and count < maxcount:
                count += 1
                pos = pos_ref.copy()
                if self.force_only:
                    print("force")
                    coeficients = np.random.default_rng().normal(loc=0.0, scale=self.learning_rate,
                                                                 size=(1)).reshape((1, 1, 1))
                    gradient_mutation_amount = coeficients * all_gradients_normalised[0]
                else:
                    coeficients = np.random.default_rng().normal(loc=0.0, scale=self.learning_rate,
                                                                 size=(3)).reshape((3, 1, 1))
                    gradient_mutation_amount = coeficients * all_gradients_normalised
                gradient_mutation_amount = gradient_mutation_amount.sum(axis=0)

                pos += gradient_mutation_amount

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
