from typing import List

import numpy as np
from ase import Atoms
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.utilities import atoms_too_close, atoms_too_close_two_sets

from csp_elites.map_elites.elites_utils import Species


class DQDMutationOMGMEGA(OffspringCreator):
    """This implementation is based on the Rattle mutation in ase.
    Rattle docstring below:

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

    def __init__(
        self,
        blmin,
        n_top,
        force_only: bool = False,
        simple: bool = True,
        learning_rate: float = 0.01,
        test_dist_to_slab=True,
        use_tags=False,
        rattle_prop=0.4,
        verbose=False,
        rng=np.random,
    ):
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.test_dist_to_slab = test_dist_to_slab
        self.use_tags = use_tags

        self.descriptor = "DQDMutation"
        self.min_inputs = 1
        self.learning_rate = learning_rate
        # self.rattle_prop = rattle_prop
        self.force_only = force_only
        self.simple = simple

    def get_new_individual(self, parents: List[Species]):
        f = parents[0]

        indi = self.mutate(f)
        if indi is None:
            return indi, "mutation: DQD rattle"

        indi = self.initialize_individual(f, indi)

        if "confid" in f.x["info"].keys():
            indi.info["data"]["parents"] = [f.x["info"]["confid"]]
        else:
            indi.info["data"]["parents"] = [None]

        return self.finalize_individual(indi), "mutation: DQD rattle"

    def mutate(self, species: Species):
        """Does the actual mutation."""
        atoms = Atoms.fromdict(species.x)
        gradient_stack = np.vstack(
            [
                [
                    species.fitness_gradient[: len(species.x["positions"])],
                    species.descriptor_gradients[0],
                    species.descriptor_gradients[1],
                ]
            ]
        )
        all_gradients_normalised = self.normalize_all_gradients_at_once(gradient_stack)

        if self.simple:
            print("simple")
            pos = atoms.get_positions()
            if self.force_only:
                print("force")
                coeficients = (
                    np.random.default_rng()
                    .normal(loc=0.0, scale=self.learning_rate, size=(1))
                    .reshape((1, 1, 1))
                )
                gradient_mutation_amount = coeficients * all_gradients_normalised[0]
            else:
                coeficients = (
                    np.random.default_rng()
                    .normal(loc=0.0, scale=self.learning_rate, size=(3))
                    .reshape((3, 1, 1))
                )
                gradient_mutation_amount = coeficients * all_gradients_normalised
            gradient_mutation_amount = gradient_mutation_amount.sum(axis=0)
            pos += gradient_mutation_amount
            atoms.set_positions(pos)
            mutant = atoms
        else:
            N = len(atoms)
            slab = atoms[: len(atoms) - N]
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
                    coeficients = (
                        np.random.default_rng()
                        .normal(loc=0.0, scale=self.learning_rate, size=(1))
                        .reshape((1, 1, 1))
                    )
                    gradient_mutation_amount = coeficients * all_gradients_normalised[0]
                else:
                    coeficients = (
                        np.random.default_rng()
                        .normal(loc=0.0, scale=self.learning_rate, size=(3))
                        .reshape((3, 1, 1))
                    )
                    gradient_mutation_amount = coeficients * all_gradients_normalised
                gradient_mutation_amount = gradient_mutation_amount.sum(axis=0)

                pos += gradient_mutation_amount

                top = Atoms(num, positions=pos, cell=cell, pbc=pbc, tags=tags)
                too_close = atoms_too_close(top, self.blmin, use_tags=self.use_tags)
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
