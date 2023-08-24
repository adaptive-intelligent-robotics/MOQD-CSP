from typing import List

import numpy as np
from ase import Atoms
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.utilities import atoms_too_close, atoms_too_close_two_sets, closest_distances_generator
from chgnet.model import CHGNet, CHGNetCalculator
from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor

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


class DQDMutation(OffspringCreator):
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
        forces = self.normalize_gradient(species.fitness_gradient)
        descriptor_gradients = np.array([self.normalize_gradient(grad) for grad in species.descriptor_gradients])

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
            coeficients = np.random.default_rng().normal(loc=0.0, scale=self.learning_rate, size=(3))

            gradient_mutation_amount = \
                coeficients[0] * forces + \
                coeficients[1] * descriptor_gradients[0] + \
                coeficients[2] * descriptor_gradients[1]

            ok = False
            # too_close = False
            # pos += coeficients[0] * forces
            for tag in np.unique(tags):
                select = np.where(tags == tag)
                if self.rng.rand() < self.rattle_prop:
                    ok = True
                    # r = self.rng.rand(3)
                    pos[select] += coeficients[0] * gradient_mutation_amount[select]

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

    def normalize_gradient(self, gradient: np.ndarray):
        return (gradient- np.min(gradient)) / (np.max(gradient) - np.min(gradient))



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

    gradient_mutator = DQDMutation(blmin=closest_distances, n_top=len(one_structure.atomic_numbers))

    mutant = gradient_mutator.mutate(atoms_2)

    print((target_atoms.get_positions() - mutant.get_positions()).sum() - initial_distance)
    a = 2
    print()
