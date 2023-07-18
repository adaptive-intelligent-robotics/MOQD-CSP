import copy
import pickle

import numpy as np
from ase.ga.ofp_comparator import OFPComparator
from tqdm import tqdm

from csp_elites.crystal.crystal_evaluator import CrystalEvaluator
from csp_elites.utils.get_mpi_structures import get_all_materials_with_formula

if __name__ == '__main__':
    formula = "TiO2"
    docs, atoms = get_all_materials_with_formula(formula)

    unrelaxed_atoms = copy.deepcopy(atoms)
    relaxed_atoms = copy.deepcopy(atoms)
    relaxed_10_steps = copy.deepcopy(atoms)

    comparator = OFPComparator(n_top=48, dE=1.0,
                               cos_dist_max=1e-3, rcut=10., binwidth=0.05,
                               pbc=[True, True, True], sigma=0.05, nsigma=4,
                               recalculate=False)

    unrelaxed_energies = []
    relaxed_energies = []

    crystal_evaluator = CrystalEvaluator(comparator)

    for i in tqdm(range(len(unrelaxed_atoms))):
        relaxed_energy, _ = crystal_evaluator.compute_energy(relaxed_atoms[i], really_relax=True)
        unrelaxed_energy, _ = crystal_evaluator.compute_energy(unrelaxed_atoms[i], really_relax=False)

        relaxed_energies.append(relaxed_energy)
        unrelaxed_energies.append(unrelaxed_energy)
    delta = np.array(relaxed_energies) - np.array(unrelaxed_energies)
    print(delta)
    print()

    with open("relaxed_vs_unrelaxed_tio2_data.pkl", "wb") as file:
        pickle.dump([relaxed_energies, unrelaxed_energies, docs, atoms])
