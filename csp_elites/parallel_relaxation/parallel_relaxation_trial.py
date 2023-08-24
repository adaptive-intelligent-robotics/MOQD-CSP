from __future__ import annotations

import copy
import time
from typing import List

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize import FIRE

from chgnet.model import CHGNet
from matplotlib import pyplot as plt
from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

from csp_elites.parallel_relaxation.fire import OverridenFire


def set_calculators_for_atoms_list(
    atoms: List[Atoms],
    forces: np.ndarray,
    energies: np.ndarray,
    stresses: np.ndarray
):
    for i in range(len(atoms)):
        calculator = SinglePointCalculator(
            atoms, energy=energies[i], forces=forces[i], stress=stresses[i],
        )
        atoms[i].calc = calculator
    return atoms

def evaluate_list_of_atoms(list_of_atoms: List[Atoms], chgnet):
    # ase_adapter = AseAtomsAdaptor
    tic = time.time()
    list_of_structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in list_of_atoms]
    print(f"convert structures: {time.time() - tic}")

    tic = time.time()
    predictions = chgnet.predict_structure(list_of_structures, batch_size=len(list_of_atoms))
    if isinstance(predictions, dict):
        predictions = [predictions]
    print(f"Get energy etc: {time.time() - tic}")
    # print(f"multiple evals {time.time() - tic}")

    return forces, energies, stresses

# code for timing
#     for i in range(len(list_of_atoms)):
#         r = list_of_atoms[i].get_positions()
#         list_of_atoms[i].set_positions(r + dr[i])
#
#     time_10_structures.append(time.time() - tac)
#     f, e, s = evaluate_list_of_atoms(list_of_atoms, chgnet)
#     # print(f" 1 step fire overriden {time.time() - tac}")
#
#     for i in range(len(list_of_atoms)):
#         print(sum(np.isclose(atoms.get_positions(), list_of_atoms[i].get_positions())))
#     # print(f"atoms fire difference {np.sum(atoms.get_positions() - atoms_3.get_positions())}")
#     # for
#     # print(
#     #     f"atoms overriden fire difference {np.sum(atoms_2.get_positions() - atoms_3.get_positions())}")
# time_1_structure = len(list_of_atoms) * np.array(time_1_structure)
# time_10_structures = np.array(time_10_structures)
# plt.plot(range(number_steps), time_1_structure, label="1 structure at a time")
# plt.plot(range(number_steps), time_10_structures, label=f"{len(list_of_atoms)} structures at a time")
# plt.legend()
# plt.show()
#
# print(f"average time 1 structure at a time {time_1_structure.mean()}, std {time_1_structure.std()}")
# print(
#     f"average time 10 structures at a time {time_10_structures.mean()}, std {time_10_structures.std()}")
# # print(f"atoms fire difference {np.sum(atoms.get_positions() - atoms_3.get_positions())}")
# # print(
# #     f"atoms overriden fire difference {np.sum(atoms_2.get_positions() - atoms_3.get_positions())}")


if __name__ == '__main__':
    chgnet = CHGNet.load()
    with MPRester(api_key="4nB757V2Puue49BqPnP3bjRPksr4J9y0") as mpr:
        one_structure = mpr.get_structure_by_material_id("mp-1341203", final=True)

    tic = time.time()
    prediction = chgnet.predict_structure(one_structure)
    # print(f"1 prediction {time.time() - tic}")

    atoms = AseAtomsAdaptor.get_atoms(structure=one_structure)
    atoms_2 = AseAtomsAdaptor.get_atoms(structure=one_structure)
    atoms_3 = AseAtomsAdaptor.get_atoms(structure=one_structure)

    calc = SinglePointCalculator(atoms, energy=prediction["e"], forces=prediction["f"],
                                 stress=prediction["s"])
    atoms.calc = calc
    atoms_2.calc = calc
    atoms_2.rattle(0.1)
    print(f"atoms 2 parameters {atoms_2.get_cell_lengths_and_angles()}")

    list_of_atoms = [copy.deepcopy(atoms_2),
                     copy.deepcopy(atoms_2),
                     copy.deepcopy(atoms_2),
                     copy.deepcopy(atoms_2),
                     ]

   #  # list_of_atoms = 10 * list_of_atoms
    forces, energies, stresses = evaluate_list_of_atoms(list_of_atoms, chgnet)
    list_of_atoms = set_calculators_for_atoms_list(list_of_atoms, forces, energies, stresses)

    fire = FIRE(atoms)
    overriden_fire = OverridenFire(atoms_2)

   # intialise all parameters that normally are attributes
    system_size = (len(list_of_atoms), (len(list_of_atoms[0])))
    f = forces
    e = energies
    v = None
    e_last = None
    r_last = None
    v_last = None
    dt = 0.1
    Nsteps = 0
    a = 0.1
    # relaxer = Relaxer()
    # relaxer.relax(steps=0)

    time_1_structure = []
    time_10_structures = []
    number_steps = 3
    for i in tqdm(range(number_steps)):
        # print(i)
        tic = time.time()
        fire.step(f=prediction["f"])

        time_1_structure.append(time.time() - tic)
        prediction = chgnet.predict_structure(AseAtomsAdaptor.get_structure(atoms))
        # print(f" 1 step fire {time.time() - tic}")
        # overriden_fire.step(atoms, f=prediction["f"])
        # overriden_fire.step(atoms_2, f=prediction["f"])
        tac = time.time()

        v, e_last, r_last, v_last, dt, Nsteps, a, dr = \
            overriden_fire.step_override(system_size, f, e, v, e_last, r_last, v_last, dt, Nsteps, a)
        forces, energies, stresses = evaluate_list_of_atoms(list_of_atoms, chgnet)
        # v, e_last, r_last, v_last, dt, Nsteps, a, dr = \
        #     overriden_fire.step_override(system_size, f, e, v, e_last, r_last, v_last, dt, Nsteps, a)


    print()
