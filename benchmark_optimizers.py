from __future__ import annotations

import copy
import pickle
import time

from chgnet.model import StructOptimizer, CHGNet
from matplotlib import pyplot as plt
from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor

from csp_elites.parallel_relaxation.structure_optimizer import MultiprocessOptimizer
from csp_elites.parallel_relaxation.structure_to_use import atoms_to_test


if __name__ == '__main__':
    chgnet = CHGNet.load()
    with MPRester(api_key="4nB757V2Puue49BqPnP3bjRPksr4J9y0") as mpr:
        one_structure = mpr.get_structure_by_material_id("mp-1341203", final=True)
        # two_structure = mpr.get_structure_by_material_id("mp-1341203", final=True)

    atoms_for_ref = AseAtomsAdaptor.get_atoms(one_structure)

    atoms_for_ref.rattle(0.1)
    atoms_to_test = copy.deepcopy(atoms_for_ref)

    optimizer = MultiprocessOptimizer(CHGNet.load())
    optimizer_ref = StructOptimizer()

    n_relax_steps = 10
    times_ase = []
    tic = time.time()
    times_ase.append(time.time() - tic)

    relax_results = optimizer_ref.relax(copy.deepcopy(atoms_to_test), fmax=0.2, steps=n_relax_steps)
    list_of_10 = [
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),

         ]

    list_of_10_marta = copy.deepcopy(list_of_10)
    tic = time.time()
    for el in list_of_10:
        relax_results = optimizer_ref.relax(el, fmax=0.2, steps=n_relax_steps)
    times_ase.append(time.time() - tic)

    list_of_20 = [
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
         ]

    list_of_20_marta = copy.deepcopy(list_of_20)

    tic = time.time()
    for el in list_of_20:
        relax_results = optimizer_ref.relax(el, fmax=0.2, steps=n_relax_steps)
    times_ase.append(time.time() - tic)

    list_of_100 = [
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
            copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
        copy.deepcopy(atoms_to_test),
         ]

    list_of_100_marta = copy.deepcopy(list_of_100)

    tic = time.time()
    for el in list_of_100:
        relax_results = optimizer_ref.relax(el, fmax=0.2, steps=n_relax_steps)
    times_ase.append(time.time() - tic)


    times_marta = []
    print("1 ind")
    tic = time.time()
    relax_results_2, fmax_over_time = optimizer.relax(
        [
            copy.deepcopy(atoms_to_test),
         ],
        n_relaxation_steps=n_relax_steps,
    )
    times_marta.append(time.time() - tic)

    print("10 ind")

    tic = time.time()
    relax_results_2, fmax_over_time = optimizer.relax(
        list_of_10_marta,
        n_relaxation_steps=n_relax_steps,
    )
    times_marta.append(time.time() - tic)

    print("20 ind")

    tic = time.time()
    relax_results_2, fmax_over_time = optimizer.relax(
        list_of_20_marta,
        n_relaxation_steps=n_relax_steps,
    )
    times_marta.append(time.time() - tic)


    print("100 ind")

    tic = time.time()

    relax_results_2, fmax_over_time = optimizer.relax(
        list_of_100_marta,
        n_relaxation_steps=n_relax_steps,
    )
    times_marta.append(time.time() - tic)

    with open("csp_investigations/parallel_relaxation/ase_fire_data_5_steps_1_10_20_100.pkl", "wb") as file:
        pickle.dump(times_ase, file)


    print(times_marta)
    print(times_ase)
    plt.title(f"N relax step {n_relax_steps}")
    plt.scatter([1, 10, 20, 100], times_marta, label="marta")
    plt.scatter([1, 10, 20, 100], times_ase, label="ase")
    plt.legend()
    plt.show()
