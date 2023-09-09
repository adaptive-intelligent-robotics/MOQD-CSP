from __future__ import annotations

import copy
import pathlib
import pickle
import time

import torch.cuda
from chgnet.model import StructOptimizer, CHGNet
from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

from csp_elites.parallel_relaxation.structure_optimizer import BatchedStructureOptimizer

if __name__ == '__main__':
    with MPRester(api_key="4nB757V2Puue49BqPnP3bjRPksr4J9y0") as mpr:
        one_structure = mpr.get_structure_by_material_id("mp-1341203", final=True)
        # two_structure = mpr.get_structure_by_material_id("mp-1341203", final=True)

    atoms_for_ref = AseAtomsAdaptor.get_atoms(one_structure)

    atoms_for_ref.rattle(0.3)
    atoms_to_test = copy.deepcopy(atoms_for_ref)
    converted_structure = AseAtomsAdaptor.get_structure(atoms_for_ref)

    optimizer = BatchedStructureOptimizer()
    optimizer_ref = StructOptimizer()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer_ref.calculator.model = CHGNet.load().to(device)
    optimizer.model.to(device)
    n_relax_steps = 50

    number_of_individuals_list = [1, 10, 20, 100]

    times_ase = []
    print("ase relax")
    for number_of_individuals in tqdm(number_of_individuals_list):
        list_of_individuals = [copy.deepcopy(atoms_for_ref) for i in range(number_of_individuals)]
        tic = time.time()
        for el in list_of_individuals:
            relax_results = optimizer_ref.relax(copy.deepcopy(atoms_to_test), fmax=0.2, steps=n_relax_steps, verbose=False)
        times_ase.append(time.time() - tic)

    with open(pathlib.Path(__file__).parent / "experiments/benchmarking/relax_comparison_ase_timings.pkl", "wb") as file:
        pickle.dump(times_ase, file)


    times_marta = []
    print("QD4CSP relax")
    for number_of_individuals in tqdm(number_of_individuals_list):
        list_of_individuals = [copy.deepcopy(atoms_for_ref) for i in range(number_of_individuals)]
        tic = time.time()
        relax_results = optimizer.relax(list_of_individuals, n_relaxation_steps=n_relax_steps, verbose=False)
        times_marta.append(time.time() - tic)

    with open(pathlib.Path(__file__).parent / "experiments/benchmarking/relax_comp_qd4csp_timings.pkl", "wb") as file:
        pickle.dump(times_marta, file)


    batch_size = [1, 10, 20, 100]
    model = CHGNet.load().to(device)
    times_ase = []
    print("ASE batch")
    for number_of_individuals in tqdm(number_of_individuals_list):
        times_per_n_individuals = []
        for batch in batch_size:
            list_of_individuals = [copy.deepcopy(converted_structure) for i in range(number_of_individuals)]
            tic = time.time()
            relax_results = model.predict_structure(list_of_individuals, batch_size=batch)
            times_per_n_individuals.append(time.time() - tic)
        times_ase.append(times_per_n_individuals)

    with open(pathlib.Path(__file__).parent / "experiments/benchmarking/bs_comp_qd4csp_timings.pkl", "wb") as file:
        pickle.dump(times_ase, file)

    times_marta = []
    print("QD4CSP batch")
    for number_of_individuals in tqdm(number_of_individuals_list):
        times_per_n_individuals = []
        for batch in batch_size:
            list_of_individuals = [copy.deepcopy(atoms_for_ref) for i in range(number_of_individuals)]
            tic = time.time()
            optimizer.batch_size = batch
            relax_results = optimizer.relax(list_of_individuals, n_relaxation_steps=0)
            times_per_n_individuals.append(time.time() - tic)
        times_marta.append(times_per_n_individuals)

    with open(pathlib.Path(__file__).parent / "experiments/benchmarking/batch_size_comp_qd4csp_timings.pkl", "wb") as file:
        pickle.dump(times_marta, file)
