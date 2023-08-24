import copy
import pickle
import time

import numpy as np
from chgnet.model import CHGNet
from matplotlib import pyplot as plt
from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor

# from csp_elites.parallel_relaxation.structure_to_use import atoms_to_test

if __name__ == '__main__':

    with MPRester(api_key="4nB757V2Puue49BqPnP3bjRPksr4J9y0") as mpr:
        one_structure = mpr.get_structure_by_material_id("mp-1341203", final=True)

    atoms_for_ref = AseAtomsAdaptor.get_atoms(one_structure)

    atoms_for_ref.rattle(0.1)
    atoms_to_test = copy.deepcopy(atoms_for_ref)

    # model = CHGNet.load().to("cuda")
    model = CHGNet.load()
    # batch_sizes_to_test = [0.05, 0.1, 0.2, 0.5]
    batch_sizes_to_test = [10, 20]
    n_individuals_to_test = [20, 50, 100]

    # batch_sizes_to_test = [0.25, 0.5]
    # n_individuals_to_test = [2, 4]

    structure = AseAtomsAdaptor.get_structure(copy.deepcopy(atoms_to_test))

    # with open("save_batch_timings.pkl", "rb") as file:
    #     all_timings = pickle.load(file)

    all_timings = []
    for batch_size_percent in batch_sizes_to_test:
        print(batch_size_percent)
        timings_per_batch_size = []
        for population_size in n_individuals_to_test:
            batch_size = batch_size_percent
            # batch_size = max(int(batch_size_percent * population_size), 1)
            structures = [copy.deepcopy(structure) for i in range(population_size)]
            tic = time.time()
            model.predict_structure(structures, batch_size=batch_size)
            time_taken = time.time() - tic
            timings_per_batch_size.append(time_taken)

        all_timings.append(timings_per_batch_size)

    # n_individuals_to_test = [10, 20, 50, 100, 200, 500]
    #
    fig, ax = plt.subplots()
    for i, batch_size_percent in enumerate(batch_sizes_to_test):
        ax.plot(n_individuals_to_test, all_timings[i], label=f"Batch size {batch_size_percent}")

    ax.plot(n_individuals_to_test, np.array(n_individuals_to_test) * all_timings[0][0], label="linear")
    ax.set_xlabel("Number of individuals tested")
    ax.set_ylabel("Time taken to evaluate fitness")
    ax.set_xticks(n_individuals_to_test)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncols=2)
    fig.tight_layout()
    # fig.legend()
    fig.savefig("experiments/chgnet_speed_batch_by_batch.png")

    all_timings_by_population_size = []
    for i, population_size in enumerate(n_individuals_to_test):
        population_timings = []
        for j in range(len(all_timings)):
            population_timings.append(all_timings[j][i])
        all_timings_by_population_size.append(population_timings)

    fig, ax = plt.subplots()
    for i, population_size in enumerate(n_individuals_to_test):
        ax.plot(batch_sizes_to_test, all_timings_by_population_size[i],
                label=f"Population size {population_size}")
    ax.set_xlabel("Batch size %")
    ax.set_ylabel("Time taken to evaluate fitness")
    ax.set_xticks(batch_sizes_to_test)
    # ax.set_aspect(0.005, adjustable='box')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncols=2)
    fig.tight_layout()
    # fig.legend()
    fig.savefig("experiments/chgnet_speed_batch_by_individuals.png")


    # with open("save_batch_timings_2.pkl", "wb") as file:
    #     pickle.dump(all_timings, file)

    #
    #
    # list_of_20 = [
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #      ]
    #
    # list_of_100 = [
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #         copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #     copy.deepcopy(structure),
    #      ]
    #
    # times = []
    #
    # # tic = time.time()
    # # model.predict_structure(copy.deepcopy(structure), batch_size=batch_size)
    # # times.append(time.time() - tic)
    # # print(f"time for 1 {time.time() - tic}")
    # #
    # # tic = time.time()
    # # model.predict_structure(list_of_20, batch_size=batch_size)
    # # times.append(time.time() - tic)
    # # print(f"time for 20 {time.time() - tic}")
    # #
    # # tic = time.time()
    # # model.predict_structure(list_of_100, batch_size=batch_size)
    # # times.append(time.time() - tic)
    # # print(f"time for 100 {time.time() - tic}")
    # #
    # # plt.title(f"Predict structure speed CHGNet batch size {batch_size}")
    # # plt.scatter([1, 20, 100], times)
    # # plt.show()
    #
    # # print("1 structure 20 times")
    # # tic = time.time()
    # # for i in range(20):
    # #     model.predict_structure(copy.deepcopy(structure))
    # #
    # # print(time.time() - tic)
    # #
    # # tic = time.time()
    # # model.predict_structure(copy.deepcopy(list_of_20), batch_size=20)
    # # times.append(time.time() - tic)
    # # print(f"time for batch size 20 {time.time() - tic}")
    # #
    #
    # tic = time.time()
    # model.predict_structure(copy.deepcopy(list_of_100), batch_size=20)
    # times.append(time.time() - tic)
    # print(f"time for batch size 20 {time.time() - tic}")
    #
    # tic = time.time()
    # model.predict_structure(copy.deepcopy(list_of_20), batch_size=5)
    # times.append(time.time() - tic)
    # print(f"time for batch size 5 {time.time() - tic}")
