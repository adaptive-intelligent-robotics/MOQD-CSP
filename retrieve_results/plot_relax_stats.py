import pathlib
import pickle

import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["savefig.dpi"] = 1000


if __name__ == "__main__":
    path_to_pickles = (
        pathlib.Path(__file__).parent.parent / ".experiment.nosync/report_data"
    )
    number_of_individuals = [1, 10, 20, 100]
    batch_size = [1, 10, 20, 100]

    with open(path_to_pickles / "bs_comp_ase_timings.pkl", "rb") as file:
        data = pickle.load(file)
        ase_batch_data = np.array(data)

    with open(path_to_pickles / "bs_comp_marta_timings.pkl", "rb") as file:
        data = pickle.load(file)
        marta_batch_data = np.array(data)

    with open(path_to_pickles / "relax_comp_ase_timings.pkl", "rb") as file:
        data = pickle.load(file)
        ase_relax_data = np.array(data)

    with open(path_to_pickles / "relax_comp_marta_timings.pkl", "rb") as file:
        data = pickle.load(file)
        marta_relax_data = np.array(data)

    # plot relax comp

    fig, ax = plt.subplots()

    ax.plot(number_of_individuals, ase_relax_data, label="Standard")
    ax.plot(number_of_individuals, marta_relax_data, label="This Work")
    ax.set_xlabel("Number of Individuals")
    ax.set_ylabel("Total Time Taken, s")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.4), ncols=4)
    fig.tight_layout()
    fig.savefig(path_to_pickles / "relax_time_comparison.png")

    fig, ax = plt.subplots()

    for i, individuals in enumerate(number_of_individuals):
        ax.plot(batch_size, ase_batch_data[i], label=individuals)
        # ax.plot(number_of_individuals, marta_batch_data[-1], label="MACS-Elites")

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Total Time Taken, s")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.4), ncols=4)
    fig.tight_layout()
    fig.savefig(path_to_pickles / "batch_size_comparison.png")

    print()
