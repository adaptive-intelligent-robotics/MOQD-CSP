import pickle

import numpy as np
from ase.ga.ofp_comparator import OFPComparator

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# from torchvision import datasets, transforms

from csp_elites.crystal.crystal_evaluator import CrystalEvaluator
from csp_elites.utils.plot import load_archive, load_archive_from_pickle
import matplotlib.pyplot as plt
class CVTWithNNForIndSelection:
    def __init__(self):
        pass

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def save_relaxation_data_for_archive():
    archive_path = "../../experiments/20230707_22_04_TiO2_no_relaxation_20k_evals/archive_20000.pkl"
    fitnesses, centroids, descriptors, individuals = load_archive_from_pickle(archive_path)

    file_save_path = "../../experiments/nn_relaxation_results/20230707_22_04_TiO2_no_relaxation_20k_evals_20000.pkl"

    blocks = [22] * 8 + [8] * 16

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    comparator = OFPComparator(n_top=len(blocks), dE=1.0,
                               cos_dist_max=1e-3, rcut=10., binwidth=0.05,
                               pbc=[True, True, True], sigma=0.05, nsigma=4,
                               recalculate=False)

    crystal_evaluator = CrystalEvaluator(comparator=comparator)
    relax_results_list = []
    for i in tqdm(range(len(individuals))):
        energy, relaxation_results = crystal_evaluator.compute_energy(individuals[i],
                                                                      really_relax=True,
                                                                      n_steps=500)
        relax_results_json = (
            relaxation_results["final_structure"].as_dict(),
            {
                "energy": relaxation_results["trajectory"].energies,
                "forces": relaxation_results["trajectory"].forces,
                "stresses": relaxation_results["trajectory"].stresses,
                "atom_positions": relaxation_results["trajectory"].atom_positions,
                "cell": relaxation_results["trajectory"].cells,
                "atomic_number": relaxation_results["trajectory"].atoms.get_atomic_numbers(),
            }

        )
        relax_results_list.append(relax_results_json)
    #     plt.plot(range(len(relaxation_results["trajectory"].energies)), np.array(relaxation_results["trajectory"].energies) / len(blocks), label=str(i))
    #
    # plt.legend()
    # plt.show()
    print()
    with open(file_save_path, "wb") as file:
        pickle.dump(relax_results_list, file)

    pass


if __name__ == '__main__':
    file_save_path = "../../experiments/nn_relaxation_results/20230707_22_04_TiO2_no_relaxation_20k_evals_20000.pkl"
    blocks = [22] * 8 + [8] * 16
    threshold = 0.075
    n_steps = 10
    with open(file_save_path, "rb") as file:
        all_data = pickle.load(file)

    all_energies = [el[1]["energy"] for el in all_data]
    all_energies = np.array(all_energies).reshape(len(all_data), -1) / len(blocks)
    energy_differences = all_energies[:, :-1] - all_energies[:, 1:]

    energy_differences_under_threshold = energy_differences < threshold
