import pathlib
import pickle
from typing import List

import numpy as np
from ase import Atoms

from csp_elites.utils.plot import load_archive_from_pickle


class Archive:
    def __init__(
        self,
        fitnesses: np.ndarray,
        centroids: np.ndarray,
        descriptors: np.ndarray,
        individuals: List[Atoms],
    ):
        self.fitnesses = fitnesses
        self.centroids = centroids
        self.descriptors = descriptors
        self.individuals = individuals

    @classmethod
    def from_archive(cls, archive_path: pathlib.Path):
        fitnesses, centroids, descriptors, individuals = load_archive_from_pickle(archive_path)
        return cls(
            fitnesses=np.array(fitnesses),
            centroids=np.array(centroids),
            descriptors=np.array(descriptors),
            individuals=[Atoms.fromdict(individual) for individual in individuals]
        )

    @classmethod
    def from_relaxed_archive(cls, relaxed_archive_path: pathlib.Path):
        with open(relaxed_archive_path, "rb") as file:
            archive = pickle.load(file)

        return cls(
            fitnesses=np.array([archive[0][i] for i in range(len(archive[0]))]),
            centroids=np.array([archive[1][i] for i in range(len(archive[1]))]),
            descriptors=np.array([archive[2][i] for i in range(len(archive[2]))]),
            individuals=[Atoms.fromdict(archive[3][i]) for i in range(len(archive[3]))]
        )
