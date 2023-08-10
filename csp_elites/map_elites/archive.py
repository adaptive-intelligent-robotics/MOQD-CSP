import pathlib
import pickle
from typing import List, Optional

import numpy as np
from ase import Atoms

from csp_elites.utils.plot import load_archive_from_pickle, load_centroids


class Archive:
    def __init__(
        self,
        fitnesses: np.ndarray,
        centroids: np.ndarray,
        descriptors: np.ndarray,
        individuals: List[Atoms],
        centroid_ids: Optional[np.ndarray]
    ):
        self.fitnesses = fitnesses
        self.centroids = centroids
        self.descriptors = descriptors
        self.individuals = individuals
        self.centroid_ids = centroid_ids

    @classmethod
    def from_archive(cls, archive_path: pathlib.Path, centroid_filepath: Optional[pathlib.Path] = None):
        fitnesses, centroids, descriptors, individuals = load_archive_from_pickle(archive_path)
        return cls(
            fitnesses=np.array(fitnesses),
            centroids=np.array(centroids),
            descriptors=np.array(descriptors),
            individuals=[Atoms.fromdict(individual) for individual in individuals],
            centroid_ids=cls.assign_centroid_ids(centroids,
                                                 centroid_filepath) if centroid_filepath is not None else None,
        )

    @classmethod
    def from_relaxed_archive(cls, relaxed_archive_path: pathlib.Path, centroid_filepath: Optional[pathlib.Path] = None):
        with open(relaxed_archive_path, "rb") as file:
            archive = pickle.load(file)

        centroids = np.array([archive[1][i] for i in range(len(archive[1]))])
        return cls(
            fitnesses=np.array([archive[0][i] for i in range(len(archive[0]))]),
            centroids=centroids,
            descriptors=np.array([archive[2][i] for i in range(len(archive[2]))]),
            individuals=[Atoms.fromdict(archive[3][i]) for i in range(len(archive[3]))],
            centroid_ids=cls.assign_centroid_ids(centroids, centroid_filepath) if centroid_filepath is not None else None,
        )

    @staticmethod
    def assign_centroid_ids(centroids_in_archive: np.ndarray, centroid_filepath: pathlib.Path):
        """Update centroid_id attribute"""
        centroids = load_centroids(centroid_filepath)
        centroid_ids = []
        for el in centroids_in_archive:
            centroid_id = np.argwhere(centroids == el)[0][0]
            centroid_ids.append(centroid_id)

        return centroid_ids
