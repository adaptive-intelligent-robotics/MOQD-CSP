import pathlib
import pickle
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from ase import Atoms
from chgnet.model import CHGNet
from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor

from csp_elites.map_elites.archive import Archive
from csp_elites.utils.asign_target_values_to_centroids import (
    reassign_mo_data_from_pkl_to_new_centroids,
)
from csp_elites.utils.plot import load_mo_archive_from_pickle, load_centroids
from csp_elites.utils.utils import normalise_between_0_and_1


class MOArchive(Archive):
    def __init__(
        self,
        energies: np.ndarray,
        magmoms: np.ndarray,
        centroids: np.ndarray,
        descriptors: np.ndarray,
        individuals: List[Atoms],
        centroid_ids: Optional[np.ndarray],
        labels: Optional[List[str]] = None,
    ):
        """This class is only used for archive analysis after optimisation is completed."""
        self.energies = energies
        self.magmoms = magmoms
        self.centroids = centroids
        self.descriptors = descriptors
        self.individuals = individuals
        self.centroid_ids = centroid_ids
        self.labels = labels

    @classmethod
    def from_archive(
        cls,
        archive_path: pathlib.Path,
        centroid_filepath: Optional[pathlib.Path] = None,
    ):
        energies, magmoms, centroids, descriptors, individuals = load_mo_archive_from_pickle(
            archive_path
        )
        return cls(
            energies=np.array(energies),
            magmoms=np.array(magmoms),
            centroids=np.array(centroids),
            descriptors=np.array(descriptors),
            individuals=[Atoms.fromdict(individual) for individual in individuals],
            centroid_ids=cls.assign_centroid_ids(centroids, centroid_filepath)
            if centroid_filepath is not None
            else None,
        )

    # @classmethod
    # def from_relaxed_archive(
    #     cls,
    #     relaxed_archive_path: pathlib.Path,
    #     centroid_filepath: Optional[pathlib.Path] = None,
    # ):
    #     with open(relaxed_archive_path, "rb") as file:
    #         archive = pickle.load(file)

    #     centroids = np.array([archive[1][i] for i in range(len(archive[1]))])
    #     return cls(
    #         fitnesses=np.array([archive[0][i] for i in range(len(archive[0]))]),
    #         centroids=centroids,
    #         descriptors=np.array([archive[2][i] for i in range(len(archive[2]))]),
    #         individuals=[Atoms.fromdict(archive[3][i]) for i in range(len(archive[3]))],
    #         centroid_ids=cls.assign_centroid_ids(centroids, centroid_filepath)
    #         if centroid_filepath is not None
    #         else None,
    #     )

    def create_mo_archive_dict(
        self,
        all_centroids: np.ndarray
    ):
        mo_archive_dict = {}
        
        energies_for_plotting = np.full((len(all_centroids)), -np.inf)
        magmoms_for_plotting = np.full((len(all_centroids)), -np.inf)
        descriptors_for_plotting = np.full(
            (len(all_centroids), len(self.descriptors[0])), -np.inf
        )
        labels_for_plotting = np.full((len(all_centroids)), -np.inf)
        labels_for_plotting = labels_for_plotting.astype(str)

        for i in range(len(self.centroids)):
            present_centroid = np.argwhere(all_centroids == self.centroids[i])[0][0]
            individual = {
                "centroid_index": present_centroid,
                "centroid": self.centroids[i],
                "energy": self.energies[i],
                "magmom": self.magmoms[i],
                "descriptor": self.descriptors[i],
                "individual": self.individuals[i],
                "label": self.labels[i] if self.labels is not None else "None"
            }
            if present_centroid in mo_archive_dict:
                mo_archive_dict[present_centroid].append(individual)
            else:
                mo_archive_dict[present_centroid] = [individual]
                
        return mo_archive_dict
            

    def convert_fitness_and_descriptors_to_plotting_format(
        self, all_centroids: np.ndarray
    ):
        energies_for_plotting = np.full((len(all_centroids)), -np.inf)
        magmoms_for_plotting = np.full((len(all_centroids)), -np.inf)
        descriptors_for_plotting = np.full(
            (len(all_centroids), len(self.descriptors[0])), -np.inf
        )
        labels_for_plotting = np.full((len(all_centroids)), -np.inf)
        labels_for_plotting = labels_for_plotting.astype(str)

        for i in range(len(self.centroids)):
            present_centroid = np.argwhere(all_centroids == self.centroids[i])
            energies_for_plotting[present_centroid[0][0]] = self.energies[i]
            magmoms_for_plotting[present_centroid[0][0]] = self.magmoms[i]
            descriptors_for_plotting[present_centroid[0][0]] = self.descriptors[i]
            if self.labels is not None:
                labels_for_plotting[present_centroid[0][0]] = str(self.labels[i])

        return energies_for_plotting, magmoms_for_plotting, descriptors_for_plotting, labels_for_plotting

    # def compute_chgnet_metrics_on_archive(self):
    #     model = CHGNet.load()
    #     predictions = model.predict_structure(
    #         [AseAtomsAdaptor.get_structure(atoms) for atoms in self.individuals],
    #         batch_size=10,
    #     )
    #     forces = [prediction["f"] for prediction in predictions]
    #     energies = [prediction["e"] for prediction in predictions]
    #     stresses = [prediction["s"] for prediction in predictions]
    #     del model
    #     return np.array(forces), np.array(energies), np.array(stresses)

    # def get_individuals_as_structures(self):
    #     return [AseAtomsAdaptor.get_structure(atoms) for atoms in self.individuals]

    @classmethod
    def create_reference_archive(
        cls,
        target_data_path: pathlib.Path,
        normalise_bd_values: List[Tuple[float, float]],
        centroids_to_assign_file: pathlib.Path,
        labels: Optional[List[str]],
    ):
        energies, magmoms, centroids, descriptors, individuals = load_mo_archive_from_pickle(
            target_data_path
        )

        if normalise_bd_values is not None:
            descriptors[:, 0] = normalise_between_0_and_1(
                descriptors[:, 0],
                (normalise_bd_values[0][0], normalise_bd_values[1][0]),
            )
            descriptors[:, 1] = normalise_between_0_and_1(
                descriptors[:, 1],
                (normalise_bd_values[0][1], normalise_bd_values[1][1]),
            )

        centroids = reassign_mo_data_from_pkl_to_new_centroids(
            centroids_to_assign_file=centroids_to_assign_file,
            target_data=(energies, magmoms, centroids, descriptors, individuals),
            filter_for_number_of_atoms=None,
            normalise_bd_values=normalise_bd_values,
        )

        centroid_ids = cls.assign_centroid_ids(centroids, centroids_to_assign_file)
        return cls(
            energies=np.array(energies),
            magmoms=np.array(magmoms),
            centroids=np.array(centroids),
            descriptors=np.array(descriptors),
            individuals=[Atoms.fromdict(individual) for individual in individuals],
            centroid_ids=centroid_ids,
            labels=labels,
        )

    @classmethod
    def from_reference_csv_path(
        cls,
        target_data_path: pathlib.Path,
        normalise_bd_values: List[Tuple[float, float]],
        centroids_path: pathlib.Path,
    ):
        reference_data = pd.read_csv(target_data_path)
        reference_data.index = reference_data["Unnamed: 0"].to_list()
        reference_data.drop(columns="Unnamed: 0", inplace=True)
        energy, magmom, band_gap, shear_modulus, fmax, _ = reference_data.to_numpy()
        labels = list(reference_data.columns)
        if normalise_bd_values is not None:
            band_gap = normalise_between_0_and_1(
                band_gap, (normalise_bd_values[0][0], normalise_bd_values[1][0])
            )
            shear_modulus = normalise_between_0_and_1(
                shear_modulus, (normalise_bd_values[0][1], normalise_bd_values[1][1])
            )

        descriptors = np.vstack([band_gap, shear_modulus]).T
        centroids = reassign_mo_data_from_pkl_to_new_centroids(
            centroids_file=centroids_path,
            target_data=(energy, magmom, None, descriptors, None),
            filter_for_number_of_atoms=None,
            normalise_bd_values=None,
        )

        with MPRester(api_key="4nB757V2Puue49BqPnP3bjRPksr4J9y0") as mpr:
            data_mp_api_data = mpr.materials.search(
                material_ids=labels, fields=["structure"]
            )
        individuals = [
            AseAtomsAdaptor.get_atoms(el.structure) for el in data_mp_api_data
        ]

        return cls(
            energies=energy,
            magmoms=magmom,
            centroids=centroids,
            descriptors=descriptors,
            labels=labels,
            individuals=individuals,
            centroid_ids=cls.assign_centroid_ids(
                centroids_in_archive=centroids, centroid_filepath=centroids_path
            ),
        )

    def to_dataframe(self):
        df = pd.DataFrame(
            [
                self.labels,
                self.energies,
                self.magmoms,
                self.descriptors[:, 0],
                self.descriptors[:, 1],
                [None] * len(self.energies),
                self.centroid_ids,
            ]
        )
        df.columns = df.iloc[0]
        df = df[1:]
        df = df.reset_index(drop=True)
        df.index = ["energy", "magmom", "band_gap", "shear_modulus", "fmax", "centroid_id"]
        return df

    # def compute_qd_score(self, top_value: Optional[int] = None):
    #     valid_solutions_mask = self.get_valid_solutions_mask(top_value)
    #     valid_fitnesses = self.fitnesses * valid_solutions_mask
    #     return np.sum(valid_fitnesses)

    # def compute_fitness_metrics(self, top_value: Optional[int] = None):
    #     valid_solutions_mask = self.get_valid_solutions_mask(top_value)
    #     valid_fitnesses = self.fitnesses * valid_solutions_mask

    #     return (
    #         np.max(valid_fitnesses),
    #         np.mean(valid_fitnesses),
    #         np.median(valid_fitnesses),
    #         np.percentile(valid_fitnesses, 5),
    #         np.percentile(valid_fitnesses, 95),
    #     )

    # def compute_coverage(
    #     self,
    #     number_of_niches: int = 200,
    #     top_value: Optional[int] = None,
    #     filter_valid_solutions: bool = False,
    # ):
    #     if filter_valid_solutions:
    #         valid_solutions_mask = self.get_valid_solutions_mask(top_value)
    #     else:
    #         valid_solutions_mask = np.ones(len(self.fitnesses), dtype=bool)
    #     return 100 * len(self.fitnesses[valid_solutions_mask]) / number_of_niches

    # def get_valid_solutions_mask(self, top_value: Optional[int] = None):
    #     valid_solutions_mask = self.fitnesses >= 0
    #     if top_value is not None:
    #         valid_solutions_mask *= np.array(self.fitnesses < top_value, dtype=bool)
    #     return valid_solutions_mask
