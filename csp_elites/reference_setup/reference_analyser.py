import copy
import json
import pathlib
import pickle
from collections import defaultdict
from typing import Optional, Tuple, List

import matplotlib as mpl
import numpy as np
import pandas as pd
from ase import Atoms
from ase.spacegroup import get_spacegroup
from matplotlib import pyplot as plt
from pymatgen.io.ase import AseAtomsAdaptor
from pyxtal import pyxtal
from pyxtal.msg import Comp_CompatibilityError
from sklearn.neighbors import KDTree

from csp_elites.crystal.mo_crystal_evaluator import MOCrystalEvaluator
from csp_elites.crystal.materials_data_model import MaterialProperties
from csp_elites.evaluation.symmetry_evaluator import SymmetryEvaluation
from csp_elites.map_elites.archive import Archive
from csp_elites.map_elites.elites_utils import (
    __centroids_filename as get_centroids_filename,
)
from csp_elites.map_elites.elites_utils import cvt, write_centroids
from csp_elites.mome.mome_archive import MOArchive
from csp_elites.reference_setup.reference_plotter import ReferencePlotter
from csp_elites.utils.asign_target_values_to_centroids import (
    reassign_data_from_pkl_to_new_centroids,
)
from csp_elites.utils.plot import load_centroids, plot_2d_map_elites_repertoire_marta


plt.rcParams["savefig.dpi"] = 300


class ReferenceAnalyser:
    def __init__(
        self,
        formula: str,
        max_n_atoms_in_cell: int,
        experimental_references_only: bool,
        save_plots: bool = True,
        normalise_bd: bool = True,
    ):
        self.formula = formula
        self.max_n_atoms_in_cell = max_n_atoms_in_cell
        self.experimental_references_only = experimental_references_only
        self.experimental_string = (
            "experimental_only" if experimental_references_only else "exp_and_theory"
        )

        self.main_experiments_directory = (
            pathlib.Path(__file__).parent.parent.parent
            / "reference_data"
        )

        centroid_folder_path = self.main_experiments_directory / "centroids"
        centroid_folder_path.mkdir(exist_ok=True)
        self.centroid_folder_path = centroid_folder_path
        
        print("Centroid folder path: ", self.centroid_folder_path)

        self.symmetry_evaluator = SymmetryEvaluation(
            formula=formula,
            tolerance=0.1,
            maximum_number_of_atoms_in_reference=max_n_atoms_in_cell,
            number_of_atoms_in_system=max_n_atoms_in_cell,
            filter_for_experimental_structures=experimental_references_only,
        )
        self.reference_structures = sorted(
            self.symmetry_evaluator.known_structures_docs,
            key=lambda x: (x.theoretical, len(x.structure)),
        )
        self.crystal_evaluator = MOCrystalEvaluator(
            with_force_threshold=False,
            relax_every_n_generations=0,
            fmax_relaxation_convergence=0.2,
            force_threshold_fmax=1.0,
            compute_gradients=True,
            bd_normalisation=None,
        )
                    
        self.reference_ids = [
            str(structure.material_id) for structure in self.reference_structures
        ]
        self.behavioural_descriptors = [
            MaterialProperties.BAND_GAP,
            MaterialProperties.SHEAR_MODULUS,
        ]

        (
            self.energies,
            self.magmoms,
            self.fmax_list,
            self.band_gaps,
            self.shear_moduli,
        ) = self.compute_target_values()
        self.centroid_filename = None
        self.plotting_helper = ReferencePlotter(save_plots)
        self.save_plot = save_plots
        self.bd_minimum_values = None
        self.bd_maximum_values = None

        self.save_path = (
            self.main_experiments_directory.parent
            / "mp_reference_analysis"
            / f"{formula}_{max_n_atoms_in_cell}"
        )
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.normalise_bd = normalise_bd

    def compute_target_values(self):
        list_of_atoms_as_dict = [
            AseAtomsAdaptor.get_atoms(el.structure).todict()
            for el in self.reference_structures
        ]
        atom_counts_per_structure = np.array(
            [len(el.structure.atomic_numbers) for el in self.reference_structures]
        )
        unique_lengths = np.unique(atom_counts_per_structure)
        energies, magmoms, band_gaps, shear_moduli, forces, reference_ids_tracking = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for el in unique_lengths:
            indices_to_check = np.argwhere(atom_counts_per_structure == el).reshape(-1)
            reference_ids_tracking += list(indices_to_check)
            structures_to_check = [list_of_atoms_as_dict[el] for el in indices_to_check]
            (
                _,
                _,
                fitness_scores,
                descriptors,
                _,
                gradients,
            ) = self.crystal_evaluator.batch_compute_fitness_and_bd(
                structures_to_check, n_relaxation_steps=0
            )
            energies += list(np.array(fitness_scores)[:, 0])
            magmoms += list(np.array(fitness_scores)[:, 1])
            band_gaps += descriptors[0]
            shear_moduli += descriptors[1]
            forces += (el[0] for el in gradients)

        indices_to_sort = np.argsort(reference_ids_tracking)

        fmax_list = []
        for force in forces:
            fmax_list.append(np.max((force[:-3] ** 2).sum(axis=1) ** 0.5))

        fmax_list = np.take(fmax_list, indices_to_sort, axis=0)
        energies = np.take(energies, indices_to_sort, axis=0)
        magmoms = np.take(magmoms, indices_to_sort, axis=0)
        shear_moduli = np.take(shear_moduli, indices_to_sort, axis=0)
        band_gaps = np.take(band_gaps, indices_to_sort, axis=0)

        return (
            np.array(energies),
            np.array(magmoms),
            fmax_list,
            np.array(band_gaps),
            np.array(shear_moduli),
        )

    def set_bd_limits(self, band_gaps: np.ndarray, shear_moduli: np.ndarray):
        band_gap_limits = np.array([band_gaps.min(), band_gaps.max()], dtype=float)
        shear_moduli_limits = np.array([shear_moduli.min(), shear_moduli.max()], dtype=float)

        band_gap_min_max_diff = band_gap_limits[1] - band_gap_limits[0]
        shear_moduli_min_max_diff = shear_moduli_limits[1] - shear_moduli_limits[0]
        if not np.abs(shear_moduli_min_max_diff - band_gap_min_max_diff) < 0.2 * max(
            [shear_moduli_min_max_diff, band_gap_min_max_diff]
        ):
            print("\n")
            print(
                f"Recommend setting band gaps manually, bg limits: {band_gap_limits}, shear moduli limits {shear_moduli_limits}"
            )
        
        else:
            print("\n")
            print(f"Band gap limits: {band_gap_limits}, shear moduli limits {shear_moduli_limits}")

        self.bd_minimum_values = (band_gap_limits[0], shear_moduli_limits[0])
        self.bd_maximum_values = (band_gap_limits[1], shear_moduli_limits[1])
        return band_gap_limits.tolist(), shear_moduli_limits.tolist()

    def propose_fitness_limits(self):
        energies = np.array(self.energies)
        magmoms = np.array(self.magmoms)
        lower_limits = np.array(
            [np.floor(energies.min() * 0.9),
             np.floor(magmoms.min() * 0.9),]
        )
        upper_limits = np.array(
            [np.ceil(energies.max() * 1.1),
             np.ceil(magmoms.max() * 1.1)]
        )
        print("\n")
        print(f"lower fitness limits: {lower_limits}, upper fitness limits: {upper_limits}")
        return lower_limits.tolist(), upper_limits.tolist()

    def initialise_kdt_and_centroids(
        self,
        number_of_niches: int,
        band_gap_limits: Optional[np.ndarray] = None,
        shear_moduli_limits: Optional[np.ndarray] = None,
    ):
        # create the CVT
        if band_gap_limits is None or shear_moduli_limits is None:
            band_gap_limits, shear_moduli_limits = self.set_bd_limits(
                self.band_gaps, self.shear_moduli
            )

        if self.normalise_bd:
            bd_minimum_values, bd_maximum_values = [0, 0], [1, 1]
        else:
            bd_minimum_values = [band_gap_limits[0], shear_moduli_limits[0]]
            bd_maximum_values = [band_gap_limits[1], shear_moduli_limits[1]]
        c = cvt(
            number_of_niches,
            len(self.behavioural_descriptors),
            25000,
            bd_minimum_values,
            bd_maximum_values,
            self.centroid_folder_path,
            bd_names=self.behavioural_descriptors,
            cvt_use_cache=True,
            formula=self.formula,
            centroids_load_dir=self.centroid_folder_path,
            centroids_save_dir=self.centroid_folder_path,
        )    
        kdt = KDTree(c, leaf_size=30, metric="euclidean")
        write_centroids(
            c,
            experiment_folder=self.centroid_folder_path,
            bd_names=self.behavioural_descriptors,
            bd_minimum_values=bd_minimum_values,
            bd_maximum_values=bd_maximum_values,
            formula=self.formula,
        )
        del c
        self.centroid_filename = get_centroids_filename(
            k=number_of_niches,
            dim=len(self.behavioural_descriptors),
            bd_names=self.behavioural_descriptors,
            bd_minimum_values=bd_minimum_values,
            bd_maximum_values=bd_maximum_values,
            formula=self.formula,
        )
        return kdt

    def create_model_archive(
        self, bd_minimum_values, bd_maximum_values, save_reference=False
    ):
        if self.centroid_filename is None:
            print("first create centroids or provide path to centroid file")

        descriptors = np.array(
            [
                (self.band_gaps[i], self.shear_moduli[i])
                for i in range(len(self.band_gaps))
            ]
        )
        individuals = [
            AseAtomsAdaptor.get_atoms(el.structure)
            for el in self.reference_structures
        ]

        normalise_bd_values = (
            (bd_minimum_values, bd_maximum_values) if self.normalise_bd else None
        )

        # Find which centroid each reference structure belongs to
        centroids = reassign_data_from_pkl_to_new_centroids(
            centroids_file=f"{self.centroid_folder_path}{self.centroid_filename}",
            target_data=[self.energies, None, copy.deepcopy(descriptors), individuals],
            filter_for_number_of_atoms=None,
            normalise_bd_values=normalise_bd_values,
        )
        target_archive = MOArchive(
            energies=np.array(self.energies),
            magmoms=np.array(self.magmoms),
            centroids=centroids,
            descriptors=np.array(descriptors),
            individuals=individuals,
            centroid_ids=None,
            labels=self.reference_ids,
        )
        
        # Find centroid ids for each reference structure
        target_archive.centroid_ids = Archive.assign_centroid_ids(
            centroids, f"{self.centroid_folder_path}{self.centroid_filename}"
        )

        if save_reference:
            all_data = []
            for i in range(len(self.energies)):
                one_data_point = []
                one_data_point.append(self.energies[i])
                one_data_point.append(self.magmoms[i])
                one_data_point.append(centroids[i])
                one_data_point.append(
                    np.array([self.band_gaps[i], self.shear_moduli[i]])
                )
                one_data_point.append(individuals[i].todict())

                all_data.append(one_data_point)

            with open(
                self.save_path / f"{self.formula}_band_gap_shear_modulus.pkl", "wb"
            ) as file:
                pickle.dump(all_data, file)

            centroid_tag = str(self.centroid_filename.split("/")[1].rstrip(".dat"))
            filename = f"{self.formula}_target_data_{centroid_tag}.csv"
            df = pd.DataFrame(
                [
                    self.reference_ids,
                    self.energies,
                    self.magmoms,
                    descriptors[:, 0],
                    descriptors[:, 1],
                    self.fmax_list,
                    target_archive.centroid_ids,
                ]
            )
            df.columns = df.iloc[0]
            df = df[1:]
            df = df.reset_index(drop=True)
            df.index = ["energy", "magmom", "band_gap", "shear_modulus", "fmax", "centroid_id"]
            df.to_csv(self.save_path / filename)

        return target_archive

    def plot_cvt_plot(
        self,
        target_archive: Archive,
        bd_minimum_values: np.ndarray,
        bd_maximum_values: np.ndarray,
        fitness_limits: np.ndarray,
        x_axis_limits=None,
        y_axis_limits=None,
    ):
        plotting_centroids = load_centroids(
            self.centroid_folder_path / self.centroid_filename
        )
        (
            fitness_for_plotting,
            descriptors_for_plotting,
            labels_for_plotting,
        ) = target_archive.convert_fitness_and_descriptors_to_plotting_format(
            plotting_centroids
        )

        if self.save_plot:
            directory_string = self.save_path
        else:
            directory_string = None

        plot_2d_map_elites_repertoire_marta(
            centroids=plotting_centroids,
            repertoire_fitnesses=fitness_for_plotting,
            minval=bd_minimum_values,
            maxval=bd_maximum_values,
            repertoire_descriptors=descriptors_for_plotting,
            vmin=fitness_limits[0],
            vmax=fitness_limits[1],
            annotations=labels_for_plotting,
            directory_string=directory_string,
            filename=f"{self.formula}_cvt_plot_{self.experimental_string}_no_annotate",
            annotate=False,
            x_axis_limits=x_axis_limits,
            y_axis_limits=y_axis_limits,
        )
        plot_2d_map_elites_repertoire_marta(
            centroids=plotting_centroids,
            repertoire_fitnesses=fitness_for_plotting,
            minval=bd_minimum_values,
            maxval=bd_maximum_values,
            repertoire_descriptors=descriptors_for_plotting,
            vmin=fitness_limits[0],
            vmax=fitness_limits[1],
            annotations=labels_for_plotting,
            directory_string=directory_string,
            filename=f"{self.formula}_cvt_plot_{self.experimental_string}_annotate",
            annotate=True,
            x_axis_limits=x_axis_limits,
            y_axis_limits=y_axis_limits,
        )

        plt.clf()

    def plot_fmax(self, histogram_range: Optional[Tuple[int, int]] = None):
        params = {"figure.figsize": [3.5, 2.625]}
        mpl.rcParams.update(params)
        histogram_range = histogram_range if histogram_range is None else (0, 1)
        fig, ax = plt.subplots()
        ax.hist(self.fmax_list, range=histogram_range, bins=40)
        ax.set_xlabel("Maximum Force on an Atom")
        ax.set_ylabel("Structure Count")
        ax.set_title("Maximum Force on Atom for Reference Structures")
        if self.save_plot:
            fig.savefig(
                self.save_path
                / f"{self.formula}_fmax_histogram_no_stress{self.experimental_string}.png",
                format="png",
            )
        else:
            fig.show()
        plt.clf()

    def plot_symmetries(self):
        params = {"figure.figsize": [3.5, 2.625]}
        mpl.rcParams["font.size"] = 8
        mpl.rcParams.update(params)
        fig, ax = plt.subplots()

        experimental = [el for el in self.reference_structures if not el.theoretical]
        labels = ["Theoretical", "Experimental"]

        all_group_information = []
        for structure_group in [self.reference_structures, experimental]:
            symmetries = defaultdict(list)
            for el in structure_group:
                symmetry = get_spacegroup(
                    AseAtomsAdaptor.get_atoms(el.structure), symprec=0.1
                ).symbol
                symmetries[symmetry].append(str(el.material_id))
            all_group_information.append(symmetries)

        for i, symmetries_in_group in enumerate(all_group_information):
            ax.bar(
                list(symmetries_in_group.keys()),
                [len(value) for value in symmetries_in_group.values()],
                label=labels[i],
            )

        ax.set_ylabel("Structure Count")
        ax.set_xlabel("Symmetry")
        ax.set_title("Symmetries Across Reference Structures")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.legend(prop={"size": 6})
        # ax.legend(loc="lower center", bbox_to_anchor=(-0.02, 0.5), ncols=2)
        # if len(self.reference_structures) == 2:
        #     fig.legend(loc="center right")
        plt.tight_layout()
        if self.save_plot:
            fig.savefig(
                self.save_path
                / f"{self.formula}_symmetries_histogram_{self.experimental_string}.png",
                format="png",
            )
        else:
            fig.show()
        plt.clf()

    def _plot_histogram_of_distances(self, distances):
        params = {"figure.figsize": [3.5, 2.625]}
        mpl.rcParams.update(params)
        distances = np.array(distances)
        minimum_distances = [np.min(el[el > 0.001]) for el in distances]

        fig, ax = plt.subplots()
        ax.hist(minimum_distances, range=(0, 0.5), bins=20)
        ax.set_xlabel("Minimum Non-Zero Cosine Distance Between Reference Structures")
        ax.set_ylabel("Number of Structures")
        ax.set_title(
            "Minimum Non-Zero Cosine Distance Between Reference Structures Across References"
        )

        if self.save_plot:
            fig.savefig(
                self.save_path
                / f"{self.formula}_{self.max_n_atoms_in_cell}_distances_histogram_{self.experimental_string}.png",
                format="png",
            )
        else:
            fig.show()
        plt.clf()

    def heatmap_structure_matcher_distances(self, annotate: bool = True):
        if len(self.reference_structures) >= 15:
            params = {"figure.figsize": [5, 5], "font.size": 4}
        else:
            params = {"figure.figsize": [3.5, 3.5], "font.size": 8}
        mpl.rcParams.update(params)
        all_structure_matcher_matches = []
        distances = []

        for strucutre_1 in self.reference_structures:
            matches_for_structure_1 = []
            distances_for_structure_1 = []
            for structure_2 in self.reference_structures:
                matches_for_structure_1.append(
                    self.symmetry_evaluator.structure_matcher.fit(
                        strucutre_1.structure, structure_2.structure
                    )
                )
                if len(strucutre_1.structure) == len(structure_2.structure):
                    strucutre_1.structure.sort()
                    structure_2.structure.sort()
                    distance_to_known_structure = float(
                        self.symmetry_evaluator.comparator._compare_structure(
                            AseAtomsAdaptor.get_atoms(strucutre_1.structure),
                            AseAtomsAdaptor.get_atoms(structure_2.structure),
                        )
                    )
                else:
                    distance_to_known_structure = 1

                distances_for_structure_1.append(distance_to_known_structure)

            all_structure_matcher_matches.append(matches_for_structure_1)
            distances.append(distances_for_structure_1)

        im, cbar = self.plotting_helper.heatmap(
            np.array(distances),
            self.reference_ids,
            self.reference_ids,
            xlabel="Material Project Reference Numbers",
            ylabel="Material Project Reference Numbers",
            title="Cosine Distances Between Fingerprints of Reference Structures",
            cmap="YlGn",
            cbarlabel="Cosine Distance Between Structure Fingerprints",
        )
        if annotate:
            texts = self.plotting_helper.annotate_heatmap(im, valfmt="{x:.1f}")

        plt.tight_layout()
        if self.save_plot:
            plt.savefig(
                self.save_path
                / f"{self.formula}_distances_heatmap_no_annotate_{self.experimental_string}.png",
                format="png",
            )
        else:
            plt.show()
        plt.clf()
        self.plotting_helper.heatmap(
            np.array(all_structure_matcher_matches),
            self.reference_ids,
            self.reference_ids,
            xlabel="Material Project Reference Numbers",
            ylabel="Material Project Reference Numbers",
            title="StructureMatcher Confusion Matrix",
            cmap="YlGn",
            cbarlabel="Match between structures True / False",
        )
        plt.tight_layout()
        if self.save_plot:
            plt.savefig(
                self.save_path
                / f"{self.formula}_structure_matcher_heatmap_{self.experimental_string}.png",
                format="png",
            )
        else:
            plt.show()
        plt.clf()

        self._plot_histogram_of_distances(distances)

    def return_valid_spacegroups_for_pyxtal(
        self, elements: List[str], atoms_counts: List[int]
    ):
        my_crystal = pyxtal()

        all_possible_sg = []
        if self.experimental_references_only:
            reference_structures = (
                self.symmetry_evaluator.initialise_reference_structures(
                    self.formula, self.max_n_atoms_in_cell, False
                )
            )
        else:
            reference_structures = self.reference_structures
        for el in reference_structures:
            all_possible_sg.append(el.structure.get_space_group_info()[1])

        valid_spacegroups_for_combination = []
        for el in all_possible_sg:
            try:
                my_crystal.from_random(3, el, elements, atoms_counts)
            except Comp_CompatibilityError:
                continue
            valid_spacegroups_for_combination.append(el)

        number_of_atoms = sum(atoms_counts)
        with open(
            self.save_path
            / f"{self.formula}_{number_of_atoms}_allowed_symmetries.json",
            "w",
        ) as file:
            json.dump(valid_spacegroups_for_combination, file)
        return valid_spacegroups_for_combination


    def return_blocks_list(self):
        temp_atoms = Atoms(self.formula)
        number_of_formula_units = int(
            self.max_n_atoms_in_cell / len(temp_atoms.get_atomic_numbers())
        )
        temp_atoms = Atoms(self.formula * number_of_formula_units)
        blocks = temp_atoms.get_atomic_numbers().tolist()
        blocks.sort()
        return blocks

    def plot_references_as_groups(self, target_archive: Archive, x_axis_limits, y_axis_limits):
        self.symmetry_evaluator.group_structures_by_symmetry(
            archive=target_archive,
            experiment_directory_path=self.save_path,
            centroid_full_path=self.centroid_folder_path
            / self.centroid_filename[1:],
            filename_tag="experimental"
            if self.experimental_references_only
            else "exp_and_theory",
        x_axis_limits=x_axis_limits,
        y_axis_limits=y_axis_limits,
        )


if __name__ == "__main__":
    # elements_list = [["C"], ["Si", "O"], ["Si"], ["Si", "C"], ["Ti", "O"]]
    # atoms_counts_list = [[24], [8, 16], [24], [12, 12], [8, 16]]
    # formulas = ["C", "SiO2", "Si", "SiC", "TiO2"]

    elements_list = [["Ti", "O"]]
    atoms_counts_list = [[8, 16]]
    formulas = ["TiO2"]
    fitness_limits = [8.7, 9.5]
    band_gap_limits = [0, 4]
    shear_moduli_limits = [0, 120]
    reference_data_dump = []
    dict_summary = {}
    # for filter_experiment in [True]:
    for filter_experiment in [False, True]:
        filter_experiment_dump = []
        for i, formula in enumerate(formulas):
            reference_analyser = ReferenceAnalyser(
                formula=formula,
                max_n_atoms_in_cell=np.sum(np.array(atoms_counts_list[i])),
                experimental_references_only=filter_experiment,
                save_plots=True,
            )
            # print(formula)
            # band_gap_limits, shear_moduli_limits = reference_analyser.set_bd_limits(
            #     reference_analyser.band_gaps, reference_analyser.shear_moduli)

            reference_analyser.return_valid_spacegroups_for_pyxtal(
                elements=elements_list[i], atoms_counts=atoms_counts_list[i]
            )
            kdt = reference_analyser.initialise_kdt_and_centroids(
                number_of_niches=200,
                band_gap_limits=band_gap_limits,
                shear_moduli_limits=shear_moduli_limits,
            )
            if fitness_limits is None:
                fitness_limits = reference_analyser.propose_fitness_limits()
            bd_minimum_values = np.array([band_gap_limits[0], shear_moduli_limits[0]])
            bd_maximum_values = np.array([band_gap_limits[1], shear_moduli_limits[1]])


            target_archive = reference_analyser.create_model_archive(
                bd_minimum_values=bd_minimum_values,
                bd_maximum_values=bd_maximum_values,
                save_reference=not filter_experiment,
            )

            normalise_bd_values = (bd_minimum_values, bd_maximum_values) if reference_analyser.normalise_bd else None
            reference_analyser.plot_cvt_plot(
                target_archive=target_archive,
                bd_minimum_values=np.array([0, 0]) if reference_analyser.normalise_bd else bd_minimum_values,
                bd_maximum_values=np.array([1, 1]) if reference_analyser.normalise_bd else bd_maximum_values,
                fitness_limits=fitness_limits,
                x_axis_limits=bd_minimum_values,
                y_axis_limits=bd_maximum_values,
            )
            reference_analyser.plot_references_as_groups(target_archive,
                                                         x_axis_limits=bd_minimum_values,
                                                         y_axis_limits=bd_maximum_values,
                                                         )
            reference_analyser.heatmap_structure_matcher_distances(annotate=False)
            reference_analyser.plot_symmetries()
            reference_analyser.plot_fmax()
            # dict_summary[f"{formula}_{filter_experiment}"] = len(reference_analyser.reference_structures)
            # print(dict_summary)

            # filter_experiment_dump.append()
    # with open("../../.experiment.nosync/mp_reference_analysis/dict_summary.json", "w") as file:
    #     json.dump(dict_summary, file)
