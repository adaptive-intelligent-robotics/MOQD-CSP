import json
import pathlib
import pickle
from collections import defaultdict
from typing import Optional, Tuple, List

import numpy as np
from ase import Atoms
from ase.spacegroup import get_spacegroup
from matplotlib import pyplot as plt
from pymatgen.io.ase import AseAtomsAdaptor
from pyxtal import pyxtal
from pyxtal.msg import Comp_CompatibilityError
from sklearn.neighbors import KDTree

from csp_elites.crystal.crystal_evaluator import CrystalEvaluator
from csp_elites.crystal.materials_data_model import MaterialProperties
from csp_elites.crystal.symmetry_evaluator import SymmetryEvaluation
from csp_elites.map_elites.archive import Archive
from csp_elites.map_elites.elites_utils import cvt, write_centroids, __centroids_filename
from csp_elites.map_elites.elites_utils import __centroids_filename as get_centroids_filename
from csp_elites.reference_setup.reference_plotter import ReferencePlotter
from csp_elites.utils.asign_target_values_to_centroids import \
    reassign_data_from_pkl_to_new_centroids
from csp_elites.utils.experiment_parameters import ExperimentParameters
from csp_elites.utils.plot import load_centroids, plot_2d_map_elites_repertoire_marta
from csp_scripts.log_target_data_band_gap_shear import \
    return_predictions_bg_shear_for_list_of_mp_docs


class ReferenceAnalyser:
    def __init__(self,
        formula: str, max_n_atoms_in_cell: int, experimental_references_only: bool,
        save_plots: bool = True,
    ):
        self.formula = formula
        self.max_n_atoms_in_cell = max_n_atoms_in_cell
        self.experimental_references_only = experimental_references_only
        self.experimental_string = "experimental_only" if experimental_references_only else "exp_and_theory"

        self.main_experiments_directory = pathlib.Path(__file__).parent.parent.parent / ".experiment.nosync" / "experiments"
        self.centroid_folder_path = self.main_experiments_directory / "centroids"

        self.symmetry_evaluator = SymmetryEvaluation(
        formula=formula,
        tolerance=0.1,
        maximum_number_of_atoms_in_reference=max_n_atoms_in_cell,
        number_of_atoms_in_system=max_n_atoms_in_cell,
        filter_for_experimental_structures=experimental_references_only,
        )
        self.structures_to_consider = sorted(self.symmetry_evaluator.known_structures_docs, key=lambda x: (x.theoretical, len(x.structure)))
        self.crystal_evaluator = CrystalEvaluator(
            comparator=None,
            with_force_threshold=False,
            constrained_qd=False,
            relax_every_n_generations=0,
            fmax_relaxation_convergence=0.2,
            force_threshold_fmax=1.0,
            compute_gradients=False
        )
        self.reference_ids = [str(structure.material_id) for structure in self.structures_to_consider]
        self.behavioural_descriptors = [MaterialProperties.BAND_GAP, MaterialProperties.SHEAR_MODULUS]

        self.energies, self.fmax_list, self.band_gaps, self.shear_moduli = self.compute_target_values()
        self.centroid_filename = None
        self.plotting_helper = ReferencePlotter(save_plots)
        self.save_plot = save_plots
        self.bd_minimum_values = None
        self.bd_maximum_values = None

        self.save_path = self.main_experiments_directory.parent / "mp_reference_analysis" / f"{formula}_{max_n_atoms_in_cell}"
        self.save_path.mkdir(parents=True, exist_ok=True)

    def compute_target_values(self):
        predictions, band_gaps, shear_moduli = return_predictions_bg_shear_for_list_of_mp_docs(
            self.structures_to_consider)
        energies = [prediction["e"] for prediction in predictions]
        energies = np.array(energies) * -1
        forces = [prediction["f"] for prediction in predictions]
        fmax_list = []
        for force in forces:
            fmax_list.append(np.max((force ** 2).sum(axis=1)) ** 0.5)

        band_gaps = np.array(band_gaps)
        shear_moduli = np.array(shear_moduli)

        return energies, fmax_list, band_gaps, shear_moduli

    @staticmethod
    def set_bd_limits(band_gaps: np.ndarray, shear_moduli: np.ndarray):
        band_gap_limits = np.array([np.floor(band_gaps.min()) * 0.9, np.ceil(band_gaps.max() * 1.1)], dtype=int)
        shear_moduli_limits = np.array([np.floor(shear_moduli.min()) * 0.9, np.ceil(shear_moduli.max()) * 1.1], dtype=int)

        band_gap_min_max_diff = band_gap_limits[1] - band_gap_limits[0]
        shear_moduli_min_max_diff = shear_moduli_limits[1] - shear_moduli_limits[0]
        if not np.abs(shear_moduli_min_max_diff - band_gap_min_max_diff) < 0.2 * max([shear_moduli_min_max_diff, band_gap_min_max_diff]):
            print(f"Recommend setting band gaps manually, bg limits: {band_gap_limits}, shear moduli limits {shear_moduli_limits}")
            return band_gap_limits.tolist(), shear_moduli_limits.tolist()
        else:
            return band_gap_limits.tolist(), shear_moduli_limits.tolist()

    def propose_fitness_limits(self):
        energies = np.array(self.energies)
        limits = np.array([np.floor(energies.min() * - 0.5), np.ceil(energies.max() * 1.2) + 0.5])
        return limits.tolist()

    def initialise_kdt_and_centroids(self,
        number_of_niches: int, band_gap_limits: Optional[np.ndarray] = None, shear_moduli_limits: Optional[np.ndarray]= None,
    ):
        # create the CVT
        if band_gap_limits is None or shear_moduli_limits is None:
            band_gap_limits, shear_moduli_limits = self.set_bd_limits(self.band_gaps, self.shear_moduli)
        if band_gap_limits is None or shear_moduli_limits is None:
            print("Rerun with preset band gaps and shear moduli limits")

        bd_minimum_values = [band_gap_limits[0], shear_moduli_limits[0]]
        bd_maximum_values = [band_gap_limits[1], shear_moduli_limits[1]]
        c = cvt(number_of_niches, len(self.behavioural_descriptors),
                25000,
                bd_minimum_values,
                bd_maximum_values,
                self.centroid_folder_path,
                self.behavioural_descriptors,
                True,
                formula=self.formula,
                )
        kdt = KDTree(c, leaf_size=30, metric='euclidean')
        write_centroids(
            c, experiment_folder=self.centroid_folder_path,
            bd_names=self.behavioural_descriptors,
            bd_minimum_values=bd_minimum_values,
            bd_maximum_values=bd_maximum_values,
            formula=self.formula
        )
        del c
        self.centroid_filename = get_centroids_filename(
            k=number_of_niches, dim=len(self.behavioural_descriptors), bd_names=self.behavioural_descriptors,
            bd_minimum_values=bd_minimum_values, bd_maximum_values=bd_maximum_values, formula=self.formula,
        )
        return kdt

    def create_model_archive(self, save_reference = False):
        if self.centroid_filename is None:
            print("first create centroids or provide path to centroid file")

        descriptors = [(self.band_gaps[i], self.shear_moduli[i]) for i in range(len(self.band_gaps))]
        individuals = [AseAtomsAdaptor.get_atoms(el.structure) for el in self.structures_to_consider]

        centroids = reassign_data_from_pkl_to_new_centroids(
            centroids_file=self.centroid_folder_path.parent / self.centroid_filename[1:],
            target_data=[self.energies, None, descriptors, individuals],
            filter_for_number_of_atoms=None,
        )
        target_archive = Archive(
            fitnesses=np.array(self.energies),
            centroids=centroids,
            descriptors=np.array(descriptors),
            individuals=individuals,
            centroid_ids=None,
            labels=self.reference_ids,
        )
        target_archive.centroid_ids = Archive.assign_centroid_ids(centroids, self.centroid_folder_path.parent / self.centroid_filename[1:])

        if save_reference:
            all_data = []
            for i in range(len(self.energies)):
                one_data_point = []
                one_data_point.append(self.energies[i])
                one_data_point.append(centroids[i])
                one_data_point.append(descriptors[i])
                one_data_point.append(individuals[i].todict())

                all_data.append(one_data_point)

            with open(self.save_path / f"{self.formula}_band_gap_shear_modulus.pkl", "wb") as file:
                pickle.dump(all_data, file)

        return target_archive

    def plot_cvt_plot(self,
        target_archive: Archive, bd_minimum_values: np.ndarray, bd_maximum_values: np.ndarray,
        fitness_limits: np.ndarray,
    ):
        plotting_centroids = load_centroids(self.centroid_folder_path.parent / self.centroid_filename[1:])
        fitness_for_plotting, descriptors_for_plotting, labels_for_plotting = target_archive.convert_fitness_and_descriptors_to_plotting_format(
            plotting_centroids)

        if self.save_plot:
            directory_string = str(self.main_experiments_directory.parent / "mp_reference_analysis")
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
            directory_string=self.save_path,
            filename=f"{self.formula}_cvt_plot_{self.experimental_string}"
        )
        plt.clf()

    def plot_fmax(self, histogram_range: Optional[Tuple[int, int]]=None):
        histogram_range = histogram_range if histogram_range is None else (0, 1)
        fig, ax = plt.subplots()
        ax.hist(self.fmax_list, range=histogram_range, bins=40)
        ax.set_xlabel("Maximum Force on an Atom")
        ax.set_ylabel("Structure Count")
        ax.set_title("Maximum Force on Atom for Reference Structures")
        if self.save_plot:
            fig.savefig(self.save_path / f"{self.formula}_fmax_histogram_{self.experimental_string}.png",
                        format="png")
        else:
            fig.show()
        plt.clf()

    def plot_symmetries(self):
        fig, ax = plt.subplots()

        all_group_information = []
        for structure_group in [self.structures_to_consider]:
            symmetries = defaultdict(list)
            for el in structure_group:
                symmetry = get_spacegroup(
                    AseAtomsAdaptor.get_atoms(el.structure),
                    symprec=0.1).symbol
                symmetries[symmetry].append(str(el.material_id))
            all_group_information.append(symmetries)

        for i, symmetries_in_group in enumerate(all_group_information):
            ax.bar(list(symmetries_in_group.keys()),
                   [len(value) for value in symmetries_in_group.values()], label=self.reference_ids[i])

        ax.set_ylabel("Structure Count")
        ax.set_xlabel("Symmetry")
        ax.set_title("Symmetries Across Reference Structures")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        if len(self.structures_to_consider) == 2:
            fig.legend(loc="center right")
        plt.tight_layout()
        if self.save_plot:
            fig.savefig(self.save_path / f"{self.formula}_symmetries_histogram_{self.experimental_string}.png",
                        format="png")
        else:
            fig.show()
        plt.clf()

    def _plot_histogram_of_distances(self, distances):
        distances = np.array(distances)
        minimum_distances = [np.min(el[el > 0.001]) for el in distances]

        fig, ax = plt.subplots()
        ax.hist(minimum_distances, range=(0, 0.5), bins=20)
        ax.set_xlabel("Minimum Non-Zero Cosine Distance Between Reference Structures")
        ax.set_ylabel("Number of Structures")
        ax.set_title("Minimum Non-Zero Cosine Distance Between Reference Structures Across References")

        if self.save_plot:
            fig.savefig(self.save_path / f"{self.formula}_{self.max_n_atoms_in_cell}_distances_histogram_{self.experimental_string}.png", format="png")
        else:
            fig.show()
        plt.clf()

    def heatmap_structure_matcher_distances(self, annotate: bool = True):
        all_structure_matcher_matches = []
        distances = []

        for strucutre_1 in self.structures_to_consider:
            matches_for_structure_1 = []
            distances_for_structure_1 = []
            for structure_2 in self.structures_to_consider:
                matches_for_structure_1.append(
                    self.symmetry_evaluator.structure_matcher.fit(strucutre_1.structure,
                                                              structure_2.structure)
                )
                if len(strucutre_1.structure) == len(structure_2.structure):
                    strucutre_1.structure.sort()
                    structure_2.structure.sort()
                    distance_to_known_structure = float(
                        self.symmetry_evaluator.comparator._compare_structure(
                            AseAtomsAdaptor.get_atoms(strucutre_1.structure),
                            AseAtomsAdaptor.get_atoms(structure_2.structure)))
                else:
                    distance_to_known_structure = 1

                distances_for_structure_1.append(distance_to_known_structure)

            all_structure_matcher_matches.append(matches_for_structure_1)
            distances.append(distances_for_structure_1)

        im, cbar = self.plotting_helper.heatmap(np.array(distances), self.reference_ids, self.reference_ids,
                           xlabel="Material Project Reference Numbers",
                           ylabel="Material Project Reference Numbers",
                           title="Cosine Distances Between Fingerprints of Reference Structures",
                           cmap="YlGn",
                           cbarlabel="Cosine Distance Between Structure Fingerprints"
                           )
        if annotate:
            texts = self.plotting_helper.annotate_heatmap(im, valfmt="{x:.1f}")

        plt.tight_layout()
        if self.save_plot:
            plt.savefig(self.save_path / f"{self.formula}_distances_heatmap_no_annotate_{self.experimental_string}.png", format="png")
        else:
            plt.show()
        plt.clf()
        self.plotting_helper.heatmap(np.array(all_structure_matcher_matches), self.reference_ids, self.reference_ids,
                xlabel="Material Project Reference Numbers",
                ylabel="Material Project Reference Numbers",
                title="StructureMatcher Confusion Matrix",
                cmap="YlGn",
                cbarlabel="Match between structures True / False"
                )
        plt.tight_layout()
        if self.save_plot:
            plt.savefig(self.save_path / f"{self.formula}_structure_matcher_heatmap_{self.experimental_string}.png", format="png")
        else:
            plt.show()
        plt.clf()

        self._plot_histogram_of_distances(distances)

    def return_valid_spacegroups_for_pyxtal(self, elements: List[str], atoms_counts: List[int]):
        my_crystal = pyxtal()

        all_possible_sg = []
        if self.experimental_references_only:
            structures_to_consider = self.symmetry_evaluator.initialise_reference_structures(
                self.formula, self.max_n_atoms_in_cell, False)
        else:
            structures_to_consider = self.structures_to_consider
        for el in structures_to_consider:
            all_possible_sg.append(el.structure.get_space_group_info()[1])

        valid_spacegroups_for_combination = []
        for el in all_possible_sg:
            try:
                my_crystal.from_random(3, el, elements, atoms_counts)
            except Comp_CompatibilityError:
                continue
            valid_spacegroups_for_combination.append(el)

        number_of_atoms = sum(atoms_counts)
        with open(self.save_path / f"{self.formula}_{number_of_atoms}_allowed_symmetries.json", "w") as file:
            json.dump(valid_spacegroups_for_combination, file)
        return valid_spacegroups_for_combination

    def write_base_config(self, bd_minimum_values: np.ndarray, bd_maximum_values: np.ndarray,
                          fitness_limits: np.ndarray):

        blocks = self.return_blocks_list()

        experiment_parameters = ExperimentParameters.generate_default_to_populate()
        experiment_parameters.system_name = self.formula
        experiment_parameters.blocks = list(blocks)
        experiment_parameters.cvt_run_parameters["bd_minimum_values"] = list(bd_minimum_values)
        experiment_parameters.cvt_run_parameters["bd_maximum_values"] = list(bd_maximum_values)
        experiment_parameters.fitness_min_max_values = list(fitness_limits)
        experiment_parameters.save_as_json(self.save_path)

    def return_blocks_list(self):
        temp_atoms = Atoms(self.formula)
        number_of_formula_units = int(self.max_n_atoms_in_cell / len(temp_atoms.get_atomic_numbers()))
        temp_atoms = Atoms(self.formula * number_of_formula_units)
        blocks = temp_atoms.get_atomic_numbers().tolist()
        blocks.sort()
        return blocks


if __name__ == '__main__':
    elements_list = [["C"], ["Si", "O"], ["Si"], ["Si", "C"], ["Ti", "O"]]
    atoms_counts_list = [[24], [8, 16], [24], [12, 12], [8, 16]]
    formulas = ["C", "SiO2", "Si", "SiC", "TiO2"]

    # elements_list = [["C"]]
    # atoms_counts_list = [[24]]
    # formulas = ["C"]

    dict_summary = {}

    for filter_experiment in [False]:
        for i, formula in enumerate(formulas):
            reference_analyser = ReferenceAnalyser(
                formula=formula,
                max_n_atoms_in_cell=24,
                experimental_references_only=filter_experiment,
                save_plots=True
            )

            band_gap_limits, shear_moduli_limits = reference_analyser.set_bd_limits(
                reference_analyser.band_gaps, reference_analyser.shear_moduli)
            # band_gap_limits, shear_moduli_limits = [100, 200], [20, 220]
            reference_analyser.return_valid_spacegroups_for_pyxtal(elements=elements_list[i], atoms_counts=atoms_counts_list[i])
            kdt = reference_analyser.initialise_kdt_and_centroids(
                number_of_niches=200,
                band_gap_limits=band_gap_limits,
                shear_moduli_limits=shear_moduli_limits,
            )
            fitness_limits = reference_analyser.propose_fitness_limits()

            reference_analyser.write_base_config(
                bd_minimum_values=[band_gap_limits[0], shear_moduli_limits[0]],
                bd_maximum_values=[band_gap_limits[1], shear_moduli_limits[1]],
                fitness_limits=fitness_limits,
            )
            # target_archive = reference_analyser.create_model_archive(not filter_experiment)
            #
            # reference_analyser.plot_cvt_plot(
            #     target_archive=target_archive,
            #     bd_minimum_values=[band_gap_limits[0], shear_moduli_limits[0]],
            #     bd_maximum_values=[band_gap_limits[1], shear_moduli_limits[1]],
            #     fitness_limits=fitness_limits,
            # )
            # reference_analyser.heatmap_structure_matcher_distances()
            # reference_analyser.plot_symmetries()
            # reference_analyser.plot_fmax()
            # dict_summary[f"{formula}_{filter_experiment}"] = len(reference_analyser.structures_to_consider)
            # print(dict_summary)
    # with open("../../.experiment.nosync/mp_reference_analysis/dict_summary.json", "w") as file:
    #     json.dump(dict_summary, file)
