import json
import os
import pathlib
from typing import Union, Optional

import numpy as np
from ase.ga.utilities import CellBounds

from csp_elites.crystal.materials_data_model import MaterialProperties, StartGenerators
from csp_elites.evaluation.symmetry_evaluator import SymmetryEvaluation

from csp_elites.map_elites.archive import Archive
from csp_elites.utils.asign_target_values_to_centroids import (
    reassign_data_from_pkl_to_new_centroids,
)
from csp_elites.utils.get_mpi_structures import get_all_materials_with_formula
from csp_elites.utils.plot import (
    load_centroids,
    load_archive_from_pickle,
    plot_all_maps_in_archive,
    plot_all_statistics_from_file,
    plot_gif,
)
from retrieve_results.experiment_organiser import ExperimentOrganiser


class ExperimentProcessor:
    def __init__(
        self,
        config,
        save_structure_images: bool = True,
        filter_for_experimental_structures: bool = False,
        centroid_filename: str = None,
        centroids_load_dir: Optional[pathlib.Path] = None,
        experiment_save_dir: Optional[pathlib.Path] = None,
    ):
        # Save config
        self.config = config
        self.formula = config.system.system_name

        # Plotting options
        self.save_structure_images = save_structure_images
        self.filter_for_experimental_structures = filter_for_experimental_structures

        # Set up directories
        self.experiment_location: Optional[pathlib.Path] = pathlib.Path(__file__).parent.parent
        self.experiment_save_dir = experiment_save_dir
        if self.config.cvt_use_cache:
            self.centroid_directory_path =  centroids_load_dir
        else:   
            self.centroid_directory_path = experiment_save_dir
        self.centroid_filename = centroid_filename
        
        # Load archive
        self.all_centroids = load_centroids(str(self.centroid_directory_path)+centroid_filename)
        self.fitness_limits = config.system.fitness_min_max_values
        
        
    def plot(self, annotate: bool = True, force_replot: bool = False):
        
        if self.config.normalise_bd:
            min_bds, max_bds = [0, 0], [1, 1]
        else:
            min_bds, max_bds = self.config.bd_minimum_values, self.config.bd_maximum_values
        
        plot_all_maps_in_archive(
            experiment_directory_path=str(self.experiment_save_dir),
            experiment_parameters=self.config,
            all_centroids=self.all_centroids,
            target_centroids=self.compute_target_centroids(),
            bd_minimum_values=min_bds,
            bd_maximum_values=max_bds,
            annotate=annotate,
            force_replot=force_replot,
        )
        plot_gif(experiment_directory_path=str(self.experiment_save_dir))
        plot_all_statistics_from_file(
            filename=f"{self.experiment_save_dir}/metrics_history.csv",
            save_location=f"{self.experiment_save_dir}/",
        )

    def _get_last_archive_number(self):
        return max(
            [
                int(name.lstrip("archive_").rstrip(".pkl"))
                for name in os.listdir(self.experiment_save_dir)
                if (
                    (not os.path.isdir(name))
                    and ("archive_" in name)
                    and (".pkl" in name)
                )
            ]
        )

    def compute_target_centroids(self,
        ):
        number_of_atoms = len(self.config.system.blocks)
        bd_tag = [
            bd.value
            for bd in self.config.behavioural_descriptors
        ]
        tag = ""
        for el in bd_tag:
            tag += f"{el}_"
            
        comparison_data_location = f"/mp_reference_analysis/{self.formula}_{number_of_atoms}/{self.formula}_{tag[:-1]}.pkl"
        comparison_data_packed = load_archive_from_pickle(str(self.experiment_location) + comparison_data_location)

        normalise_bd_values = (
            (
                self.config.system.bd_minimum_values,
                self.config.system.bd_maximum_values,
            )
            if self.config.normalise_bd
            else None
        )

        target_centroids = reassign_data_from_pkl_to_new_centroids(
            centroids_file=str(self.centroid_directory_path) + self.centroid_filename,
            target_data=comparison_data_packed,
            filter_for_number_of_atoms=self.config.system.fitler_comparison_data_for_n_atoms,
            normalise_bd_values=normalise_bd_values,
        )
        return target_centroids

    def get_material_project_info(self):
        structure_info, known_structures = get_all_materials_with_formula(
            self.config.system.system_name
        )
        return structure_info, known_structures

    def process_symmetry(self, annotate=True):
        archive_number = self._get_last_archive_number()
        unrelaxed_archive_location = (
            self.experiment_save_dir / f"archive_{archive_number}.pkl"
        )

        centroid_tag = str(self.centroid_filename)[1:].rstrip(".dat")

        number_of_atoms = self.config.filter_starting_Structures
        target_data_path = (
            self.experiment_location
            / "mp_reference_analysis"
            / f"{self.formula}_{number_of_atoms}"
            / f"{self.formula}_target_data_{centroid_tag}.csv"
        )
        print(target_data_path)
        # todo: change to include formula
        if not os.path.isfile(target_data_path):
            target_data_path = None

        archive = Archive.from_archive(
            unrelaxed_archive_location,
            centroid_filepath=str(self.centroid_directory_path)+self.centroid_filename
        )

        normalise_bd_values = (
            (
                self.config.system.bd_minimum_values,
                self.config.system.bd_maximum_values,
            )
            if self.config.normalise_bd
            else None
        )

        tareget_archive = Archive.from_reference_csv_path(
            target_data_path,
            normalise_bd_values=normalise_bd_values,
            centroids_path=str(self.centroid_directory_path)+self.centroid_filename,
        )
        symmetry_evaluation = SymmetryEvaluation(
            formula=self.formula,
            filter_for_experimental_structures=self.filter_for_experimental_structures,
            reference_data_archive=tareget_archive,
        )

        (
            matched_space_group_dict,
            spacegroup_dictionary,
        ) = symmetry_evaluation.find_individuals_with_reference_symmetries(
            individuals=archive.individuals, indices_to_check=None
        )

        energy_indices = symmetry_evaluation.save_best_structures_by_energy(
            archive=archive,
            fitness_range=(9.1, 9.5),
            top_n_individuals_to_save=10,
            directory_to_save=self.experiment_save_dir,
            save_primitive=False,
            save_visuals=self.save_structure_images,
        )

        symmetry_indices = symmetry_evaluation.save_best_structures_by_symmetry(
            archive=archive,
            matched_space_group_dict=matched_space_group_dict,
            directory_to_save=self.experiment_save_dir,
            save_primitive=False,
            save_visuals=self.save_structure_images,
        )

        all_individual_indices_to_check = np.unique(symmetry_indices + energy_indices)

        df, individuals_with_matches = symmetry_evaluation.executive_summary_csv(
            archive=archive,
            indices_to_compare=list(all_individual_indices_to_check),
            directory_to_save=self.experiment_save_dir,
        )

        symmetry_evaluation.group_structures_by_symmetry(
            archive=archive,
            experiment_directory_path=self.experiment_save_dir,
            centroid_full_path=str(self.centroid_directory_path)+self.centroid_filename,
            x_axis_limits=(
                self.config.system.bd_minimum_values[0],
                self.config.system.bd_maximum_values[0],
            ),
            y_axis_limits=(
                self.config.system.bd_minimum_values[1],
                self.config.system.bd_maximum_values[1],
            ),
        )

        if individuals_with_matches and target_data_path is not None:
            (
                plotting_from_archive,
                plotting_from_mp,
            ) = symmetry_evaluation.matches_for_plotting(individuals_with_matches)

            report_statistic_summary_dict = (
                symmetry_evaluation.write_report_summary_json(
                    plotting_from_archive,
                    directory_string=str(self.experiment_save_dir),
                )
            )

            symmetry_evaluation.plot_matches_energy_difference(
                archive=archive,
                plotting_matches=plotting_from_archive,
                centroids=self.all_centroids,
                centroids_from_archive=archive.centroid_ids,
                minval=[0, 0],
                maxval=[1, 1],
                directory_string=str(self.experiment_save_dir),
                annotate=False,
                x_axis_limits=(
                    self.config.system.bd_minimum_values[0],
                    self.config.system.bd_maximum_values[0],
                ),
                y_axis_limits=(
                    self.config.system.bd_minimum_values[1],
                    self.config.system.bd_maximum_values[1],
                ),
            )

            symmetry_evaluation.plot_matches_mapped_to_references(
                plotting_matches=plotting_from_mp,
                centroids=self.all_centroids,
                centroids_from_archive=archive.centroid_ids,
                minval=[0, 0]
                if self.config.normalise_bd
                else self.config.system.bd_minimum_values,
                maxval=[1, 1]
                if self.config.normalise_bd
                else self.config.system.bd_maximum_values,
                directory_string=str(self.experiment_save_dir),
                annotate=annotate,
                x_axis_limits=(
                    self.config.system.bd_minimum_values[0],
                    self.config.system.bd_maximum_values[0],
                ),
                y_axis_limits=(
                    self.config.system.bd_minimum_values[1],
                    self.config.system.bd_maximum_values[1],
                ),
            )

            symmetry_evaluation.plot_matches_mapped_to_references(
                plotting_matches=plotting_from_archive,
                centroids=self.all_centroids,
                centroids_from_archive=archive.centroid_ids,
                minval=[0, 0]
                if self.config.normalise_bd
                else self.config.system.bd_minimum_values,
                maxval=[1, 1]
                if self.config.normalise_bd
                else self.config.system.bd_maximum_values,
                directory_string=str(self.experiment_save_dir),
                annotate=annotate,
                x_axis_limits=(
                    self.config.system.bd_minimum_values[0],
                    self.config.system.bd_maximum_values[0],
                ),
                y_axis_limits=(
                    self.config.system.bd_minimum_values[1],
                    self.config.system.bd_maximum_values[1],
                ),
            )



if __name__ == "__main__":
    experiment_date = "0822"
    save_structure_images = True
    filter_for_experimental_structures = False

    experiment_organiser = ExperimentOrganiser()
    folder_list = experiment_organiser.get_all_folders_with_date(experiment_date)
    config_mapping, config_dict_csv = experiment_organiser.get_config_data(
        experiment_date
    )
    config_names = list(config_mapping.keys())
    experiment_tags_list = list(config_mapping.values())
    experiment_organiser.map_config_data_to_experiment(
        folder_list, config_dict_csv, experiment_date
    )
