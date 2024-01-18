import os
import yaml
import numpy as np
from csp_elites.map_elites.archive import Archive
from csp_elites.mome.mome_archive import MOArchive
from csp_elites.utils.plot import (
    load_centroids,
    get_voronoi_finite_polygons_2d,
    plot_2d_map_elites_repertoire_grid
)
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors


import os
from typing import List, Dict, Optional, Tuple, Union
from omegaconf import OmegaConf

from typing import List

class ReferenceMOQDIllumination:
    
    def __init__(self,
        parent_dirname: str,
        experiment_names: List[str],
        median_replications: List[str],
        algo_name: str = "mome_biased",
    )-> None:
        
        self.parent_dirname = parent_dirname
        self.experiment_names = experiment_names
        self.algo_name = algo_name
        
        self.configs = {}
        self.centroids = {}
        self.archives = {}
        self.archive_dicts = {}
        self.reference_data = {}
        self.reference_data_dicts = {}
        
        # Load in all archives
        for experiment_num, experiment in enumerate(self.experiment_names):
            parent_dirname = self.parent_dirname + experiment + "/" + self.algo_name + "/" + median_replications[experiment_num] + "/"
            config_dict = self.load_config(parent_dirname)
            config = OmegaConf.create(config_dict)
            self.configs[experiment] = config
            reference_centroids, centroid_directory_path, _ = self.load_reference_centroids(config)
            self.centroids[experiment] = reference_centroids
            self.archives[experiment] = self.load_archive(parent_dirname, config, centroid_directory_path, reference_centroids)[0]
            self.archive_dicts[experiment] = self.load_archive(parent_dirname, config, centroid_directory_path, reference_centroids)[1]
            self.reference_data[experiment] = self.load_reference_data(config, centroid_directory_path, reference_centroids)[0]
            self.reference_data_dicts[experiment] = self.load_reference_data(config, centroid_directory_path, reference_centroids)[1]                    
        
    def load_config(self, parent_dirname):
        with open(os.path.join(parent_dirname, ".hydra/config.yaml"), "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
        return config
  
    def load_reference_centroids(self, config):
        
        if config.system.system_name == "TiO2" or config.system.system_name is None:
            bd_tag = ""
        else:
            bd_tag = "_" + config.system.system_name

        for i, bd_name in enumerate(config.behavioural_descriptors):
            if config.normalise_bd:
                bd_min, bd_max = 0, 1
            else:
                bd_min, bd_max = config.system.bd_minimum_values[i], config.system.bd_maximum_values[i]
            bd_tag += f"_{bd_name}_{bd_min}_{bd_max}"

        centroid_filename ="/centroids_" + str(config.number_of_niches) + "_" + str(config.system.n_behavioural_descriptor_dimensions) + bd_tag + ".dat"
        centroid_directory_path = "./reference_data/centroids"+centroid_filename
        centroids = load_centroids(centroid_directory_path)
        
        return centroids, centroid_directory_path, centroid_filename
  
    def load_archive(self, parent_dirname, config, centroid_directory_path, reference_centroids):
        archive = MOArchive.from_archive(
            parent_dirname + "archive_" +str(config.maximum_evaluations) + ".pkl",
            centroid_filepath=centroid_directory_path,
        )
        archive_dict = archive.create_mo_archive_dict(reference_centroids)

        return archive, archive_dict
    
    def load_reference_data(
        self,
        config,
        centroid_directory_path,
        centroids,
    ):  
        
        if config.system.system_name == "TiO2":
            filename = f"mp_reference_analysis/TiO2_24/TiO2_target_data_centroids_200_2_band_gap_0_1_shear_modulus_0_1.csv"
        
        else: 
            filename = f"mp_reference_analysis/{config.system.system_name}_24/{config.system.system_name}_target_data_centroids_200_2_{config.system.system_name}_band_gap_0_1_shear_modulus_0_1.csv"

        normalise_bd_values = (
            (
                config.system.bd_minimum_values,
                config.system.bd_maximum_values,
            )
            if config.normalise_bd
            else None
        )
        
        reference_archive = MOArchive.from_reference_csv_path(
            filename,
            normalise_bd_values=normalise_bd_values,
            centroids_path=centroid_directory_path,
        )

        reference_archive_dict = reference_archive.create_mo_archive_dict(centroids)
        
        return reference_archive, reference_archive_dict
         
    def plot_reference_illumination(
        self,
        objective: str = "energy",
        save_dir: str = "results/analysis/reference_data_plots/",
    ):
        
        num_cols = len(self.experiment_names)
        
        fig, ax = plt.subplots(2,
            num_cols,
            figsize=(num_cols*5 + 1, 5*2),
            sharex=True,
            sharey=True
        )
        
        for axes_num, system in enumerate(self.experiment_names):
            colours = self.compare_to_reference_data(
                self.archive_dicts[system],
                self.reference_data_dicts[system],
                centroids=self.centroids[system],
                objective="energy",
            )
            ax.ravel()[axes_num] = self.plot_one_archive(
                colours=colours,
                reference_archive_dict=self.reference_data_dicts[system],
                archive_dict=self.archive_dicts[system],
                centroids=self.centroids[system],
                subplot_title=f"{system}",  
                ax=ax.ravel()[axes_num],
                ax_number=axes_num,
                max_axis_number=num_cols * 2,
                objective="energy",
            )
            
        

        for axes_num, system in enumerate(self.experiment_names):
            ax_plot_num = num_cols + axes_num
            colours = self.compare_to_reference_data(
                self.archive_dicts[system],
                self.reference_data_dicts[system],
                centroids=self.centroids[system],
                objective="magmom",
            )
            ax.ravel()[ax_plot_num] = self.plot_one_archive(
                colours=colours,
                reference_archive_dict=self.reference_data_dicts[system],
                archive_dict=self.archive_dicts[system],
                centroids=self.centroids[system],
                subplot_title=f"{system}",  
                ax=ax.ravel()[ax_plot_num],
                ax_number=ax_plot_num,
                max_axis_number=num_cols * 2,
                objective="magmom",
            )   

        handles, labels = ax.ravel()[-1].get_legend_handles_labels()
        unique = [
            (h, l)
            for i, (h, l) in enumerate(zip(handles, labels))
            if l not in labels[:i]
        ]
        plt.figlegend(
            *zip(*unique),
            loc="lower center",
            # bbox_to_anchor=(0.5, -0.2),
            fontsize=20,
            markerscale=5,
            ncols=2,
        )
        plt.subplots_adjust(
            bottom=0.15,
            wspace=0.05,
        )
            
        plt.savefig(os.path.join(save_dir, f"reference_comparison_plot.png"))
        plt.close()
        
    
    def compare_to_reference_data(
        self,
        archive_dict,
        reference_data_dict,
        centroids,
        objective: str = "energy",
    ):
        
        colours_for_plotting = np.full((len(centroids)), -np.inf)
        
        for cell in reference_data_dict.keys():
            if cell in archive_dict.keys():
                archive_values = [i[objective] for i in archive_dict[cell]]
            else:
                archive_values = [-np.inf]
            reference_values = [i[objective] for i in reference_data_dict[cell]]
            
            better_than_reference = np.max(archive_values) > np.max(reference_values)
                
            colours_for_plotting[cell] = better_than_reference
        
        return colours_for_plotting
         
            
    def plot_one_archive(
        self,
        colours,
        reference_archive_dict,
        archive_dict,
        centroids,  
        subplot_title: str = None,
        ax: plt.Axes = None,
        ax_number: int = 0,
        max_axis_number: int = 1,
        num_systems: int = 5,
        objective="energy"
        ):
                
        # add map elites plot on last axes
        _, axes = self.plot_matches_difference(
            fill_colours=colours,
            reference_data_dict=reference_archive_dict,
            archive_dict=archive_dict,
            objective=objective,
            centroids=centroids,
            ax=ax,
            ax_number=ax_number,
            max_axis_number=max_axis_number,
            num_systems=num_systems,
            subplot_title=subplot_title,
        )
         
        return axes
    
    
    def plot_matches_difference(
        self,
        fill_colours,
        reference_data_dict,
        archive_dict: Archive,
        objective: str,
        centroids: np.ndarray,
        ax: plt.Axes = None,
        ax_number: int = 0,
        max_axis_number: int = 1,
        num_systems: int = 1,
        subplot_title: str = None,
        axis_labels: List[str] = ["Band Gap, eV", "Shear Modulus, GPa"],
        x_axis_limits: Optional[Tuple[float, float]] = None,
        y_axis_limits: Optional[Tuple[float, float]] = None,
        
    ):
        if objective == "magmom":
            objective = "magnetism"
        """Adapted from wdac plot 2d cvt centroids function"""
        fig = None
        if ax is None:
            fig, ax = plt.subplots(facecolor="white", edgecolor="white")

        # create the plot object
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.set(adjustable="box", aspect="equal")

        # create the regions and vertices from centroids
        regions, vertices = get_voronoi_finite_polygons_2d(centroids)
        # fill the plot with contours
        target_centroid_ids = reference_data_dict.keys()
        
        colour_dict = {
            "no_match": mcolors.CSS4_COLORS["silver"],
            f"{objective}_above_reference": mcolors.CSS4_COLORS["mediumaquamarine"],
            f"{objective}_below_reference": mcolors.CSS4_COLORS["rosybrown"],
        }
        
        label_dict = {
            f"{objective}_below_reference": f"Below Reference",
            f"{objective}_above_reference": f"Above Reference",
        }
        
        for i, region in enumerate(regions):
            polygon = vertices[region]
            ax.fill(
                *zip(*polygon), alpha=0.05, edgecolor="black", facecolor="white", lw=1
            )
            if i in target_centroid_ids:
                ax.fill(*zip(*polygon), edgecolor="gray", facecolor="none", lw=2)
                if fill_colours[i] == 0:
                    ax.fill(
                        *zip(*polygon),
                        facecolor=colour_dict[f"{objective}_below_reference"],
                        label=label_dict[f"{objective}_below_reference"],
                    )
                elif fill_colours[i] == 1:
                    ax.fill(
                        *zip(*polygon),
                        facecolor=colour_dict[f"{objective}_above_reference"],
                        label=label_dict[f"{objective}_above_reference"],
                    )
                        
        
        if ax_number == 0 or ax_number == num_systems:
            ax.set_ylabel(f"{axis_labels[1]}", size=20, labelpad=10)
        
        if ax_number > num_systems:
            ax.set_xlabel(f"{axis_labels[0]}", size=20, labelpad=10)

        title = f"{subplot_title} " + f"{objective}".capitalize()
        ax.set_title(title, size=20)
        ax.set_aspect("equal")

        if x_axis_limits is not None and y_axis_limits is not None:
            x_tick_labels = np.linspace(x_axis_limits[0], x_axis_limits[1], 6)
            y_tick_labels = np.linspace(y_axis_limits[0], y_axis_limits[1], 6)
            ax.set_xticklabels([np.around(el, 1) for el in x_tick_labels])
            ax.set_yticklabels([np.around(el, 1) for el in y_tick_labels])
            
        return fig, ax


if __name__ == "__main__":
    
    dirname = "results/"
    experiment_names = ["C", "Si", "SiC", "SiO2", "TiO2"]
    median_replications = ["2024-01-13_165633", "2024-01-16_013843", "2024-01-16_093645", "2024-01-12_125056", "2024-01-09_230842"]
    algo_name = "mome_biased"


    illumination_plotter = ReferenceMOQDIllumination(
        parent_dirname=dirname,
        experiment_names=experiment_names,
        median_replications=median_replications,
        algo_name=algo_name,
    )
    

    illumination_plotter.plot_reference_illumination()
