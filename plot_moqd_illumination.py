import os
import yaml
import numpy as np
from csp_elites.mome.mome_archive import MOArchive
from csp_elites.utils.plot import (
    load_centroids,
    plot_2d_map_elites_repertoire_grid
)
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from omegaconf import OmegaConf

from typing import List

class MOQDIllumination:
    
    def __init__(self,
        parent_dirname: str,
    )-> None:
        
        self.parent_dirname = parent_dirname
        config_dict = self.load_config()
        self.config = OmegaConf.create(config_dict)
        
        self.reference_centroids = self.load_reference_centroids()
        
        self.archive, self.archive_dict = self.load_archive()
                
    def load_config(self):
        with open(os.path.join(self.parent_dirname, ".hydra/config.yaml"), "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
        return config
  
    def load_reference_centroids(self):
        
        if self.config.system.system_name == "TiO2" or self.config.system.system_name is None:
            bd_tag = ""
        else:
            bd_tag = "_" + self.config.system.system_name

        for i, bd_name in enumerate(self.config.behavioural_descriptors):
            if self.config.normalise_bd:
                bd_min, bd_max = 0, 1
            else:
                bd_min, bd_max = self.config.system.bd_minimum_values[i], self.config.system.bd_maximum_values[i]
            bd_tag += f"_{bd_name}_{bd_min}_{bd_max}"

        self.centroid_filename ="/centroids_" + str(self.config.number_of_niches) + "_" + str(self.config.system.n_behavioural_descriptor_dimensions) + bd_tag + ".dat"
        self.centroid_directory_path = "./reference_data/centroids"+self.centroid_filename
        centroids = load_centroids(self.centroid_directory_path)
        
        return centroids
  
    def load_archive(self):
        
        archive = MOArchive.from_archive(
            self.parent_dirname + "archive_" +str(self.config.maximum_evaluations) + ".pkl",
            centroid_filepath=self.centroid_directory_path,
        )
        archive_dict = archive.create_mo_archive_dict(self.reference_centroids)

        return archive, archive_dict
            
    def plot_all_archives(
        self,
        percentages_list: List[float] = [0, 0.25, 0.5, 0.75, 1],
        save_dir: str = "results/analysis/illumination_plots/",
    ):
        
        # First find max energy per cell
        self.find_max_energy_per_cell()
        self.find_max_magmom_per_cell()
        
        num_plots = len(percentages_list)
        
        fig, ax = plt.subplots(1,
            num_plots,
            figsize=(num_plots*5 + 1, 5),
            sharex=True,
            sharey=True
        )

        cmap = mpl.cm.get_cmap('viridis')
        normalizer = Normalize(vmin=self.archive_min_magmom, vmax=self.archive_max_magmom)
        im = mpl.cm.ScalarMappable(norm=normalizer)

        for axes_num, interpolation_percentage in enumerate(percentages_list):
            magmoms = self.find_projection_per_cell(interpolation_percentage)
            ax.ravel()[axes_num] = self.plot_one_archive(
                magmoms_for_plotting=magmoms,
                subplot_title=f"{interpolation_percentage*100}% Max Energy",  
                ax=ax.ravel()[axes_num],
                ax_number=axes_num,
                max_axis_number=num_plots-1,
            )

        cax,kw = mpl.colorbar.make_axes([axe for axe in ax.flat], shrink=0.75, pad=0.03, aspect=10)
        cbar = fig.colorbar(im, cax=cax, **kw)
        cbar.ax.set_ylabel("Magnetic Moment, $\mu_B$", size=15, labelpad=10)
            
        # cbar.ax.tick_params()
        
        ax[0].set_ylabel("Shear Modulus, GPa", loc="center", size=20, labelpad=10)
        ax[2].set_xlabel("Band Gap, eV", loc="center", size=20, labelpad=10)
        plt.savefig(os.path.join(save_dir, f"illumination_plot_{self.config.system.system_name}_{self.config.algo.algo_name}_{self.config.random_seed}.png"))
        plt.close()


    def find_max_energy_per_cell(self):
        
        max_energy_dict = {}
        min_energy_dict = {}
        
        for cell in self.archive_dict.keys():
            energies = [i["energy"] for i in self.archive_dict[cell]]
            max_energy_dict[cell] = np.max(energies)
            min_energy_dict[cell] = np.min(energies)
        
        self.archive_max_energy = np.max(list(max_energy_dict.values()))
        self.archive_min_energy = np.min(list(min_energy_dict.values()))
        self.max_energy_per_cell_dict = max_energy_dict
        self.min_energy_per_cell_dict = min_energy_dict
    
    def find_max_magmom_per_cell(self):
        
        max_magmom_dict = {}
        min_magmom_dict = {}
        
        for cell in self.archive_dict.keys():
            magmoms = [i["magmom"] for i in self.archive_dict[cell]]
            max_magmom_dict[cell] = np.max(magmoms)
            min_magmom_dict[cell] = np.min(magmoms)
        
        self.archive_max_magmom = np.max(list(max_magmom_dict.values()))
        self.archive_min_magmom = np.min(list(min_magmom_dict.values()))
        self.max_magmom_per_cell_dict = max_magmom_dict
        self.min_magmom_per_cell_dict = min_magmom_dict
    
    def find_projection_per_cell(self,
        interpolation_percentage
        ):
            
        magmoms_for_plotting = np.full((len(self.reference_centroids)), -np.inf)
        
        for cell in self.archive_dict.keys():
    
        # for cell in [34]:
            # print("------------")
            # print("CELL MAX ENERGY: ", self.max_energy_per_cell_dict[cell])
            # print("THRESHOLD: ", self.max_energy_per_cell_dict[cell]*percentage_of_max)
            # print("MAGMOMS:", [i["magmom"] for i in self.archive_dict[cell]])
            # print("ENERGIES:", [i["energy"] for i in self.archive_dict[cell]])
            # print("VALID MAGMOMS", [i["magmom"] for i in self.archive_dict[cell] if i["energy"]>=self.max_energy_per_cell_dict[cell]*percentage_of_max])
            threshold = self.min_energy_per_cell_dict[cell] + (self.max_energy_per_cell_dict[cell] - self.min_energy_per_cell_dict[cell])*interpolation_percentage

            valid_magmoms = [i["magmom"] for i in self.archive_dict[cell] if i["energy"]>=threshold]
            
            if len(valid_magmoms) > 0:
                max_magmom = np.max(valid_magmoms)
            else:
                max_magmom = -np.inf
            magmoms_for_plotting[cell] = max_magmom
        
        return magmoms_for_plotting
            
    def plot_one_archive(
        self,
        magmoms_for_plotting,
        subplot_title: str = None,
        ax: plt.Axes = None,
        ax_number: int = 0,
        max_axis_number: int = 1,
        ):
        
        # add map elites plot on last axes
        _, axes = plot_2d_map_elites_repertoire_grid(
            centroids=self.reference_centroids,
            repertoire_fitnesses=magmoms_for_plotting,
            minval=0,
            maxval=1,
            vmin=self.archive_min_magmom,
            vmax=self.archive_max_magmom,
            ax=ax,
            ax_number=ax_number,
            max_axis_number=max_axis_number,
            subplot_title=subplot_title,
            annotate=False,
        )
         
        return axes
    



if __name__ == "__main__":
    
    dirname = "results/"
    experiment_names = ["C", "Si", "SiC", "SiO2", "TiO2"]
    
    for experiment_name in experiment_names:
        for experiment_replication in os.scandir(os.path.join(dirname, experiment_name, "mome_biased")):
            parent_dirname = os.path.join(dirname, experiment_name,"mome_biased", experiment_replication.name) + "/"
            print("PARENT DIRNAME: ", parent_dirname)
            illumination_plotter = MOQDIllumination(
                parent_dirname=parent_dirname,
            )
            illumination_plotter.plot_all_archives()    
