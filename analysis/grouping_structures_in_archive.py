import pathlib
from typing import Optional, List, Tuple, Union

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor
from matplotlib import pyplot as plt, cm

from csp_elites.map_elites.archive import Archive
import matplotlib as mpl

from csp_elites.utils.plot import get_voronoi_finite_polygons_2d, load_centroids


def plot_2d_groups_of_structures(
    centroids: np.ndarray,
    list_of_centroid_groups: List[int],
    list_of_colors: List[str],
    minval: np.ndarray,
    maxval: np.ndarray,
    target_centroids: Optional[np.ndarray] = None,
    directory_string: Optional[str]= None,
    filename: Optional[str] = "cvt_plot",
    axis_labels: List[str] = ["band_gap", "shear_modulus"],

) -> Tuple[Optional[Figure], Axes]:
    num_descriptors = centroids.shape[1]
    if num_descriptors != 2:
        raise NotImplementedError("Grid plot supports 2 descriptors only for now.")

    my_cmap = cm.viridis

    # set the parameters
    font_size = 12
    params = {
        "axes.labelsize": font_size,
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "text.usetex": False,
        "figure.figsize": [10, 10],
    }

    mpl.rcParams.update(params)

    # create the plot object
    fig, ax = plt.subplots(facecolor="white", edgecolor="white")

    if len(np.array(minval).shape) == 0 and len(np.array(maxval).shape) == 0:
        ax.set_xlim(minval, maxval)
        ax.set_ylim(minval, maxval)
    else:
        ax.set_xlim(minval[0], maxval[0])
        ax.set_ylim(minval[1], maxval[1])

    ax.set(adjustable="box", aspect="equal")

    # create the regions and vertices from centroids
    regions, vertices = get_voronoi_finite_polygons_2d(centroids)


    # fill the plot with contours
    for i, region in enumerate(regions):
        polygon = vertices[region]
        ax.fill(*zip(*polygon), alpha=0.05, edgecolor="black", facecolor="white", lw=1)
        if target_centroids is not None:
            if centroids[i] in np.array(target_centroids):
                ax.fill(*zip(*polygon), edgecolor="red", facecolor="white", lw=4)
    # fill the plot with the colors
    for group_id, group in enumerate(list_of_centroid_groups):
        for idx in group:
            region = regions[idx]
            polygon = vertices[region]
            ax.fill(*zip(*polygon), alpha=0.8, color=list_of_colors[group_id])
            ax.annotate(group_id, (centroids[idx, 0], centroids[idx, 1]))
    np.set_printoptions(2)
    # aesthetic
    ax.set_xlabel(f"BD1 - {axis_labels[0]}")
    ax.set_ylabel(f"BD2 - {axis_labels[1]}")

    ax.set_title("MAP-Elites Grid")
    ax.set_aspect("equal")

    if directory_string is None:
        plt.show()
    else:
        plt.savefig(f"{directory_string}/{filename}.png", format="png")
    return fig, ax



if __name__ == '__main__':
    experiment_labels = [
        "20230813_01_48_TiO2_200_niches_for benchmark_100_relax_2"
    ]
    centroid_filepaths = [
        "centroids_200_2_band_gap_0_100_shear_modulus_0_120.dat",
    ]
    archive_number = 5079
    experiment_directory_path = pathlib.Path(__file__).parent.parent / ".experiment.nosync" / "experiments" /experiment_labels[0]
    centroid_full_path = pathlib.Path(__file__).parent.parent / ".experiment.nosync" / "experiments" / "centroids" / centroid_filepaths[0]

    unrelaxed_archive_location = experiment_directory_path / f"archive_{archive_number}.pkl"

    archive = Archive.from_archive(unrelaxed_archive_location, centroid_filepath=centroid_full_path)

    structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in archive.individuals]

    groups = StructureMatcher().group_structures(structures)
    ids_by_group = []
    for group in groups:
        id_in_group = []
        for el in group:
            match = [structures[i] == el for i in range(len(structures))]
            match_id = np.argwhere(np.array(match)).reshape(-1)
            id_in_group.append(archive.centroid_ids[match_id[0]])
        ids_by_group.append(id_in_group)

    all_centroids = load_centroids(centroid_full_path)


    color_indices = np.linspace(0, 1, len(ids_by_group))
    cmap = cm.get_cmap('rainbow')
    list_of_colors = []
    for color_id in color_indices:
        list_of_colors.append(cmap(color_id)[:3])


    plot_2d_groups_of_structures(
        centroids=all_centroids,
        list_of_centroid_groups=ids_by_group,
        list_of_colors=list_of_colors,
        directory_string=experiment_directory_path,
        filename="cvt_by_structure_similarity.png",
        minval=[0, 0],
        maxval= [100,120],
    )
