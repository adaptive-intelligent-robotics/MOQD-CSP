import os
import pathlib
import pickle
from typing import Optional, Tuple, List, Dict, TYPE_CHECKING, Union

import imageio
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Voronoi
from sklearn.neighbors import KDTree
from tqdm import tqdm

import matplotlib.colors as mcolors
from csp_elites.map_elites.elites_utils import make_hashable
from csp_elites.utils.asign_target_values_to_centroids import (
    reassign_data_from_pkl_to_new_centroids,
)

if TYPE_CHECKING:
    from csp_elites.utils.experiment_parameters import ExperimentParameters

import scienceplots

plt.style.use("science")
plt.rcParams["savefig.dpi"] = 300


def get_voronoi_finite_polygons_2d(
    centroids: np.ndarray, radius: Optional[float] = None
) -> Tuple[List, np.ndarray]:
    """COPIED FROM QDAX
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions."""
    voronoi_diagram = Voronoi(centroids)
    if voronoi_diagram.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = voronoi_diagram.vertices.tolist()

    center = voronoi_diagram.points.mean(axis=0)
    if radius is None:
        radius = voronoi_diagram.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges: Dict[np.ndarray, np.ndarray] = {}
    for (p1, p2), (v1, v2) in zip(
        voronoi_diagram.ridge_points, voronoi_diagram.ridge_vertices
    ):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(voronoi_diagram.point_region):
        vertices = voronoi_diagram.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge
            t = voronoi_diagram.points[p2] - voronoi_diagram.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = voronoi_diagram.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = voronoi_diagram.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def plot_2d_map_elites_repertoire_marta(
    centroids: np.ndarray,
    repertoire_fitnesses: np.ndarray,
    minval: np.ndarray,
    maxval: np.ndarray,
    repertoire_descriptors: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    target_centroids: Optional[np.ndarray] = None,
    directory_string: Optional[str] = None,
    filename: Optional[str] = "cvt_plot",
    axis_labels: List[str] = ["Band Gap, eV", "Shear Modulus, GPa"],
    annotations: Optional[Union[List[str], np.ndarray]] = None,
    annotate: bool = True,
    x_axis_limits: Optional[Tuple[float, float]] = None,
    y_axis_limits: Optional[Tuple[float, float]] = None,
) -> Tuple[Optional[Figure], Axes]:
    """Function adapted from QDAX
    Plot a visual representation of a 2d map elites repertoire.

    function is very specific to repertoires.

    Args:
        centroids: the centroids of the repertoire
        repertoire_fitnesses: the fitness of the repertoire
        minval: minimum values for the descritors
        maxval: maximum values for the descriptors
        repertoire_descriptors: the descriptors. Defaults to None.
        ax: a matplotlib axe for the figure to plot. Defaults to None.
        vmin: minimum value for the fitness. Defaults to None. If not given,
            the value will be set to the minimum fitness in the repertoire.
        vmax: maximum value for the fitness. Defaults to None. If not given,
            the value will be set to the maximum fitness in the repertoire.

    Raises:
        NotImplementedError: does not work for descriptors dimension different
        from 2.

    Returns:
        A figure and axes object, corresponding to the visualisation of the
        repertoire.
    """
    grid_empty = repertoire_fitnesses == -np.inf
    num_descriptors = centroids.shape[1]
    if num_descriptors != 2:
        raise NotImplementedError("Grid plot supports 2 descriptors only for now.")

    my_cmap = cm.viridis

    fitnesses = repertoire_fitnesses
    if vmin is None:
        vmin = float(np.min(fitnesses[~grid_empty]))
    if vmax is None:
        vmax = float(np.max(fitnesses[~grid_empty]))

    params = {"figure.figsize": [3.5, 3.5]}
    mpl.rcParams.update(params)

    # create the plot object
    fig = None
    if ax is None:
        fig, ax = plt.subplots(facecolor="white", edgecolor="white")

    assert (
        len(np.array(minval).shape) < 2
    ), f"minval : {minval} should be float or couple of floats"
    assert (
        len(np.array(maxval).shape) < 2
    ), f"maxval : {maxval} should be float or couple of floats"

    if len(np.array(minval).shape) == 0 and len(np.array(maxval).shape) == 0:
        ax.set_xlim(minval, maxval)
        ax.set_ylim(minval, maxval)
    else:
        ax.set_xlim(minval[0], maxval[0])
        ax.set_ylim(minval[1], maxval[1])

    ax.set(adjustable="box", aspect="equal")

    # create the regions and vertices from centroids
    regions, vertices = get_voronoi_finite_polygons_2d(centroids)

    norm = Normalize(vmin=vmin, vmax=vmax)

    # fill the plot with contours
    for i, region in enumerate(regions):
        polygon = vertices[region]
        ax.fill(*zip(*polygon), alpha=0.05, edgecolor="black", facecolor="white", lw=1)
        if target_centroids is not None:
            if centroids[i] in np.array(target_centroids):
                ax.fill(
                    *zip(*polygon),
                    edgecolor=mcolors.CSS4_COLORS["salmon"],
                    facecolor="none",
                    lw=1,
                )
    # fill the plot with the colors
    for idx, fitness in enumerate(fitnesses):
        if fitness > -np.inf:
            region = regions[idx]
            polygon = vertices[region]
            ax.fill(*zip(*polygon), alpha=0.8, color=my_cmap(norm(fitness)))
            # if target_centroids is not None:
            #     if centroids[idx] in np.array(target_centroids):
            #         ax.fill(*zip(*polygon), edgecolor="orange", facecolor="none", lw=2, alpha=0.8)

    for i, region in enumerate(regions):
        polygon = vertices[region]
        if target_centroids is not None:
            if centroids[i] in np.array(target_centroids):
                ax.fill(
                    *zip(*polygon),
                    edgecolor=mcolors.CSS4_COLORS["salmon"],
                    facecolor="none",
                    lw=1,
                    alpha=1,
                )

    np.set_printoptions(2)
    # if descriptors are specified, add points location
    if repertoire_descriptors is not None:
        descriptors = repertoire_descriptors[~grid_empty]
        ax.scatter(
            descriptors[:, 0],
            descriptors[:, 1],
            c="black",
            # c=fitnesses[~grid_empty],
            # cmap=my_cmap,
            s=1,
            zorder=0,
        )
        for i in range(len(fitnesses)):
            if annotate:
                if annotations is None:
                    annotations = np.around(fitnesses, decimals=3)
                if isinstance(annotations[i], str) and annotations[i] != "-inf":
                    ax.annotate(annotations[i], (centroids[i, 0], centroids[i, 1]))
                elif isinstance(annotations[i], float) and annotations[i] != -np.inf:
                    ax.annotate(
                        annotations[i], (centroids[i, 0], centroids[i, 1]), fontsize=4
                    )
    # aesthetic
    if x_axis_limits is not None and y_axis_limits is not None:
        x_tick_labels = np.linspace(x_axis_limits[0], x_axis_limits[1], 6)
        y_tick_labels = np.linspace(y_axis_limits[0], y_axis_limits[1], 6)
        ax.set_xticklabels([np.around(el, 1) for el in x_tick_labels])
        ax.set_yticklabels([np.around(el, 1) for el in y_tick_labels])

    ax.set_xlabel(f"{axis_labels[0]}")
    ax.set_ylabel(f"{axis_labels[1]}")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=my_cmap), cax=cax)
    cbar.ax.tick_params(labelsize=mpl.rcParams["font.size"])

    ax.set_title("MAP-Elites Grid")
    ax.set_aspect("equal")

    if directory_string is None:
        plt.show()
    else:
        plt.savefig(f"{directory_string}/{filename}.png", format="png")
    return fig, ax


def plot_numbered_centroids(
    centroids: np.ndarray,
    minval: np.ndarray,
    maxval: np.ndarray,
):
    params = {"figure.figsize": [3.5, 3.5]}
    mpl.rcParams.update(params)

    fig, ax = plt.subplots(facecolor="white", edgecolor="white")

    assert (
        len(np.array(minval).shape) < 2
    ), f"minval : {minval} should be float or couple of floats"
    assert (
        len(np.array(maxval).shape) < 2
    ), f"maxval : {maxval} should be float or couple of floats"

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
        ax.annotate(str(i), (centroids[i, 0], centroids[i, 1]))
    # aesthetic
    ax.set_xlabel(f"BD1")
    ax.set_ylabel(f"BD2")
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)

    ax.set_title("MAP-Elites Grid")
    ax.set_aspect("equal")
    fig.show()


def plot_all_statistics_from_file(filename: str, save_location: Optional[str]):
    params = {"figure.figsize": [3.5, 2.625]}
    mpl.rcParams.update(params)
    with open(filename, "r") as file:
        generation_data = np.loadtxt(file)

    generation_data = generation_data.T

    number_of_metrics = len(generation_data)
    if number_of_metrics == 7:
        metric_names = [
            "Evaluation number",
            "Archive size",
            "Maximum Fitness",
            "Mean Fitness",
            "Median Fitness",
            "Fitness 5th Percentile",
            "Fitness 95th Percentile",
        ]
    elif number_of_metrics == 9:
        metric_names = [
            "Evaluation number",
            "Archive size",
            "Maximum Fitness",
            "Mean Fitness",
            "Median Fitness",
            "Fitness 5th Percentile",
            "Fitness 95th Percentile",
            "Coverage",
            "QD score",
        ]
    else:
        raise ValueError("unknown metric present in log file, check, number of columns")

    for metric_id in range(1, number_of_metrics):
        fig, ax = plt.subplots()
        ax.plot(generation_data[0], generation_data[metric_id])
        ax.set_xlabel("Evaluation Count")
        ax.set_ylabel(metric_names[metric_id])
        if save_location is None or save_location == "":
            plt.show()
        else:
            file_tag = metric_names[metric_id].replace(" ", "")
            plt.savefig(f"{save_location}stats_{file_tag}")


def load_centroids(filename: str) -> np.ndarray:
    with open(filename, "r") as file:
        centroids = np.loadtxt(file)
    return centroids


def load_archive_from_pickle(filename: str):
    with open(filename, "rb") as file:
        archive = pickle.load(file)

    fitnesses = []
    centroids = []
    descriptors = []
    individuals = []
    for el in archive:
        fitnesses.append(el[0])
        centroids.append(list(el[1]))

        descriptors.append(list(el[2]))
        individuals.append(el[3])

    fitnesses = np.array(fitnesses)
    centroids = np.array(centroids)
    descriptors = np.array(descriptors)

    return fitnesses, centroids, descriptors, individuals


def convert_fitness_and_descriptors_to_plotting_format(
    all_centroids: np.ndarray,
    centroids_from_archive: np.ndarray,
    fitnesses_from_archive: np.ndarray,
    descriptors_from_archive: np.ndarray,
):
    fitness_for_plotting = np.full((len(all_centroids)), -np.inf)
    descriptors_for_plotting = np.full(
        (len(all_centroids), len(descriptors_from_archive[0])), -np.inf
    )
    for i in range(len(centroids_from_archive)):
        present_centroid = np.argwhere(all_centroids == centroids_from_archive[i])
        fitness_for_plotting[present_centroid[0][0]] = fitnesses_from_archive[i]
        descriptors_for_plotting[present_centroid[0][0]] = descriptors_from_archive[i]

    return fitness_for_plotting, descriptors_for_plotting


def plot_all_maps_in_archive(
    experiment_directory_path: str,
    experiment_parameters: "ExperimentParameters",
    all_centroids,
    target_centroids,
    annotate: bool = True,
    force_replot: bool = False,
):
    list_of_files = [
        name
        for name in os.listdir(f"{experiment_directory_path}")
        if not os.path.isdir(name)
    ]
    list_of_archives = [
        filename
        for filename in list_of_files
        if ("archive_" in filename) and (".pkl" in filename)
    ]
    list_of_plots = [
        filename
        for filename in list_of_files
        if ("cvt_plot" in filename) and (".png" in filename)
    ]
    list_of_plot_ids = [
        filename.lstrip("cvt_plot_").rstrip(".png") for filename in list_of_plots
    ]

    for filename in tqdm(list_of_archives):
        if "relaxed_archive" in filename:
            continue
        archive_id = filename.lstrip("relaxed_archive_").rstrip(".pkl")
        if force_replot or (archive_id not in list_of_plot_ids):
            fitnesses, centroids, descriptors, individuals = load_archive_from_pickle(
                f"{experiment_directory_path}/{filename}"
            )
            (
                fitnesses_for_plotting,
                descriptors_for_plotting,
            ) = convert_fitness_and_descriptors_to_plotting_format(
                all_centroids=all_centroids,
                centroids_from_archive=centroids,
                fitnesses_from_archive=fitnesses,
                descriptors_from_archive=descriptors,
            )
            if "relaxed" in filename:
                archive_id += "_relaxed"
            (
                bd_minimum_values,
                bd_maximum_values,
            ) = experiment_parameters.return_min_max_bd_values()
            plot_2d_map_elites_repertoire_marta(
                centroids=all_centroids,
                repertoire_fitnesses=fitnesses_for_plotting,
                minval=bd_minimum_values,
                maxval=bd_maximum_values,
                # minval=[0, 0] if experiment_parameters.cvt_run_parameters["normalise_bd"] else experiment_parameters.cvt_run_parameters["bd_minimum_values"],
                # maxval=[1, 1] if experiment_parameters.cvt_run_parameters["normalise_bd"] else experiment_parameters.cvt_run_parameters["bd_maximum_values"],
                repertoire_descriptors=descriptors_for_plotting,
                vmin=experiment_parameters.fitness_min_max_values[0],
                vmax=experiment_parameters.fitness_min_max_values[1],
                target_centroids=target_centroids,
                directory_string=experiment_directory_path,
                filename=f"cvt_plot_{archive_id}",
                axis_labels=["Band Gap, eV", "Shear Modulus, GPa"],
                annotate=annotate,
                x_axis_limits=(
                    experiment_parameters.cvt_run_parameters["bd_minimum_values"][0],
                    experiment_parameters.cvt_run_parameters["bd_maximum_values"][0],
                ),
                y_axis_limits=(
                    experiment_parameters.cvt_run_parameters["bd_minimum_values"][1],
                    experiment_parameters.cvt_run_parameters["bd_maximum_values"][1],
                ),
            )


def plot_gif(experiment_directory_path: str):
    plot_list = [
        name
        for name in os.listdir(f"{experiment_directory_path}")
        if not os.path.isdir(name) and "cvt_plot_" in name and ".png" in name
    ]
    sorted_plot_list = sorted(
        plot_list, key=lambda x: int(x.lstrip("cvt_plot_").rstrip(".png"))
    )

    frames = []
    for plot_name in sorted_plot_list:
        image = imageio.v2.imread(f"{experiment_directory_path}/{plot_name}")
        frames.append(image)

    imageio.mimsave(
        f"{experiment_directory_path}/cvt_plot_gif.gif",  # output gif
        frames,
    )  # array of input frames)
