import matplotlib.pyplot as plt
import jax
import numpy as np
import os
import pandas as pd
import seaborn as sns
from typing import List, Any, Dict
from analysis_functions.pairwise_coverage_analysis import get_global_pareto_front
from functools import partial

from qdax.utils.pareto_front import compute_hypervolume, compute_hypervolume_3d

# CHANGE THESE TO ADJUST APPEARANCE OF PLOT
FIG_WIDTH = 12
FIG_HEIGHT = 10
FIGURE_DPI = 200

# ---- layout of plot ---
NUM_ROWS = 2
NUM_COLS = 3

# ---- font sizes and weights ------
BIG_GRID_FONT_SIZE  = 14 # size of labels for environment
TITLE_FONT_WEIGHT = 'bold' # Can be: ['normal' | 'bold' | 'heavy' | 'light' | 'ultrabold' | 'ultralight']
GLOBAL_TITLE_SIZE = 18 # size of big title
LEGEND_FONT_SIZE = 'x-large'

SCATTER_DOT_SIZE = 7
LEGEND_DOT_SIZE = 4

# ----- colour palettes ------
COLOUR_PALETTE = "colorblind"

# ---- spacing ----
RIGHTSPACING = 0.9   # the right side of the subplots of the figure
BOTTOMSPACING = 0.1  # the bottom of the subplots of the figure
WIDTHSPACING = 0.2  # the proportion of width reserved for blank space between subplots
HEIGHTSPACING = 0.25  # the proportion of height reserved for blank space between subplots
TOPSPACING = 0.9  # the top of the subplots of the figure



def plot_pfs(parent_dirname: str,
    env_names: List[str],
    env_dicts: Dict,
    experiment_names: List[str],
    experiment_dicts: Dict,
    num_replications: int=20,
) -> None:
    print("\n")
    print("--------------------------------------------------")
    print("Plotting PFs for each env, for each experiment")
    print("-------------------------------------------------")

    _analysis_dir = os.path.join(parent_dirname, "analysis/")
    _global_pfs_dir = os.path.join(_analysis_dir, "global_pfs/")
    _max_pfs_dir = os.path.join(_analysis_dir, "max_pfs/")
    _global_max_pfs_dir = os.path.join(_analysis_dir, "global_and_max_pfs/")

    os.makedirs(_analysis_dir, exist_ok=True)
    os.makedirs(_global_pfs_dir, exist_ok=True)
    os.makedirs(_max_pfs_dir, exist_ok=True)
    os.makedirs(_global_max_pfs_dir, exist_ok=True)

    # Calculate coverage scores for each environment
    for replication in range(num_replications):
        
        replication_global_pfs = {}
        replication_max_pfs = {}

        for env in env_names:
            print("\n")
            print(f"      REP: {replication+1}    ENV: {env}             ")

            env_dirname = os.path.join(parent_dirname, f"{env}/")


            # Calculate coverage scores for each experiment
            env_global_pfs = {}
            env_max_pfs = {}

            for experiment in experiment_names:
                if experiment not in env_dicts[env]["exceptions"]:
                    replication_name = os.listdir(os.path.join(env_dirname, experiment))[replication]
                    replication_dir = os.path.join(env_dirname, experiment, replication_name)
                    fitnesses = np.load(os.path.join(replication_dir, "repertoirefitnesses.npy"))
                    exp_rep_global_pf = get_global_pareto_front(fitnesses)
                    exp_rep_max_pf = get_max_pareto_front(fitnesses, env_dicts[env]["reference_point"])
                    if len(exp_rep_global_pf.shape) == 1:
                        exp_rep_global_pf = np.expand_dims(exp_rep_global_pf, axis=0)
                    if len(exp_rep_max_pf.shape) == 1:
                        exp_rep_max_pf = np.expand_dims(exp_rep_max_pf, axis=0)
                    env_global_pfs[experiment] = exp_rep_global_pf
                    env_max_pfs[experiment] = exp_rep_max_pf

            replication_global_pfs[env] = env_global_pfs
            replication_max_pfs[env] = env_max_pfs
                
        plot_experiments_pfs_grid(
            env_names,
            env_dicts,
            experiment_names,
            experiment_dicts,
            replication_global_pfs,
            suptitle="Global Pareto Fronts",
            replication=replication,
            save_dir=_global_pfs_dir,
        )
    
        plot_experiments_pfs_grid(
            env_names,
            env_dicts,
            experiment_names,
            experiment_dicts,
            replication_max_pfs,
            suptitle="Max Pareto Fronts",
            replication=replication,
            save_dir=_max_pfs_dir,
        )


def plot_experiments_pfs_grid(
    env_names: List[str],
    env_dicts: Dict,
    experiment_names: List[str],
    experiment_dicts: Dict,
    replication_pfs: Dict,
    suptitle: str,
    replication: int,
    save_dir: str,
) -> None:

    num_envs = len(replication_pfs.keys())
    num_exps = len(replication_pfs[list(replication_pfs.keys())[0]].keys())

    experiment_labels = []
    
    for exp_name in experiment_names:
        experiment_labels.append(experiment_dicts[exp_name]["label"])
        
    # Create color palette
    experiment_colours = sns.color_palette(COLOUR_PALETTE, len(experiment_names))
    colour_frame = pd.DataFrame(data={"Label": experiment_names, "Colour": experiment_colours})

    params = {
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.titlesize': BIG_GRID_FONT_SIZE,
        'axes.titleweight': TITLE_FONT_WEIGHT,
        'figure.dpi': FIGURE_DPI,
    }

    plt.rcParams.update(params)

    fig, ax = plt.subplots(
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        nrows=NUM_ROWS, 
        ncols=NUM_COLS,
    )

    for env_num, env_fig in enumerate(ax.ravel()):
        
        env_num_objectives = len(env_dicts[env_names[env_num]]["reward_labels"])
        
        if env_num_objectives == 3:
            env_fig.remove()
            env_fig = fig.add_subplot(NUM_ROWS, NUM_COLS, env_num+1, projection="3d")  
        
        env_fig = plot_grid_square(env_fig,
            env_dict = env_dicts[env_names[env_num]],
            experiment_names = experiment_names,
            experiment_dicts = experiment_dicts,
            env_pfs = replication_pfs[env_names[env_num]],
            colour_frame = colour_frame,
        )
        env_fig.spines["top"].set_visible(False)
        env_fig.spines["right"].set_visible(False)

    handles, labels = ax.ravel()[-1].get_legend_handles_labels()
    
        
    plt.figlegend(experiment_labels, 
        loc = 'lower center',
        ncol=len(experiment_labels), 
        fontsize=LEGEND_FONT_SIZE,
        markerscale=LEGEND_DOT_SIZE,
    )

    plt.suptitle(suptitle, 
        fontsize=GLOBAL_TITLE_SIZE,
        fontweight=TITLE_FONT_WEIGHT,
    )


    plt.subplots_adjust(
        bottom = BOTTOMSPACING,
        top = TOPSPACING,
        wspace = WIDTHSPACING,  
        hspace = HEIGHTSPACING,  

    )   
    plt.savefig(os.path.join(save_dir, f"pfs_rep_{replication+1}"), bbox_inches='tight')
    plt.close()




def plot_grid_square(
    env_ax: plt.Axes,
    env_dict: Dict,
    env_pfs: List[np.array],
    experiment_names:  List[str],
    experiment_dicts: Dict,
    colour_frame: pd.DataFrame,
):
    """
    Plots one subplot of grid
    """

    # Getting the correct color palette
    exp_palette = colour_frame["Colour"].values
    sns.set_palette(exp_palette)
    
    num_objectives = len(env_dict["reward_labels"])

    for exp_num, exp_name in enumerate(experiment_names):
        
        if exp_name not in env_dict["exceptions"]:
            if num_objectives == 2: 
                env_ax.scatter(
                    env_pfs[exp_name][:,0], # first fitnesses
                    env_pfs[exp_name][:,1], # second fitnesses
                    label=experiment_dicts[exp_name]["label"],
                    s=SCATTER_DOT_SIZE,
                    c=exp_palette[exp_num]
                )
                env_ax.set_xlabel(env_dict["reward_labels"][0])
                env_ax.set_ylabel(env_dict["reward_labels"][1])
            
            elif num_objectives == 3:
                env_ax.scatter(
                    env_pfs[exp_name][:,0], # first fitnesses
                    env_pfs[exp_name][:,1], # second fitnesses
                    env_pfs[exp_name][:,2], # third fitnesses
                    label=experiment_dicts[exp_name]["label"],
                    s=SCATTER_DOT_SIZE,
                    c=exp_palette[exp_num]
                )
                env_ax.set_xlabel(env_dict["reward_labels"][0])
                env_ax.set_ylabel(env_dict["reward_labels"][1])
                env_ax.set_zlabel(env_dict["reward_labels"][2])

            
            env_ax.set_title(env_dict["label"])

    return env_ax




def get_max_pareto_front(
    fitnesses,
    reference_point,
):

    num_objectives = fitnesses.shape[-1]

    # recompute hypervolumes
    if num_objectives == 2:
        hypervolume_function = partial(compute_hypervolume, reference_point=np.array(reference_point))
        
    elif num_objectives == 3:
        hypervolume_function = partial(compute_hypervolume_3d, reference_point=np.array(reference_point))
        
    hypervolumes = jax.vmap(hypervolume_function)(fitnesses)  # num centroids

    # mask empty hypervolumes
    repertoire_empty = fitnesses == -np.inf # num centroids x pareto-front length x num criteria
    repertoire_not_empty = np.any(~np.all(repertoire_empty, axis=-1), axis=-1) # num centroids x pareto-front length
    hypervolumes = np.where(repertoire_not_empty, hypervolumes, -np.inf) # num_centroids

    # find max hypervolume
    max_hypervolume = np.max(hypervolumes)

    # get mask for centroid with max hypervolume
    max_hypervolume_mask = hypervolumes == max_hypervolume

    # get cell fitnesses with max hypervolume
    max_cell_fitnesses = np.take(fitnesses, np.argwhere(max_hypervolume_mask), axis=0).squeeze(axis=(0,1))

    # create mask for -inf fitnesses
    non_empty_indices = np.argwhere(np.all(max_cell_fitnesses != -np.inf, axis=-1))
    pareto_front = np.take(max_cell_fitnesses, non_empty_indices, axis=0).squeeze()
    
    return pareto_front
