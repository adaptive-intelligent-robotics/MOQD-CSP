from analysis_functions.run_analysis import MOQD_Analysis


# Parent directory of results
parent_dirname = "results/"


# Directory names of experiments
experiment_names = [
    # main
    # "mome"
    # "mome_biased",
    "mome_biased",
    "mome_biased_addition",
    # "mome_biased_selection",

    # baselines
    "map_elites_energy",
    # "map_elites_magmom",
    # "map_elites_sum",

]

# Directory names of environments
env_names=[
    # "C",
    # "S",
    # "SiC",
    # "SiO2",
    # "TiO2",
    "SiC",
    "SiC",
    "SiC",
    "SiC",
    "SiC",
]


experiment_dicts = {
    
    ## MAIN
    
    "mome_biased": {
        "label": "MOME-Crowding",
        "emitter_names": [],
        "emitter_labels": [],
        "grid_plot_linestyle": "solid",
    },

    "mome_biased_selection": {
        "label": "MOME-Crowding-Selection",
        "emitter_names": [],
        "emitter_labels": [],
        "grid_plot_linestyle": "solid",
    },

    "mome_biased_addition": {
        "label": "MOME-Crowding-Addition",
        "emitter_names": [],
        "emitter_labels": [],
        "grid_plot_linestyle": "solid",
    },
    
    ## BASELINES
    
    "mome": {
        "label": "MOME",
        "emitter_names": [],
        "emitter_labels": [],
        "grid_plot_linestyle": "dashed", 
    },
    
    "map_elites_energy": {
        "label": "MAP-Elites (Energy)",
        "emitter_names": [],
        "emitter_labels": [],
        "grid_plot_linestyle": "dashdot"
    },
    
    "map_elites_magmom": {
        "label": "MAP-Elites (Magmom)",
        "emitter_names": [],
        "emitter_labels": [],
        "grid_plot_linestyle": "dotted"
    },
    
    "map_elites_energy": {
        "label": "MAP-Elites (Sum)",
        "emitter_names": [],
        "emitter_labels": [],
        "grid_plot_linestyle": "dashdot"
    },
}




env_dicts = {
    "C": {
        "label": "C",
        "reward_labels": ["Energy", "Magmom"],
        "reference_point": [0, 0],
        "exceptions": [],
    },
    
    "Si": {
        "label": "Si",
        "reward_labels": ["Energy", "Magmom"],
        "reference_point": [0, 0],
        "exceptions": [],
    },
    "SiC": {
        "label": "SiC",
        "reward_labels": ["Energy", "Magmom"],
        "reference_point": [0, 0],
        "exceptions": [],
    },
    "SiO2": {
        "label": "SiO2",
        "reward_labels": ["Energy", "Magmom"],
        "reference_point": [0, 0],
        "exceptions": [],
    },
    "TiO2": {
        "label": "TiO2",
        "reward_labels": ["Energy", "Magmom"],
        "reference_point": [0, 0],
        "exceptions": [],
    },
}





# List of metrics to calculate p-values for
p_value_metrics_list = [
    "moqd_score",
]

# Which algorithms to compare data-efficiency and which metric for comparison
data_efficiency_params={}


if __name__ == "__main__":
    
    
    analysis_helper = MOQD_Analysis(
        parent_dirname=parent_dirname,
        env_names=env_names,
        env_dicts=env_dicts,
        experiment_names=experiment_names,
        experiment_dicts=experiment_dicts,
        num_replications=5,
        num_iterations=50,
        episode_length=0,
        batch_size=100
    )
    
    # Metrics to plot in grid plot
    grid_plot_metrics_list = [
        "moqd_score", 
        # "energy_qd_score",
        # "magmom_qd_score",
        "max_energy_fitness",
        "max_magmom_fitness",
        "global_hypervolume", 
        # "max_sum_scores",
        "num_solutions",
        # "coverage"
    ]

    grid_plot_metrics_labels = {
        "moqd_score": "MOQD Score",
        "energy_qd_score": "Energy QD Score",
        "magmom_qd_score": "Magmom QD Score",
        "max_energy_fitness": "Max Energy Fitness",
        "max_magmom_fitness": "Max Magmom Fitness",
        "global_hypervolume": "Global Hypervolume", 
        "max_sum_scores": "Max Sum Scores",
        "num_solutions": "Number of Solutions",
        "coverage": "Coverage",
    }


    analysis_helper.plot_grid(
        grid_plot_metrics_list,
        grid_plot_metrics_labels,
    )

    grid_plot_metrics_list = [
        "moqd_score", 
        # "max_energy_fitness",
        # "max_magmom_fitness",
        # "num_solutions",
        # "global_hypervolume", 
        # "max_sum_scores",
        # "coverage"
    ]

    analysis_helper.plot_grid(
        grid_plot_metrics_list,
        grid_plot_metrics_labels,
        x_axis_evaluations=True
    )

    analysis_helper.calculate_wilcoxon(
        p_value_metrics_list
    )
    
    # analysis_helper.sparsity_analysis()

    # analysis_helper.plot_final_pfs()
