from analysis_functions.run_analysis import MOQD_Analysis


# Parent directory of results
parent_dirname = "results/"


# Directory names of experiments
experiment_names = [
    # main
    "biased_mome",

    # baselines
    # "map_elites_energy",
    # "map_elites_magmom",
    "mome",

]

# Directory names of environments
env_names=[
    # "C",
    # "S",
    "SiC",
    "SiO2",
    # "TiO2",
]


experiment_dicts = {
    
    ## MAIN
    
    "biased_mome": {
        "label": "MOME-Crowding",
        "emitter_names": [],
        "emitter_labels": [],
        "grid_plot_linestyle": "solid",
    },
    
    ## BASELINES
    
    "mome": {
        "label": "MOME-PGX",
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
}




env_dicts = {
    "C": {
        "label": "C",
        "reward_labels": ["Energy", "Magmom"],
        "reference_point": [0, 0],
        "exceptions": [],
    },
    
    "S": {
        "label": "S",
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
        num_replications=4,
        num_iterations=250,
        episode_length=1000,
        batch_size=20
    )
    
    # Metrics to plot in grid plot
    grid_plot_metrics_list = [
        "moqd_score", 
        "energy_qd_score",
        "magmom_qd_score",
        # "max_energy_fitness",
        # "max_magmom_fitness",
        # "global_hypervolume", 
        # "max_sum_scores",
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
        "coverage": "Coverage",
    }


    analysis_helper.plot_grid(
        grid_plot_metrics_list,
        grid_plot_metrics_labels,
    )

    grid_plot_metrics_list = [
        "moqd_score", 
        "max_energy_fitness",
        "max_magmom_fitness",
        # "global_hypervolume", 
        # "max_sum_scores",
        # "coverage"
    ]

    analysis_helper.plot_grid(
        grid_plot_metrics_list,
        grid_plot_metrics_labels,
    )

    analysis_helper.calculate_wilcoxon(
        p_value_metrics_list
    )
    
    analysis_helper.sparsity_analysis()

    # analysis_helper.plot_final_pfs()
