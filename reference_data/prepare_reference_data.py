import numpy as np
from csp_elites.reference_setup.reference_analyser import ReferenceAnalyser
from hydra import compose, initialize


def prepare_reference_datasets(system_formula):
    with initialize(version_base=None, config_path="../configs"):
        config = compose(config_name="csp", overrides=[f"system={system_formula}"])   
    
    for filter_experiment in [False, True]:

        reference_analyser = ReferenceAnalyser(
            formula=formula,
            max_n_atoms_in_cell=np.sum(np.array(config.system.atom_counts)),
            experimental_references_only=filter_experiment,
            save_plots=True,
        )
        
        # Find BD limits
        band_gap_limits, shear_moduli_limits = reference_analyser.set_bd_limits(
            reference_analyser.band_gaps, reference_analyser.shear_moduli
        )
        bd_minimum_values = np.array([band_gap_limits[0], shear_moduli_limits[0]])
        bd_maximum_values = np.array([band_gap_limits[1], shear_moduli_limits[1]])
        
        reference_analyser.return_valid_spacegroups_for_pyxtal(
            elements=config.system.elements,
            atoms_counts=config.system.atom_counts
        )
        
        # Check fitness limits
        fitness_min_values, fitness_max_values = reference_analyser.propose_fitness_limits()
        
        if config.system.fitness_min_values is not None:
            assert(fitness_min_values > list(config.system.fitness_min_values), "Need to adjust fitness min values")
            assert(fitness_max_values < list(config.system.fitness_max_values), "Need to adjust fitness max values")
            fitness_min_values = config.system.fitness_min_values
            fitness_max_values = config.system.fitness_max_values

        # Make sure blocks are the same as in the config
        blocks = reference_analyser.return_blocks_list()
        if config.system.blocks:
            assert(blocks==config.system.blocks, "Blocks in config are not the same as in the reference data")
        else:
            config.system.blocks = blocks

        
        # Create centroids for system
        _ = reference_analyser.initialise_kdt_and_centroids(
            number_of_niches=config.number_of_niches,
            band_gap_limits=band_gap_limits,
            shear_moduli_limits=shear_moduli_limits,
        )

        # Create target archive
        target_archive = reference_analyser.create_model_archive(
            bd_minimum_values=bd_minimum_values,
            bd_maximum_values=bd_maximum_values,
            save_reference=not filter_experiment,
        )

        # reference_analyser.plot_cvt_plot(
        #     target_archive=target_archive,
        #     bd_minimum_values=np.array([0, 0])
        #     if reference_analyser.normalise_bd
        #     else bd_minimum_values,
        #     bd_maximum_values=np.array([1, 1])
        #     if reference_analyser.normalise_bd
        #     else bd_maximum_values,
        #     fitness_limits=fitness_limits,
        #     x_axis_limits=bd_minimum_values,
        #     y_axis_limits=bd_maximum_values,
        # )
        # reference_analyser.plot_references_as_groups(target_archive)
        # reference_analyser.heatmap_structure_matcher_distances(annotate=False)
        # reference_analyser.plot_symmetries()
        # reference_analyser.plot_fmax()

    

if __name__ == "__main__":
    # formulas = ["C", "SiO2", "Si", "SiC", "TiO2"]
    formulas = ["SiO2"]
    
    for formula in formulas:
        prepare_reference_datasets(formula)

   