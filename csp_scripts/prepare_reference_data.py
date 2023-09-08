# elements_list = [["C"], ["Si", "O"], ["Si"], ["Si", "C"], ["Ti", "O"]]
# atoms_counts_list = [[24], [8, 16], [24], [12, 12], [8, 16]]
# formulas = ["C", "SiO2", "Si", "SiC", "TiO2"]

elements_list = [["Ti", "O"]]
atoms_counts_list = [[8, 16]]
formulas = ["TiO2"]
fitness_limits = [8.7, 9.5]
band_gap_limits = [0, 4]
shear_moduli_limits = [0, 120]
reference_data_dump = []
dict_summary = {}
for filter_experiment in [True]:
    # for filter_experiment in [False, True]:
    filter_experiment_dump = []
    for i, formula in enumerate(formulas):
        reference_analyser = ReferenceAnalyser(
            formula=formula,
            max_n_atoms_in_cell=np.sum(np.array(atoms_counts_list[i])),
            experimental_references_only=filter_experiment,
            save_plots=True
        )
        # print(formula)
        # band_gap_limits, shear_moduli_limits = reference_analyser.set_bd_limits(
        #     reference_analyser.band_gaps, reference_analyser.shear_moduli)

        reference_analyser.return_valid_spacegroups_for_pyxtal(elements=elements_list[i],
                                                               atoms_counts=atoms_counts_list[i])
        kdt = reference_analyser.initialise_kdt_and_centroids(
            number_of_niches=200,
            band_gap_limits=band_gap_limits,
            shear_moduli_limits=shear_moduli_limits,
        )
        if fitness_limits is None:
            fitness_limits = reference_analyser.propose_fitness_limits()
        bd_minimum_values = np.array([band_gap_limits[0], shear_moduli_limits[0]])
        bd_maximum_values = np.array([band_gap_limits[1], shear_moduli_limits[1]])
        reference_analyser.write_base_config(
            bd_minimum_values=bd_minimum_values.tolist(),
            bd_maximum_values=bd_maximum_values.tolist(),
            fitness_limits=fitness_limits,
        )
        target_archive = reference_analyser.create_model_archive(
            bd_minimum_values=bd_minimum_values,
            bd_maximum_values=bd_maximum_values,
            save_reference=not filter_experiment,
        )

        # normalise_bd_values = (bd_minimum_values, bd_maximum_values) if reference_analyser.normalise_bd else None
        # reference_analyser.plot_cvt_plot(
        #     target_archive=target_archive,
        #     bd_minimum_values=np.array([0, 0]) if reference_analyser.normalise_bd else bd_minimum_values,
        #     bd_maximum_values=np.array([1, 1]) if reference_analyser.normalise_bd else bd_maximum_values,
        #     fitness_limits=fitness_limits,
        #     x_axis_limits=bd_minimum_values,
        #     y_axis_limits=bd_maximum_values,
        # )
        # reference_analyser.plot_references_as_groups(target_archive)
        # reference_analyser.heatmap_structure_matcher_distances(annotate=False)
        # reference_analyser.plot_symmetries()
        reference_analyser.plot_fmax()
        # dict_summary[f"{formula}_{filter_experiment}"] = len(reference_analyser.structures_to_consider)
        # print(dict_summary)

        # filter_experiment_dump.append()
# with open("../../.experiment.nosync/mp_reference_analysis/dict_summary.json", "w") as file:
#     json.dump(dict_summary, file)
