from ase.ga.ofp_comparator import OFPComparator
from matplotlib import pyplot as plt
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from ase.utils.structure_comparator import SymmetryEquivalenceCheck


from csp_elites.utils.get_mpi_structures import get_all_materials_with_formula
from csp_elites.utils.plot import load_archive_from_pickle

if __name__ == '__main__':
    fitnesses, centroids, descriptors, individuals = load_archive_from_pickle("../csp_experiments/experiments/20230707_13_21_TiO2_5050_relaxation/archive_200.pkl")

    # for individual in individuals:
    #     print(individual.get_volume())

    structure_info, known_structures = get_all_materials_with_formula("TiO2")

    number_of_atoms = 24

    structures_for_comparison = []

    for i, structure in enumerate(known_structures):
        structure = AseAtomsAdaptor.get_structure(structure)
        conventional_structure = SpacegroupAnalyzer(structure=structure).get_refined_structure()
        print(len(structure.atomic_numbers))
        print(len(conventional_structure.atomic_numbers))
        structures_for_comparison.append(conventional_structure)
        # if len(structure.get_atomic_numbers()) == number_of_atoms:
        #     structures_for_comparison.append(structure)


    comp = SymmetryEquivalenceCheck()
    comparator = OFPComparator(n_top=48, dE=1.0,
                         cos_dist_max=1e-3, rcut=10., binwidth=0.05,
                         pbc=[True, True, True], sigma=0.05, nsigma=4,
                         recalculate=False)

    # for structure in structures_for_comparison:
    #     scores_for_target = []
    #     for second_structure in structures_for_comparison:
    #         cosine_distance = comparator._compare_structure(structure, second_structure)
    #         scores_for_target.append(cosine_distance)

    scores = []

    for target_structure in structures_for_comparison:
        target_atoms =AseAtomsAdaptor.get_atoms(target_structure)
        scores_for_target = []
        for generated_structure in individuals:
            structure = AseAtomsAdaptor.get_structure(generated_structure)
            conventional_struct = SpacegroupAnalyzer(structure=structure).get_refined_structure()
            conventional_struct_as_atoms = AseAtomsAdaptor.get_atoms(conventional_struct)
            cosine_distance = comparator._compare_structure(conventional_struct_as_atoms, target_atoms)
            scores_for_target.append(cosine_distance)
            if cosine_distance <= 0.15:
                print(f"cosine distance {cosine_distance}")
                print(f"target structure params {target_structure.get_cell_lengths_and_angles()}")
                print(f"predicted structure params {generated_structure.get_cell_lengths_and_angles()}")

        scores.append(scores_for_target)
        plt.hist(scores_for_target, range=(0, 0.5), bins=20)
        plt.show()
    # print(scores)
    print()
