from ase import Atoms
from ase.ga.ofp_comparator import OFPComparator
from chgnet.model import CHGNet
from matplotlib import pyplot as plt
from pymatgen.io.ase import AseAtomsAdaptor

from csp_elites.utils.get_mpi_structures import get_all_materials_with_formula
from csp_elites.utils.plot import load_archive_from_pickle

if __name__ == '__main__':
    fitnesses, centroids, descriptors, individuals = load_archive_from_pickle("../experiment/experiments/20230727_03_43_TiO2_test/archive_436020.pkl")
    structure_info, known_structures = get_all_materials_with_formula("TiO2")
    experimentally_observed = [structure_info[i] for i in range(len(structure_info)) if not structure_info[i].theoretical]

    number_of_atoms = 24
    #
    # structures_for_comparison = []
    #
    # for structure in known_structures:
    #     if len(structure.get_atomic_numbers()) == number_of_atoms:
    #         structures_for_comparison.append(structure)

    chgnet = CHGNet.load()

    computed_energies = []
    for structure in experimentally_observed:
        prediction = chgnet.predict_structure(structure.structure)
        energy = prediction["e"] * -1
        computed_energies.append(energy)

    plt.hist(fitnesses, range=(8,10), bins=50, label="predicted fitnesses", alpha=0.9)
    plt.hist(computed_energies, range=(8,10), bins=50, label="energies from MP", alpha=0.9)
    plt.legend()
    plt.xlabel("Energy eV/atom")
    plt.ylabel("Individual count")
    plt.show()

    comparator = OFPComparator(n_top=number_of_atoms, dE=1.0,
                         cos_dist_max=1e-1, rcut=10., binwidth=0.05,
                         pbc=[True, True, True], sigma=0.05, nsigma=4,
                         recalculate=False)

    # for structure in structures_for_comparison:
    #     scores_for_target = []
    #     for second_structure in structures_for_comparison:
    #         cosine_distance = comparator._compare_structure(structure, second_structure)
    #         scores_for_target.append(cosine_distance)


    scores = []
    
    individuals = [Atoms.fromdict(atoms) for atoms in individuals]
    # individuals = [AseAtomsAdaptor.get_structure(atoms) for atoms in individuals]
    
    for struct in experimentally_observed:
        target_structure = struct.structure
        if len(target_structure.species) != 24:
            multiplier = 24 / len(target_structure.species)
            target_structure = target_structure * (1, 1, multiplier)
            target_structure = AseAtomsAdaptor.get_atoms(target_structure)
            # try:
            #     target_structure = target_structure * (1, 1, multiplier)
            #     target_structure = AseAtomsAdaptor.get_atoms(target_structure)
            #     print(len(target_structure.species))
            # except:
            #     print("error")
            #     continue
        scores_for_target = []
        for generated_structure in individuals:
            if len(target_structure) != len(generated_structure):
                continue
            else:
                cosine_distance = comparator._compare_structure(generated_structure, target_structure)
                scores_for_target.append(cosine_distance)
                if cosine_distance <= 0.15:
                    print(f"cosine distance {cosine_distance}")
                    print(f"target structure params {target_structure.get_cell_lengths_and_angles()}")
                    print(f"predicted structure params {generated_structure.get_cell_lengths_and_angles()}")

        scores.append(scores_for_target)
        plt.hist(scores_for_target, range=(0, 0.5), bins=100)
        plt.show()
    print(scores)
    print()
