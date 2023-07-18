import numpy as np
from ase.ga.ofp_comparator import OFPComparator
from megnet.utils.models import load_model
from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor

from csp_elites.crystal.crystal_evaluator import CrystalEvaluator

if __name__ == '__main__':
    formula = "TiO2"
    blocks =  [22] * 8 + [8] * 16,
    comparator = OFPComparator(n_top=len(blocks), dE=1.0,
                               cos_dist_max=1e-3, rcut=10., binwidth=0.05,
                               pbc=[True, True, True], sigma=0.05, nsigma=4,
                               recalculate=False)

    crystal_evaluator = CrystalEvaluator(comparator=comparator)

    with MPRester(api_key="4nB757V2Puue49BqPnP3bjRPksr4J9y0") as mpr:
        docs = mpr.summary.search(formula=formula,
                                  # band_gap=(0.5, 1.0),
                                  fields=["material_id",
                                          "band_gap",
                                          "volume", "lattice", "formation_energy_per_atom",
                                          "energy_above_hull",
                                          "structure",
                                          'k_voigt', 'k_reuss', 'k_vrh', 'g_voigt', 'g_reuss',
                                          'g_vrh'
                                          ])

    # load a model in megnet.utils.models.AVAILABLE_MODELS
    model = load_model("logG_MP_2018")

    energies = []
    predicted_gs = []
    predicted_band_gaps = []
    queried_gs = []
    all_individuals = []
    errors = []

    for mp_structure in docs:
        structure = mp_structure.structure
        predicted_G = crystal_evaluator.compute_shear_modulus(structure)
        predicted_band_gap = crystal_evaluator.compute_band_gap(structure)
        predicted_band_gaps.append(predicted_band_gap)

        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.calc = None
        all_individuals.append(atoms)
        energy = crystal_evaluator.compute_energy(atoms, really_relax=False)
        energies.append(energy[0])
        # predicted_G = 10 ** model.predict_structure(structure).ravel()[0]
        predicted_gs.append(predicted_G)
        print(f'The predicted K for {structure.composition.reduced_formula} is {predicted_G:.0f} GPa.')
        queried_shear_modulus = mp_structure.g_vrh

        if queried_shear_modulus is None:
            print(predicted_G)
        else:
            queried_gs.append(queried_shear_modulus)
            print(f"Error {(queried_shear_modulus - predicted_G) / queried_shear_modulus}")
            error = (queried_shear_modulus - predicted_G) / queried_shear_modulus
            errors.append(error)

    errors = np.array(errors)

    print(f"Mean error {errors.mean() * 100} %")
    print(f"Error {errors.std() * 100} %")
    print()
