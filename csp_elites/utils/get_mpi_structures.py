import pickle
from typing import List

from ase import Atoms
from ase.cell import Cell
from ase.ga.utilities import CellBounds
from mp_api.client import MPRester
from pymatgen.core import Structure
from tqdm import tqdm

from csp_elites.crystal.crystal_evaluator import CrystalEvaluator


# Querying data in materi als project https://docs.materialsproject.org/downloading-data/using-the-api/querying-data

def return_structure_information_from_mp_api(structure_ids: List[str]):
    crystal_information = {}

    with MPRester(api_key="4nB757V2Puue49BqPnP3bjRPksr4J9y0") as mpr:
        data = mpr.materials.search(material_ids=structure_ids)
        formation_energies = mpr.thermo.search(material_ids=structure_ids)
        energies = mpr.summary.search(material_ids=structure_ids, fields=["uncorrected_energy_per_atom", "energy_above_hull"])


    for i, material_id in enumerate(structure_ids):
        band_structure = mpr.get_bandstructure_by_material_id(material_id)

        structure_dict = data[i].structure.as_dict()

        list_of_atoms = []
        list_of_positions = []
        for element in structure_dict["sites"]:
            list_of_atoms.append(element["species"][0]["element"])
            list_of_positions.append(element["xyz"])

        crystal_information[material_id] = {
            "formation_energy": max([el.formation_energy_per_atom for el in formation_energies]), # TODO: is taking the maximum of the approximations correct?
            "band_gap": band_structure.get_band_gap()["energy"],
            "uncorrected_energy_per_atom": energies[i].uncorrected_energy_per_atom,
            "energy_above_hull": energies[i].energy_above_hull,
            "atoms_object": Atoms(list_of_atoms, pbc= [True, True, True], cell=Cell.new([structure_dict["lattice"]["a"],
                structure_dict["lattice"]["b"],
                structure_dict["lattice"]["c"],
                structure_dict["lattice"]["alpha"],
                structure_dict["lattice"]["beta"],
                structure_dict["lattice"]["gamma"]]), positions=list_of_positions)
        }

    return crystal_information


def get_all_materials_with_formula(formula: str):
    with MPRester(api_key="4nB757V2Puue49BqPnP3bjRPksr4J9y0") as mpr:
        docs = mpr.summary.search(formula=formula,
                                  # band_gap=(0.5, 1.0),
                                  fields=["material_id",
                                          "band_gap",
                                          "volume", "lattice", "formation_energy_per_atom", "energy_above_hull",
                                          "structure", "energy_per_atom", "uncorrected_energy_per_atom",
                                          "theoretical"
                                          ])

    atoms = []
    for material in docs:
        new_atoms = convert_structure_to_atoms(material.structure)
        atoms.append(new_atoms)
    return docs, atoms

def convert_structure_to_atoms(structure: Structure) -> Atoms:
    structure_dict = structure.as_dict()
    list_of_atoms = []
    list_of_positions = []
    for element in structure_dict["sites"]:
        list_of_atoms.append(element["species"][0]["element"])
        list_of_positions.append(element["xyz"])

    atoms = Atoms(
        symbols=list_of_atoms,
        pbc=[True, True, True],
        cell=Cell.new([structure_dict["lattice"]["a"],
                     structure_dict["lattice"]["b"],
                     structure_dict["lattice"]["c"],
                     structure_dict["lattice"]["alpha"],
                     structure_dict["lattice"]["beta"],
                     structure_dict["lattice"]["gamma"]]),
        positions=list_of_positions,
    )
    return atoms


def get_all_and_model_from_formula(formula: str):
    docs, atom_objects = get_all_materials_with_formula(formula=formula)

    target_fitness = []
    target_band_gap = []
    target_formation_energy = []
    evaluator = CrystalEvaluator()
    cellbounds = CellBounds(bounds={'phi': [20, 160], 'chi': [20, 160],
                                    'psi': [20, 160], 'a': [2, 60],
                                    'b': [2, 60], 'c': [2, 60]})
    for i in tqdm(range(len(atom_objects))):
        # print(i)
        fitness, form_energy_and_band_gap, _ = evaluator.compute_fitness_and_bd(atoms=atom_objects[i], cellbounds=cellbounds, populations=None)
        print(fitness)
        print(form_energy_and_band_gap)
        target_fitness.append(fitness)
        target_formation_energy.append(form_energy_and_band_gap[0])
        target_band_gap.append(form_energy_and_band_gap[1])

    with open("tio2_target_data.pkl", "wb") as file:
        pickle.dump([target_fitness, target_band_gap, target_formation_energy], file)

if __name__ == '__main__':
    docs, atom_objects = get_all_materials_with_formula("TiO2")
    print()
