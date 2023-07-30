import warnings
from typing import Optional, List, Dict

import matgl
import numpy as np
import torch
from ase import Atoms
from ase.build import niggli_reduce
from ase.ga.ofp_comparator import OFPComparator
from ase.ga.utilities import CellBounds
from chgnet.model import CHGNet
from megnet.utils.models import load_model as megnet_load_model
# from numba import jit, njit, prange
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from csp_elites.crystal.materials_data_model import BandGapEnum, MaterialProperties
from csp_elites.map_elites.elites_utils import Species

warnings.simplefilter("ignore")


class CrystalEvaluatorHunting:
    def __init__(self,
        comparator: OFPComparator = None,
        remove_energy_model = False,
        remove_band_gap_model = False,
        remove_shear_model = False,
        no_check_population_to_kill = False
):
        self.band_gap_calculator = matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi") if not remove_band_gap_model else None
        self.shear_modulus_calculator = megnet_load_model("logG_MP_2018") if not remove_shear_model else None
        self.model = CHGNet.load() if not remove_energy_model else None
        self.no_check_population_to_kill = no_check_population_to_kill

        print(self.model, self.band_gap_calculator, self.shear_modulus_calculator)

    def compute_band_gap(self, relaxed_structure, bandgap_type: Optional[
        BandGapEnum] = BandGapEnum.SCAN):
        if bandgap_type is None:
            for i, method in ((0, "PBE"), (1, "GLLB-SC"), (2, "HSE"), (3, "SCAN")):
                graph_attrs = torch.tensor([i])
                bandgap = self.band_gap_calculator.predict_structure(
                    structure=relaxed_structure, state_feats=graph_attrs
                )

                print(f"{method} band gap")
                print(f"\tRelaxed STO = {float(bandgap):.2f} eV.")
        else:
            graph_attrs = torch.tensor([bandgap_type.value])
            bandgap = self.band_gap_calculator.predict_structure(
                structure=relaxed_structure, state_feats=graph_attrs
            )
        return float(bandgap) * 25 # TODO CHANGE THIS

    def compute_shear_modulus(self, relaxed_structure: Structure) -> float:
        predicted_G = 10 ** self.shear_modulus_calculator.predict_structure(relaxed_structure).ravel()[0]
        return predicted_G


    def batch_compute_fitness_and_bd(self,
                                     list_of_atoms: List[Dict[str, np.ndarray]], cellbounds: CellBounds,
                                     really_relax: bool, behavioral_descriptor_names: List[MaterialProperties],
                                     n_relaxation_steps: int,
                                     fake_data: bool = False
                                     ):

        list_of_dict_atoms = list_of_atoms
        list_of_atoms = [Atoms.fromdict(atoms) for atoms in list_of_atoms]
        structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in list_of_atoms]

        if self.no_check_population_to_kill:
            kill_list = [False] * len(list_of_atoms)
        else:
            kill_list = self.check_atoms_in_cellbounds(list_of_atoms, cellbounds)

        if self.model is None:
            fitness_scores = np.array([0] * len(list_of_atoms))
        else:
            fitness_scores, _ = self.batch_compute_energy(
                list_of_structures=structures,
                really_relax=really_relax,
                n_steps=n_relaxation_steps,
            )

        if self.band_gap_calculator is None:
            band_gaps = np.array([0] * len(list_of_atoms))
        else:
            band_gaps = self._batch_band_gap_compute(structures)

        if self.shear_modulus_calculator is None:
            shear_moduli = np.array([0] * len(list_of_atoms))
        else:
            shear_moduli = self._batch_shear_modulus_compute(structures)

        del structures

        return list_of_atoms, list_of_dict_atoms, fitness_scores, (band_gaps, shear_moduli), kill_list

    def batch_create_species(self, list_of_atoms, fitness_scores, descriptors, kill_list):
        # todo: here could do dict -> atoms conversion

        kill_list = np.array(kill_list)
        individual_indexes_to_add = np.arange(len(list_of_atoms))[~kill_list]
        species_list = []
        for i in individual_indexes_to_add:
            new_specie = Species(
                x=list_of_atoms[i],
                fitness=fitness_scores[i],
                desc=(descriptors[0][i], descriptors[1][i])
            )
            species_list.append(new_specie)

        return species_list

    def _batch_band_gap_compute(self, list_of_structures: List[Structure]):
        # structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in list_of_atoms]
        band_gaps = []
        for i in range(len(list_of_structures)):
            band_gap = self.compute_band_gap(relaxed_structure=list_of_structures[i])
            band_gaps.append(band_gap)
        return band_gaps

    def _batch_shear_modulus_compute(self, list_of_structures: List[Structure]):
        shear_moduli = 10 ** self.shear_modulus_calculator.predict_structures(list_of_structures).ravel()
        return shear_moduli

    def check_atoms_in_cellbounds(self, list_of_atoms: List[Atoms],
                                  cellbounds: CellBounds,
                                  ) -> List[bool]:
        kill_list = []
        for i, atoms in enumerate(list_of_atoms):
            if not cellbounds.is_within_bounds(atoms.get_cell()):
                niggli_reduce(atoms)
                if not cellbounds.is_within_bounds(atoms.get_cell()):
                    kill_individual = True
                else:
                    kill_individual = True
            else:
                kill_individual = False

            kill_list.append(kill_individual)
        return kill_list

    def batch_compute_energy(self, list_of_structures: List[Structure], really_relax, n_steps: int = 10) -> float:
        forces, energies, stresses = self._evaluate_list_of_atoms(list_of_structures)
        reformated_output = []
        for i in range(len(list_of_structures)):
            reformated_output.append(
                {"final_structure": list_of_structures[i],
                 "trajectory": {
                     "energies": energies[i],
                     "forces": forces[i],
                     "stresses": stresses[i],
                 },
                 }
            )
        return -1 * energies, reformated_output

    def _evaluate_list_of_atoms(self, list_of_structures: List[Structure]):
        # list_of_structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in list_of_atoms]

        predictions = self.model.predict_structure(list_of_structures,
                                                   batch_size=len(list_of_structures))
        if isinstance(predictions, dict):
            predictions = [predictions]

        forces = np.array([pred["f"] for pred in predictions])
        energies = np.array([pred["e"] for pred in predictions])
        stresses = np.array([pred["s"] for pred in predictions])
        return forces, energies, stresses
