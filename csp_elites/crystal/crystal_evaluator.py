import warnings
from enum import Enum
from typing import Optional, Tuple, List, Dict

import matgl
from megnet.utils.models import load_model as megnet_load_model
import torch
from ase import Atoms
from ase.build import niggli_reduce
from ase.calculators.singlepoint import SinglePointCalculator
from ase.ga import set_raw_score
from ase.ga.ofp_comparator import OFPComparator
from ase.ga.utilities import CellBounds
from matplotlib import pyplot as plt
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from csp_elites.crystal.override_relaxer import OverridenRelaxer

warnings.simplefilter("ignore")


class BandGapEnum(int, Enum):
    PBE = 0
    GLLB_SC = 1
    MSE = 2
    SCAN = 3

class MaterialProperties(str, Enum):
    BAND_GAP = "band_gap"
    ENERGY_FORMATION = "energy_formation"
    SHEAR_MODULUS = "shear_modulus"

class CrystalEvaluator:
    def __init__(self,
        comparator: OFPComparator = None,

):
        # device = torch.device('cpu' if torch.cuda.is_available() else 'cuda')
        relax_model = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        # relax_model = relax_model.to(device)
        self.relaxer = OverridenRelaxer(potential=relax_model)
        self.comparator = comparator
        self.band_gap_calculator = matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi")
        # self.band_gap_calculator.to(device)
        self.formation_energy_calculator = matgl.load_model("M3GNet-MP-2018.6.1-Eform")
        self.shear_modulus_calculator = megnet_load_model("logG_MP_2018")
        # self.formation_energy_calculator.to(device)
        self.property_to_function_mapping = {
            MaterialProperties.BAND_GAP: self.compute_band_gap,
            MaterialProperties.ENERGY_FORMATION: self.compute_formation_energy,
            MaterialProperties.SHEAR_MODULUS: self.compute_shear_modulus
        }

    def compute_energy(self, atoms: Atoms, really_relax) -> float:
        relaxation_results = self.relaxer.relax(atoms, fmax=0.01, marta_realy_relax=really_relax, steps=10)
        energy = float(
            relaxation_results["trajectory"].energies[-1] / len(atoms.get_atomic_numbers()))
        forces = relaxation_results["trajectory"].forces[-1]
        stresses = relaxation_results["trajectory"].stresses[-1]
        self._finalize_atoms(atoms, energy=energy, forces=forces, stress=stresses)

        return float(-atoms.get_potential_energy()), relaxation_results

    def compute_fitness_and_bd(self,
        atoms: Atoms,
        cellbounds: CellBounds,
        population: List[Atoms],
        really_relax: bool,
        behavioral_descriptor_names: List[MaterialProperties],
                               ) -> Tuple[float, Tuple[float, float], bool]:
        if not cellbounds.is_within_bounds(atoms.get_cell()):
            niggli_reduce(atoms)
            if not cellbounds.is_within_bounds(atoms.get_cell()):
                return 0, (0, 0), True

        # Compute fitness (i.e. energy)
        fitness_score, relaxation_results = self.compute_energy(atoms=atoms, really_relax=really_relax)

        cell = atoms.get_cell()

        # Check whether the individual is valid to be added to the archive
        kill_individual = False
        if not cellbounds.is_within_bounds(cell):
            kill_individual = True

        # if population is not None:
        #     for el in population:
        #         if el.info["confid"] != atoms.info["confid"]:
        #             cosine_distance = self.comparator._compare_structure(atoms, el)
        #             if cosine_distance == 0:
        #                 return 0, (0, 0), True

        # Compute bd dimension

        behavioral_descriptor_values = []

        for property in behavioral_descriptor_names:
            bd_value = self.property_to_function_mapping[property](relaxed_structure=relaxation_results["final_structure"])
            behavioral_descriptor_values.append(bd_value)

        return fitness_score, behavioral_descriptor_values, kill_individual

    def compute_band_gap(self, relaxed_structure, bandgap_type: Optional[BandGapEnum] = BandGapEnum.SCAN):
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

    def compute_formation_energy(self, relaxed_structure):
        return float(self.formation_energy_calculator.predict_structure(relaxed_structure))

    def _finalize_atoms(self, atoms, energy=None, forces=None, stress=None):
        # atoms.wrap() # todo: what does atoms.wrap() do? Why does it not work with M3gnet
        calc = SinglePointCalculator(atoms, energy=energy, forces=forces,
                                     stress=stress)
        atoms.calc = calc
        raw_score = float(-atoms.get_potential_energy())
        set_raw_score(atoms, raw_score)

    def compare_to_target_structures(self,
        generated_structures: List[Atoms],
        target_structures: Dict[str, List[Atoms]],
        directory_string: str,
    ):
        scores = []

        for target_id, target_structure in target_structures.items():
            scores_for_target = []
            for generated_structure in generated_structures:
                cosine_distance = self.comparator._compare_structure(generated_structure,
                                                                target_structure)
                scores_for_target.append(cosine_distance)
                if cosine_distance <= 0.15:
                    print(f"cosine distance {cosine_distance}")
                    print(
                        f"target structure params {target_structure.get_cell_lengths_and_angles()}")
                    print(
                        f"predicted structure params {generated_structure.get_cell_lengths_and_angles()}")

            scores.append(scores_for_target)
            fig, ax = plt.subplots()
            ax.hist(scores_for_target, range=(0, 0.5), bins=20)
            ax.set_title(f"{target_id}")
            ax.set_xlabel("Cosine distance from structure")
            ax.set_ylabel("Number of generated structures")
            if directory_string is None:
                plt.show()
            else:
                plt.savefig(f"{directory_string}/hist_comp_{target_id}.svg", format="svg")

    def compute_shear_modulus(self, relaxed_structure: Structure) -> float:
        predicted_G = 10 ** self.shear_modulus_calculator.predict_structure(relaxed_structure).ravel()[0]
        return predicted_G

    def reduce_structure_to_conventional(self, atoms: Atoms):
        structure = AseAtomsAdaptor.get_structure(atoms)
        conventional_structure = SpacegroupAnalyzer(structure=structure)
        return conventional_structure
