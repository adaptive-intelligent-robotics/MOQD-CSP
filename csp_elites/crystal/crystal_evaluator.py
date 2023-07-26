import time
import warnings
from enum import Enum
from typing import Optional, Tuple, List, Dict

import matgl
from chgnet.model import StructOptimizer, CHGNet
from chgnet.model.dynamics import TrajectoryObserver
from megnet.utils.models import load_model as megnet_load_model
import torch
from ase import Atoms
from ase.build import niggli_reduce
from ase.calculators.singlepoint import SinglePointCalculator
from ase.ga import set_raw_score
from ase.ga.ofp_comparator import OFPComparator
from ase.ga.utilities import CellBounds
from matplotlib import pyplot as plt
# from numba import jit, njit, prange
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
        # self.relaxer = OverridenRelaxer(potential=relax_model)
        # self.relaxer = StructOptimizer()
        # self.relaxer = MultiprocessingOptimizer()
        self.comparator = comparator
        self.band_gap_calculator = matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi")
        # self.band_gap_calculator.to(device)
        # self.formation_energy_calculator = matgl.load_model("M3GNet-MP-2018.6.1-Eform")
        self.shear_modulus_calculator = megnet_load_model("logG_MP_2018")
        # self.formation_energy_calculator.to(device)
        self.property_to_function_mapping = {
            MaterialProperties.BAND_GAP: self.compute_band_gap,
            MaterialProperties.ENERGY_FORMATION: self.compute_formation_energy,
            MaterialProperties.SHEAR_MODULUS: self.compute_shear_modulus
        }
        self.model = CHGNet.load()

    # todo:  @jit(cache=True)
    def compute_energy(self, atoms: Atoms, really_relax, n_steps: int = 10) -> float:
        # relaxation_results = self.relaxer.relax(atoms, steps=n_steps, verbose=False)

        # energy = float(
        #     relaxation_results["trajectory"].energies[-1] / len(atoms.get_atomic_numbers()))
        # forces = relaxation_results["trajectory"].forces[-1]
        # stresses = relaxation_results["trajectory"].stresses[-1]
        # self._finalize_atoms(atoms, energy=energy, forces=forces, stress=stresses)
        # relaxation_results = None

        structure = AseAtomsAdaptor.get_structure(atoms)
        prediction = self.model.predict_structure(structure)
        traj_observer = TrajectoryObserver(atoms)
        traj_observer.energies.append(prediction["e"])
        traj_observer.forces.append(prediction["f"])
        traj_observer.stresses.append(prediction["s"])
        relaxation_results ={
            "final_structure": structure,
            "trajectory": traj_observer
        }
        return -prediction["e"], relaxation_results
        # return float(-atoms.get_potential_energy()), relaxation_results

    #todo: @jit(parallel=True, cache=True)
    # @jit(parallel=True)
    def compute_fitness_and_bd(self,
        atoms: Atoms,
        cellbounds: CellBounds,
        really_relax: bool,
        behavioral_descriptor_names: List[MaterialProperties],
        n_relaxation_steps: int,
    ) -> Tuple[float, Tuple[float, float], bool]:
        if not cellbounds.is_within_bounds(atoms.get_cell()):
            niggli_reduce(atoms)
            if not cellbounds.is_within_bounds(atoms.get_cell()):
                return 0, (0, 0), True

        # Compute fitness (i.e. energy)
        fitness_score, relaxation_results = self.compute_energy(atoms=atoms, really_relax=really_relax,
                                                                n_steps=n_relaxation_steps)

        cell = atoms.get_cell()

        # Check whether the individual is valid to be added to the archive
        kill_individual = False
        if not cellbounds.is_within_bounds(cell):
            kill_individual = True

        # Compute bd dimension

        behavioral_descriptor_values = []

        for i in range(len(behavioral_descriptor_names)):
            bd_value = self.property_to_function_mapping[behavioral_descriptor_names[i]](relaxed_structure=relaxation_results["final_structure"])
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

    def compute_local_order(self, atoms: Atoms):
        all_local_orders = self.comparator.get_local_orders(atoms)
        pass

    def compute_mixing_energy(
            self, crystal_energy: float,
            crystal_1_proportion: float,
            edge_crystal_1_energy: float,
            edge_crystal_2_energy: float,
    ):
        return crystal_energy * (
                (1 - crystal_1_proportion) * edge_crystal_2_energy +
                crystal_1_proportion * edge_crystal_1_energy
        )

    def compute_fitness_and_bd_mixing_system(
            self, atoms: Atoms, cellbounds, really_relax: bool, crystal_1_proportion: float,
            edge_crystal_1_energy: float, edge_crystal_2_energy: float,
    ):
        if not cellbounds.is_within_bounds(atoms.get_cell()):
            niggli_reduce(atoms)
            if not cellbounds.is_within_bounds(atoms.get_cell()):
                return 0, (0, 0), True

        # Compute fitness (i.e. energy)
        crystal_energy, relaxation_results = self.compute_energy(atoms=atoms,
                                                                 really_relax=really_relax)

        cell = atoms.get_cell()

        # Check whether the individual is valid to be added to the archive
        kill_individual = False
        if not cellbounds.is_within_bounds(cell):
            kill_individual = True

        fitness_score = self.compute_mixing_energy(
            crystal_energy=crystal_energy,
            crystal_1_proportion=crystal_1_proportion,
            edge_crystal_1_energy=edge_crystal_1_energy,
            edge_crystal_2_energy=edge_crystal_2_energy,
        )

        band_gap = self.compute_band_gap(relaxed_structure=relaxation_results["final_structure"])

        return fitness_score(band_gap, crystal_energy), kill_individual
