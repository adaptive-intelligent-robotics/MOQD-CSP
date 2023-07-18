

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

from csp_elites.crystal.crystal_evaluator import MaterialProperties, BandGapEnum
from csp_elites.crystal.override_relaxer import OverridenRelaxer

warnings.simplefilter("ignore")

def parallel_fitness_func_and_bd_computation(
        atoms: Atoms,
        cellbounds: CellBounds,
        population: List[Atoms],
        really_relax: bool,
        behavioral_descriptor_names: [MaterialProperties.BAND_GAP, MaterialProperties.SHEAR_MODULUS],
):
    relax_model = matgl.load_model("M3GNet-MP-2021.2.8-PES")
    relaxer = OverridenRelaxer(potential=relax_model)
    band_gap_calculator = matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi")
    formation_energy_calculator = matgl.load_model("M3GNet-MP-2018.6.1-Eform")
    shear_modulus_calculator = megnet_load_model("logG_MP_2018")

    relaxation_results = relaxer.relax(atoms, fmax=0.01, marta_realy_relax=really_relax)
    energy = float(
        relaxation_results["trajectory"].energies[-1] / len(atoms.get_atomic_numbers()))
    forces = relaxation_results["trajectory"].forces[-1]
    stresses = relaxation_results["trajectory"].stresses[-1]

    # atoms.wrap() # todo: what does atoms.wrap() do? Why does it not work with M3gnet
    calc = SinglePointCalculator(atoms, energy=energy, forces=forces,
                                 stress=stresses)
    atoms.calc = calc
    raw_score = float(-atoms.get_potential_energy())
    set_raw_score(atoms, raw_score)

    graph_attrs = torch.tensor([BandGapEnum.SCAN.value])


    bandgap = band_gap_calculator.predict_structure(
        structure=relaxation_results["final_structure"], state_feats=graph_attrs
    )
    bandgap = float(bandgap) * 25

    if MaterialProperties.SHEAR_MODULUS in behavioral_descriptor_names:
        bd_2 =  10 ** shear_modulus_calculator.predict_structure(relaxation_results["final_structure"]).ravel()[0]
    else:

        bd_2 = float(formation_energy_calculator.predict_structure(relaxation_results["final_structure"]))

    cell = atoms.get_cell()

    # Check whether the individual is valid to be added to the archive
    kill_individual = False
    if not cellbounds.is_within_bounds(cell):
        kill_individual = True

    return energy, (bandgap, bd_2), kill_individual
