import pathlib
from typing import Tuple, List, Optional

import numpy as np
from ase.ga.utilities import closest_distances_generator
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from mp_api.client import MPRester
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from csp_elites.crystal.force_mutation import DQDMutationOMGMEGA
from csp_elites.crystal.materials_data_model import MaterialProperties
from csp_elites.property_calculators.band_gap_calculator import BandGapCalculator
from csp_elites.property_calculators.shear_modulus_calculator import ShearModulusCalculator
from csp_elites.property_calculators.structure_optimizer import MultiprocessOptimizer


# check forces
# check shear
# check band gap
class PropertyToCalculatorMatcher:
    property_to_calculator_dictionary = {
        MaterialProperties.BAND_GAP: BandGapCalculator,
        MaterialProperties.ENERGY_FORMATION: None,
        MaterialProperties.SHEAR_MODULUS: ShearModulusCalculator,
        MaterialProperties.CONSTRAINT_FORCE: None,
        MaterialProperties.CONSTRAINT_BG: None,
        MaterialProperties.CONSTRAINT_SHEAR: None,
        MaterialProperties.ENERGY: MultiprocessOptimizer,
    }

    property_to_ylabel = {
        MaterialProperties.BAND_GAP: "Band Gap, eV",
        MaterialProperties.ENERGY_FORMATION: None,
        MaterialProperties.SHEAR_MODULUS: "Shear Modulus, GPa",
        MaterialProperties.CONSTRAINT_FORCE: None,
        MaterialProperties.CONSTRAINT_BG: None,
        MaterialProperties.CONSTRAINT_SHEAR: None,
        MaterialProperties.ENERGY: "Energy per Atom, eV/Atom",
    }


    @classmethod
    def get_calculator(cls, property: MaterialProperties):
        return cls.property_to_calculator_dictionary[property]

    @classmethod
    def get_ylabel(cls, property: MaterialProperties):
        return cls.property_to_ylabel[property]


class DQDMutation:
    pass


class GradientComputationValidator:
    def __init__(self, material_property: MaterialProperties, learning_rate: float = 1e-3):
        self.calculator = PropertyToCalculatorMatcher.get_calculator(material_property)()
        self.learning_rate = learning_rate
        self._test_structure_mp_references = ["mp-390", "mp-1840"] # structures have 6,, and 24 atoms respectively
        self.property = material_property
        self.omg_mega_mutation = DQDMutation
        self.save_directory = pathlib.Path(__file__).parent.parent.parent / ".experiment.nosync/validation"

    def _get_test_structure(self, test_structure_index: int = 0):
        with MPRester(api_key="4nB757V2Puue49BqPnP3bjRPksr4J9y0") as mpr:
            structure = mpr.get_structure_by_material_id(
                self._test_structure_mp_references[test_structure_index], final=True
            )
        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.rattle(0.3)
        return AseAtomsAdaptor.get_structure(atoms)

    def _update_atoms_positions_with_gradient(self, structure: Structure, gradient: np.ndarray):
        position_change = self.learning_rate * gradient
        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.set_positions(atoms.get_positions() + position_change)
        return AseAtomsAdaptor.get_structure(atoms)

    def compute_gradients(self, structure: Structure):
        return self.calculator.compute(structure, compute_gradients=True)

    def step(self, structure: Structure) -> Tuple[Structure, float]:
        value, gradients = self.compute_gradients(structure)
        new_structure = self._update_atoms_positions_with_gradient(structure, gradients)
        return new_structure, value

    def step_for_n_steps(self, structure: Optional[Structure], n_steps: int= 5):
        if structure is None:
            structure = self._get_test_structure()

        values_over_steps = []
        for i in range(n_steps):
            structure, value = self.step(structure)
            values_over_steps.append(value)

        return values_over_steps

    def plot_values_over_steps(self, values: List[float], label: str, fig: Optional[Axes]=None, ax: Optional[Axes]=None):

        if ax is None:
            ylabel = PropertyToCalculatorMatcher.get_ylabel(self.property)
            title = ylabel.split(", ")[0]
            fig, ax = plt.subplots()
            ax.set_title(f"Change in {title}")
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Number of Steps Taken")
            if len(values) > 100:
                ax.set_xticks(np.linspace(0, len(values)), 100)

        ax.plot(np.arange(len(values)), values, label=label)

        return fig, ax

    def loop_for_all_test_structures(self, n_steps: int = 5):
        structure_indices = np.arange(len(self._test_structure_mp_references))
        values_by_structure = []
        fig, ax = None, None
        for structure_id in structure_indices:
            structure = self._get_test_structure(structure_id)
            values = self.step_for_n_steps(structure, n_steps=n_steps)
            values_by_structure.append(values)

            fig, ax = self.plot_values_over_steps(values, label=self._test_structure_mp_references[structure_id], fig=fig, ax=ax)

        plt.legend()
        plt.savefig(self.save_directory / f"gradient_stepping_{self.property.value}_lr_{self.learning_rate}.png", format="png")



if __name__ == '__main__':
    n_steps = 1000
    for learning_rate in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        for property in [MaterialProperties.BAND_GAP, MaterialProperties.SHEAR_MODULUS, MaterialProperties.ENERGY]:
            gradient_validator = GradientComputationValidator(material_property=property, learning_rate=learning_rate)
            gradient_validator.loop_for_all_test_structures(n_steps=n_steps)
