import copy

import numpy as np
import pytest
from ase.constraints import UnitCellFilter
from ase.stress import voigt_6_to_full_3x3_stress
from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor

from csp_elites.parallel_relaxation.unit_cell_filter import AtomsFilterForRelaxation

with MPRester(api_key="4nB757V2Puue49BqPnP3bjRPksr4J9y0") as mpr:
    one_structure = mpr.get_structure_by_material_id("mp-1341203", final=True)

og_atoms = AseAtomsAdaptor.get_atoms(structure=one_structure)

atoms = AseAtomsAdaptor.get_atoms(structure=one_structure)
atoms.rattle((0.7))

list_of_atoms = [copy.deepcopy(atoms),
                copy.deepcopy(atoms),
                copy.deepcopy(atoms),
                ]

unit_cell_filter = UnitCellFilter(atoms)
original_cells = [list_of_atoms[i].cell for i in range(len(list_of_atoms))]
list_of_atom_cells = [list_of_atoms[i].cell for i in range(len(list_of_atoms))]
masks = np.ones((len(list_of_atoms), 6))
masks = voigt_6_to_full_3x3_stress(masks)
cell_factors = np.array([len(atom) for atom in list_of_atoms])

position = unit_cell_filter.get_positions()
positions = np.array([copy.deepcopy(position), copy.deepcopy(position), copy.deepcopy(position)])

class TestUnitCellFilter:
    # @pytest.fixture
    # def single_atoms(self):
    #     with MPRester(api_key="4nB757V2Puue49BqPnP3bjRPksr4J9y0") as mpr:
    #         one_structure = mpr.get_structure_by_material_id("mp-1341203", final=True)
    #     atoms =  AseAtomsAdaptor.get_atoms(structure=one_structure)
    #     atoms.rattle((0.7))
    #     return atoms
    #
    # @pytest.fixture
    # def ase_unit_cell_filter(self) -> UnitCellFilter:
    #     """Creates a generator with grid size of 5 and 3 agents."""
    #     return UnitCellFilter(atoms=self.single_atoms)
    #
    # @pytest.fixture
    # def list_of_atoms(self):
    #     return [copy.deepcopy(self.single_atoms),
    #             copy.deepcopy(self.single_atoms),
    #             copy.deepcopy(self.single_atoms),
    #             ]

    @pytest.fixture
    def parallel_unit_cell_filter(self):
        return AtomsFilterForRelaxation()

    @pytest.mark.parametrize(
        ("function_input", "expected_output"),
        [
            (
                (original_cells, list_of_atom_cells, list_of_atoms, cell_factors),
                unit_cell_filter.get_positions()
            ),
        ]
    )
    def test_get_positions(self, parallel_unit_cell_filter: AtomsFilterForRelaxation,
                           function_input, expected_output):
        original_cells, list_of_atom_cells, list_of_atoms, cell_factors = function_input
        atom_positions = parallel_unit_cell_filter._get_positions_unit_cell_filter(original_cells, list_of_atom_cells, list_of_atoms, cell_factors)
        for el in atom_positions:
            assert (el == expected_output).all()

    @pytest.mark.parametrize(
        ("function_input", "expected_output"),
        [
            (
                (original_cells, list_of_atoms, positions, cell_factors),
                unit_cell_filter
            )
        ]
    )
    def test_set_positions(self, parallel_unit_cell_filter: AtomsFilterForRelaxation,
                           function_input, expected_output):
        original_cells, list_of_atoms, positions, cell_factors = function_input
        updated_atoms = parallel_unit_cell_filter._set_positions_unit_cell_filter(original_cells, list_of_atoms, positions, cell_factors)

        unit_cell_filter.set_positions(positions[0])
        for el in updated_atoms:
            assert (el._get_positions_unit_cell_filter() == unit_cell_filter.atoms._get_positions_unit_cell_filter()).all()
