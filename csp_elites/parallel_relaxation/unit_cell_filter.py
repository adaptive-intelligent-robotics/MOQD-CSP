from typing import List

import numpy as np
from ase import Atoms
from ase.cell import Cell
from ase.stress import voigt_6_to_full_3x3_stress, full_3x3_to_voigt_6_stress
from scipy.linalg import logm, expm


#

class AtomsFilterForRelaxation:
    def __init__(self, scalar_pressure: float = 0):
        self.scalar_pressure = scalar_pressure

    @staticmethod
    def deform_grad(
        original_cells: List[Cell],
        new_cells: List[Cell],
    ):
        deformation_gradients = []
        for i in range(len(original_cells)):
            deformation_gradients.append(np.linalg.solve(original_cells[i], new_cells[i]).T)
        return np.array(deformation_gradients)

    def get_positions(self, original_cells, new_cells, list_of_atoms: List[Atoms], cell_factor: np.ndarray):
        positions = self._get_positions_unit_cell_filter(original_cells, new_cells, list_of_atoms, cell_factor)
        natoms = len(list_of_atoms[0])
        cur_deform_grad = self.deform_grad(original_cells, new_cells)
        positions[:, natoms:, :] = np.array([logm(gradient_matrix) for gradient_matrix in cur_deform_grad])
        return positions

    def _get_positions_unit_cell_filter(self, original_cells, new_cells, atoms:List[Atoms], cell_factor: np.ndarray):
        """
        this returns an array with shape (natoms + 3,3).

        the first natoms rows are the positions of the atoms, the last
        three rows are the deformation tensor associated with the unit cell,
        scaled by self.cell_factor.
        """
        cur_deform_grad = self.deform_grad(original_cells, new_cells)
        natoms = np.array(atoms).shape
        positions = np.zeros((natoms[0], natoms[1] + 3, 3))
        for i in range(len(atoms)):
            positions[i, :natoms[1], :] = np.linalg.solve(cur_deform_grad[i],
                                       atoms[i].positions.T).T
        positions[:, natoms[1]:, :] = cell_factor.reshape((-1, 1, 1)) * cur_deform_grad
        return positions

    def set_positions(
        self,
        original_cells: List[Cell],
        atoms_to_update: List[Atoms],
        new_atoms_positions: np.ndarray,
        cell_factors: np.ndarray,
            # **kwargs?
    ):
        natoms = len(atoms_to_update[0])
        updated_positions = new_atoms_positions.copy()
        # updated_positions[:, natoms:, :] = np.array(
        #     [np.expm1(updated_positions[i, natoms:, :]) for i in range(len(updated_positions))])
        updated_positions[:, natoms:, :] = np.array([expm(updated_positions[i, natoms:, :]) for i in range(len(updated_positions))])
        updated_atoms = self._set_positions_unit_cell_filter(
            original_cells,
            atoms_to_update,
            updated_positions,
            cell_factors,
        )
        return updated_atoms

    def _set_positions_unit_cell_filter(
        self,
        original_cells: List[Cell],
        atoms_to_update: List[Atoms],
        new_atoms_positions: np.ndarray,
        cell_factors: np.ndarray,
            # **kwargs?
    ):
        natoms = np.array(atoms_to_update).shape
        new_atom_positions_updated = new_atoms_positions[:, :natoms[1], :]
        new_deform_grad = new_atoms_positions[:, natoms[1]:, :] / cell_factors.reshape((-1, 1, 1))

        # todo copy and create new list for updated atoms and delte teh one you pass?
        for i in range(len(atoms_to_update)):
            atoms_to_update[i].set_cell(original_cells[i] @ new_deform_grad[i].T, scale_atoms=True)
            atoms_to_update[i].set_positions(new_atom_positions_updated[i] @ new_deform_grad[i].T) #, **kwargs)

        return atoms_to_update

    def get_potential_energy(self, energies: np.ndarray, list_of_atoms: List[Atoms]) -> np.ndarray:
        # NB get potential energy has two methods depending ion value of force_consistent  - in CHGNET it point to the same value
        return energies + self.scalar_pressure * np.array([atoms.get_volume() for atoms in list_of_atoms])

    def get_forces_exp_cell_filter(
        self,
        forces_from_chgnet: np.ndarray,
        stresses_from_chgnet: np.ndarray,
        list_of_atoms: List[Atoms],
        original_cells: List[Cell],
        current_atom_cells: List[Cell],
        cell_factors: np.ndarray,
    ):

        forces, stress = self._get_forces_unit_cell_filter(
            forces_from_chgnet,
            stresses_from_chgnet,
            list_of_atoms,
            original_cells,
            current_atom_cells,
            np.array([len(atoms) for atoms in list_of_atoms]),
            # cell_factors,
        )

        stresses = self._process_stress_like_ase_atoms(stresses_from_chgnet)
        volumes = np.array([atoms.get_volume() for atoms in list_of_atoms])

        virial = -volumes.reshape((-1, 1, 1)) * (voigt_6_to_full_3x3_stress(stresses) +
                            np.diag([self.scalar_pressure] * 3))

        cur_deform_grad = self.deform_grad(original_cells, current_atom_cells)
        cur_deform_grad_log = np.array([logm(gradient_matrix) for gradient_matrix in cur_deform_grad])
        if not (cur_deform_grad_log.shape[1] == cur_deform_grad_log.shape[2]):
            print(cur_deform_grad_log.shape)
            print("matrix isnt square")
            print(len(original_cells))
            print(len(current_atom_cells))

        deform_grad_log_force_naive = virial.copy()
        Y = np.zeros((len(list_of_atoms), 6, 6))
        Y[:, 0:3, 0:3] = cur_deform_grad_log
        Y[:, 3:6, 3:6] = cur_deform_grad_log
        try:
            # Y[:, 0:3, 3:6] = - virial @ expm(-cur_deform_grad_log)
            Y[:, 0:3, 3:6] = - np.array([virial[i] @ expm(cur_deform_grad_log[i]) for i in range(len(cur_deform_grad_log))])
            # Y[:, 0:3, 3:6] = -virial @ np.expm1(-cur_deform_grad_log)
        except ValueError:
            print(virial.shape)
            print(cur_deform_grad.shape)
            print(cur_deform_grad_log.shape)
            print(len(original_cells))
            print(len(current_atom_cells))

            for i in range(len(cur_deform_grad)):
                print(cur_deform_grad_log[i].shape)

            print()

        deform_grad_log_force_naive = virial.copy()
        deform_grad_log_force = -np.array([expm(Y[i])[0:3, 3:6] for i in range(len(Y))])

        # deform_grad_log_force = -expm(Y)[:, 0:3, 3:6]
        # deform_grad_log_force = -np.expm1(Y)[:, 0:3, 3:6]
        for (i1, i2) in [(0, 1), (0, 2), (1, 2)]:
            ff = 0.5 * (deform_grad_log_force[:, i1, i2] +
                        deform_grad_log_force[:, i2, i1])
            deform_grad_log_force[:, i1, i2] = ff
            deform_grad_log_force[:, i2, i1] = ff

        # not implemented
        # if self.hydrostatic_strain:
        #     vtr = virial.trace()
        #     virial = np.diag([vtr / 3.0, vtr / 3.0, vtr / 3.0])
        #
        # # Zero out components corresponding to fixed lattice elements
        # if (self.mask != 1.0).any():
        #     virial *= self.mask

        all_are_equal = np.all(np.isclose(deform_grad_log_force,
                                          deform_grad_log_force_naive))

        if all_are_equal or \
            (np.sum(deform_grad_log_force * deform_grad_log_force_naive) /
             np.sqrt(np.sum(deform_grad_log_force**2) *
                     np.sum(deform_grad_log_force_naive**2)) > 0.8):
            deform_grad_log_force = deform_grad_log_force_naive

        # Cauchy stress used for convergence testing
        convergence_crit_stress = -(virial / volumes.reshape((-1, 1, 1)))

        # Not implemented because default = False
        # if self.constant_volume:
        #     # apply constraint to force
        #     dglf_trace = deform_grad_log_force.trace()
        #     np.fill_diagonal(deform_grad_log_force,
        #                      np.diag(deform_grad_log_force) - dglf_trace / 3.0)
        #     # apply constraint to Cauchy stress used for convergence testing
        #     ccs_trace = convergence_crit_stress.trace()
        #     np.fill_diagonal(convergence_crit_stress,
        #                      np.diag(convergence_crit_stress) - ccs_trace / 3.0)

        natoms = forces_from_chgnet.shape[1]
        forces[:, natoms:, :] = deform_grad_log_force
        stress = full_3x3_to_voigt_6_stress(convergence_crit_stress)
        return forces, stress

    def _get_forces_unit_cell_filter(
        self,
        forces_from_chgnet: np.ndarray,
        stresses_from_chgnet: np.ndarray,
        list_of_atoms: List[Atoms],
        original_cells: List[Cell],
        current_atom_cells: List[Cell],
        cell_factors: np.ndarray,
    ):
        stress = self._process_stress_like_ase_atoms(stresses_from_chgnet) # * 1 / 160.21766208)

        volumes = np.array([atoms.get_volume() for atoms in list_of_atoms])
        virial = -volumes.reshape((-1, 1, 1)) * (voigt_6_to_full_3x3_stress(stress) +
                            np.diag([self.scalar_pressure] * 3))
        cur_deform_grad = self.deform_grad(original_cells, current_atom_cells)
        atoms_forces = forces_from_chgnet @ cur_deform_grad
        for i in range(len(list_of_atoms)):
            try:
                virial[i] = np.linalg.solve(cur_deform_grad[i], virial[i].T).T
            except ValueError:
                print(cur_deform_grad[i], virial[i])
                for atoms in list_of_atoms:
                    print(len(atoms))

        # Not implementing because these are not  used as default
        # if self.hydrostatic_strain:
        #     vtr = virial.trace()
        #     virial = np.diag([vtr / 3.0, vtr / 3.0, vtr / 3.0])
        #
        # # Zero out components corresponding to fixed lattice elements
        # if (self.mask != 1.0).any():
        #     virial *= self.mask
        #
        # if self.constant_volume:
        #     vtr = virial.trace()
        #     np.fill_diagonal(virial, np.diag(virial) - vtr / 3.0)

        natoms = forces_from_chgnet.shape[1]
        forces = np.zeros((len(list_of_atoms), natoms + 3, 3))
        forces[:, :natoms, :] = atoms_forces
        forces[:, natoms:, :] = virial / cell_factors.reshape((-1, 1, 1))

        stress = -full_3x3_to_voigt_6_stress(virial) / volumes.reshape((-1, 1, 1))

        return forces, stress


    def _process_stress_like_ase_atoms(self, stresses: np.ndarray):
        """NB ase method also ddifferentiates between voigt = true / false, in teis implementation we only implement the default"""
        return full_3x3_to_voigt_6_stress(stress_matrix=stresses) * 0.006241509125883258  #* 1 / 160.21766208


#
# if __name__ == '__main__':
#     chgnet = CHGNet.load()
#     with MPRester(api_key="4nB757V2Puue49BqPnP3bjRPksr4J9y0") as mpr:
#         one_structure = mpr.get_structure_by_material_id("mp-1341203", final=True)
#
#     tic = time.time()
#     # prediction = chgnet.predict_structure(one_structure)
#
#     atoms = AseAtomsAdaptor.get_atoms(structure=one_structure)
#     atoms_2 = AseAtomsAdaptor.get_atoms(structure=one_structure)
#
#     # calc = SinglePointCalculator(atoms, energy=prediction["e"], forces=prediction["f"],
#     #                              stress=prediction["s"])
#     # atoms.calc = calc
#     # atoms_2.calc = calc
#     atoms_2.rattle(0.1)
#
#     list_of_atoms_2 = [copy.deepcopy(atoms_2),
#                        ]
#     #
#     # mask = None
#     # cell_factor = None
#     # hydrostatic_strain = False
#     # constant_volume = False
#     # scalar_pressure = 0.0
#     model = CHGNet.load()
#     StructOptimizer()
#     parallel_unit_cell = AtomsFilterForRelaxation()
#     # parameters which change from init
#     orig_cell = atoms.get_cell()
#     mask = np.ones(6)
#     # mask = np.arange(6)
#     mask = voigt_6_to_full_3x3_stress(mask)
#     cell_factor = len(atoms)
#     hydrostatic_strain = False
#     constant_volume = False
#     scalar_pressure = 0.0
#     cell_factor = cell_factor
#     atoms_copy = atoms.copy
#     arrays = atoms.arrays
#     atoms_3 = copy.deepcopy(atoms_2)
#     atoms_3.calc = CHGNetCalculator(None, use_device=None, stress_weight=1 / 160.21766208)
#
#     reference = UnitCellFilter(atoms_3)
#
#     reference_exp_filter = ExpCellFilter(copy.deepcopy(atoms_3))
#
#     original_cells = [copy.deepcopy(orig_cell),
#                       copy.deepcopy(orig_cell),
#                       ]
#     atom_cells_2 = [copy.deepcopy(atoms_2.cell),
#                     copy.deepcopy(atoms_2.cell),
#                     ]
#
#     list_of_atoms_1 = [copy.deepcopy(atoms), copy.deepcopy(atoms)]
#     list_of_atoms_2 = [copy.deepcopy(atoms_2),
#                        copy.deepcopy(atoms_2),
#                        ]
#     cell_factors = np.array([cell_factor,
#                              cell_factor,
#                              ])
#
#
#     structures = [AseAtomsAdaptor.get_structure(at) for at in list_of_atoms_2]
#     forces, energies, stresses = parallel_unit_cell._evaluate_list_of_atoms(structures)
#     # pot_en = parallel_unit_cell.get_potential_energy(energies, list_of_atoms_2)
#     # pot_en_reference = reference.get_potential_energy()
#
#     # forces_unit_cell = reference.get_forces()
#     # new_forces, _ = parallel_unit_cell._get_forces_unit_cell_filter(
#     #     forces_from_chgnet=forces,
#     #     stresses_from_chgnet=stresses,
#     #     list_of_atoms=list_of_atoms_2,
#     #     original_cells=original_cells,
#     #     current_atom_cells=atom_cells_2,
#     #     cell_factors=cell_factors
#     # )
#     #
#     #
#     # new_forces = parallel_unit_cell.get_forces_exp_cell_filter(
#     #     forces_from_chgnet=forces,
#     #     stresses_from_chgnet=stresses,
#     #     list_of_atoms=list_of_atoms_2,
#     #     original_cells=original_cells,
#     #     current_atom_cells=atom_cells_2,
#     #     cell_factors=cell_factors
#     # )
#     #
#     # force_target = reference_exp_filter.get_forces()
#
#     a = 2
#     print()
#
#
#
#     positions = parallel_unit_cell.get_positions(original_cells, atom_cells_2, list_of_atoms_2, cell_factors)
#     pos = reference_exp_filter.get_positions()
#     for el in positions:
#         assert (el == pos).all()
#
#     updated_atoms = parallel_unit_cell.set_positions(original_cells, list_of_atoms_2, positions, cell_factors)
#     positions_2 = parallel_unit_cell.get_positions(original_cells, atom_cells_2, list_of_atoms_2,
#                                                                      cell_factors)
#     reference_exp_filter.set_positions(pos)
#     pos_2 = reference_exp_filter.get_positions()
#     print()
