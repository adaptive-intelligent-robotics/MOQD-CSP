from __future__ import annotations

import contextlib
import io
import sys
from typing import TYPE_CHECKING, Optional

import torch
from ase import Atoms
from ase.constraints import ExpCellFilter
from matgl.ext.ase import Relaxer, TrajectoryObserver
from numba import jit
from pymatgen.core.structure import Molecule, Structure

if TYPE_CHECKING:
    from ase.optimize.optimize import Optimizer
    from matgl.apps.pes import Potential

class OverridenRelaxer(Relaxer):
    def __init__(self,
        potential: Potential | None = None,
        state_attr: torch.Tensor | None = None,
        optimizer: Optimizer | str = "FIRE",
        relax_cell: bool = True,
        stress_weight: float = 0.01,
    ):
        super().__init__(potential, state_attr, optimizer, relax_cell, stress_weight)
        self.marta_relax_cells = False

    # todo: @jit()
    def relax(
        self,
        atoms: Atoms,
        fmax: float = 0.1,
        steps: int = 500,
        traj_file: Optional[str] = None,
        interval=1,
        verbose=False,
        marta_realy_relax=False,
        **kwargs,
    ):
        """
        Relax an input Atoms.

        Args:
            atoms (Atoms): the atoms for relaxation
            fmax (float): total force tolerance for relaxation convergence.
            Here fmax is a sum of force and stress forces
            steps (int): max number of steps for relaxation
            traj_file (str): the trajectory file for saving
            interval (int): the step interval for saving the trajectories
            verbose (bool): Whether to have verbose output.
            kwargs: Kwargs pass-through to optimizer.
        """
        if isinstance(atoms, (Structure, Molecule)):
            atoms = self.ase_adaptor.get_atoms(atoms)
        atoms.set_calculator(self.calculator)
        stream = sys.stdout if verbose else io.StringIO()
        with contextlib.redirect_stdout(stream):
            obs = TrajectoryObserver(atoms)
            if self.relax_cell:
                atoms = ExpCellFilter(atoms)
            if marta_realy_relax:
                optimizer = self.opt_class(atoms, **kwargs)
                optimizer.attach(obs, interval=interval)
                optimizer.run(fmax=fmax, steps=steps)
            obs()
        if traj_file is not None:
            obs.save(traj_file)
        if isinstance(atoms, ExpCellFilter):
            atoms = atoms.atoms
        return {
            "final_structure": self.ase_adaptor.get_structure(atoms),
            "trajectory": obs,
        }
