from __future__ import annotations

from typing import Sequence, Literal

from chgnet import PredTask
from chgnet.model import CHGNet
from pymatgen.core import Structure
from torch import Tensor, nn

class OverloadCHGnet:
    def __init__(self):
        self.model = CHGNet.load()

    def predict_structure_marta(
            self,
            structure: Structure | Sequence[Structure],
            task: PredTask = "efsm",
            return_atom_feas: bool = False,
            return_crystal_feas: bool = False,
            batch_size: int = 100,
    ) -> dict[str, Tensor] | list[dict[str, Tensor]]:
        """Predict from pymatgen.core.Structure.

        Args:
            structure (Structure | Sequence[Structure]): structure or a list of structures
                to predict.
            task (str): can be 'e' 'ef', 'em', 'efs', 'efsm'
                Default = "efsm"
            return_atom_feas (bool): whether to return atom features.
                Default = False
            return_crystal_feas (bool): whether to return crystal features.
                only available if self.mlp_first is False
                Default = False
            batch_size (int): batch_size for predict structures.
                Default = 100

        Returns:
            prediction (dict[str, Tensor]): containing the keys:
                e: energy of structures [batch_size, 1] in eV/atom
                f: force on atoms [num_batch_atoms, 3] in eV/A
                s: stress of structure [3 * batch_size, 3] in GPa
                m: magnetic moments of sites [num_batch_atoms, 3] in Bohr magneton mu_B
        """
        if self.model.graph_converter is None:
            raise ValueError("graph_converter cannot be None!")

        structures = [structure] if isinstance(structure, Structure) else structure

        graphs = [self.model.graph_converter(struct, on_isolated_atoms="warn") for struct in structures]
        return self.model.predict_graph(
            graphs,
            task=task,
            return_atom_feas=return_atom_feas,
            return_crystal_feas=return_crystal_feas,
            batch_size=batch_size,
        )
