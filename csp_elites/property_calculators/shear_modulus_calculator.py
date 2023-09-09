from typing import Optional, Tuple, List

import dgl
import numpy as np
import tensorflow as tf
import torch
# from matgl.config import DEFAULT_ELEMENT_TYPES
from matgl.ext.pymatgen import Structure2Graph
from matgl.graph.compute import compute_pair_vector_and_distance
from megnet.utils.models import load_model as megnet_load_model
from mp_api.client import MPRester
from pymatgen.core import Structure

from csp_elites.property_calculators.bond_converter import BondConverterTorch
from csp_elites.utils.utils import normalise_between_0_and_1

# copying default element types from matgl config because there was a weird bug. Should be able to remove this and replace with import
DEFAULT_ELEMENT_TYPES = (
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
)

class ShearModulusCalculator:
    def __init__(self, normalisation_values: Optional[Tuple[float, float]] = None):
        self.model_wrapper = megnet_load_model("logG_MP_2018")
        self.graph_converter_torch = Structure2Graph(element_types=DEFAULT_ELEMENT_TYPES, cutoff=4.0) # todo: find where default cutoff loaded from
        self.bond_converter = BondConverterTorch()
        self.normalisation_values = normalisation_values

    def compute(self, structure: Structure, compute_gradients: bool = False):
        if compute_gradients:
            log_shear_modulus, gradients = self._compute_log_shear_modulus_with_gradients(structure)
        else:
            log_shear_modulus = self._compute_log_shear_modulus_no_gradients(structure)
            gradients = None
        shear_modulus = 10 ** log_shear_modulus
        if self.normalisation_values is not None:
            shear_modulus = normalise_between_0_and_1(shear_modulus, self.normalisation_values)
        return shear_modulus, gradients

    def _compute_log_shear_modulus_no_gradients(self, structure: Structure):
        shear_modulus_log = self.model_wrapper.predict_structure(structure).ravel()[0]
        return shear_modulus_log

    def compute_no_grad_batch(self, list_of_structures: List[Structure]):
        return 10 ** self.model_wrapper.predict_structures(list_of_structures).ravel()

    def _compute_log_shear_modulus_with_gradients(self, structure: Structure):
        bond_distances, bond_distance_gradient_wrt_positions = \
            self._compute_bond_distances_with_gradients(structure)

        graph = self.model_wrapper.graph_converter.convert(structure)
        graph["bond"] = tf.Variable(graph["bond"])

        inputs = self._graph_to_input(graph)

        with tf.GradientTape() as tape:
            tape.watch(inputs[1])
            model_output = self.model_wrapper.model(inputs)
            model_output_converted = self.model_wrapper.target_scaler.inverse_transform(
                model_output[0, 0], len(graph["atom"]))

        model_output_converted = model_output_converted.numpy()

        graph_gradient_wrt_bond_distances = tape.gradient(model_output, inputs[1])
        del tape
        del graph
        del inputs
        graph_gradient_wrt_bond_distances = np.array(graph_gradient_wrt_bond_distances[0])
        derivatives_multiplied = bond_distance_gradient_wrt_positions * \
                                 graph_gradient_wrt_bond_distances.reshape((-1, 100, 1, 1))
        sum_1 = derivatives_multiplied.sum(axis=1)
        derivative_graph_wrt_atom_positions = sum_1.sum(axis=0)
        return model_output_converted[0], derivative_graph_wrt_atom_positions

    def _compute_bond_distances_with_gradients(self, structure: Structure):
        graph, state_feats_default = self.graph_converter_torch.get_graph(structure)
        graph.ndata["pos"].requires_grad_()
        graph.edata["pbc_offset"].requires_grad_()
        graph.edata["lattice"].requires_grad_()
        bond_vec, bond_dist = compute_pair_vector_and_distance(graph)

        bond_dist_converted = self.bond_converter.convert(bond_dist)
        all_torch_grad = \
        torch.autograd.grad(bond_dist_converted.sum(), graph.ndata["pos"], create_graph=True,
                            retain_graph=True)[0].detach().numpy()
        return bond_dist_converted, all_torch_grad

    def _graph_to_input(self, graph: dgl.DGLGraph):
        """This function mirrors XXX from megnet
        megnet/data/graph.py StructureGraph graph_to_input
        """

        gnode = [0] * len(graph["atom"])
        gbond = [0] * len(graph["index1"])

        return [
            np.expand_dims(self.model_wrapper.graph_converter.atom_converter.convert(graph["atom"]), axis=0),
            tf.expand_dims(self.model_wrapper.graph_converter.bond_converter.convert(graph["bond"]), axis=0),
            np.expand_dims(np.array(graph["state"]), axis=0),
            np.expand_dims(np.array(graph["index1"]), axis=0),
            np.expand_dims(np.array(graph["index2"]), axis=0),
            np.expand_dims(np.array(gnode), axis=0),
            np.expand_dims(np.array(gbond), axis=0),
        ]

    def _compute_pair_vector_and_distance(self, g: dgl.DGLGraph):
        """Calculate bond vectors and distances using dgl graphs.

        This function mirrors compute_pair_and_distances from matgl

        Args:
        g: DGL graph

        Returns:
        bond_vec (torch.tensor): bond distance between two atoms
        bond_dist (torch.tensor): vector from src node to dst node
        """
        bond_vec = torch.zeros(g.num_edges(), 3)
        bond_vec[:, :] = (
                g.ndata["pos"][g.edges()[1][:].long(), :]
                + torch.squeeze(
            torch.matmul(g.edata["pbc_offset"].unsqueeze(1), torch.squeeze(g.edata["lattice"])))
                - g.ndata["pos"][g.edges()[0][:].long(), :]
        )

        bond_dist = torch.norm(bond_vec, dim=1)

        return bond_vec, bond_dist


if __name__ == '__main__':
    shear_calculator = ShearModulusCalculator()

    with MPRester(api_key="4nB757V2Puue49BqPnP3bjRPksr4J9y0") as mpr:
        structure = mpr.get_structure_by_material_id("mp-1840", final=True)
    shear_no_grad, _ = shear_calculator.compute(structure, compute_gradients=False)
    shear_with_grad, gradient = shear_calculator.compute(structure, compute_gradients=True)
    assert shear_no_grad == shear_with_grad
    print(gradient)
