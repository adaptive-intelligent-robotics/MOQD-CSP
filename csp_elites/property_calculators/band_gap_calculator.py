from typing import Optional, Tuple

import dgl
import matgl
import torch
from matgl.ext.pymatgen import Structure2Graph
from mp_api.client import MPRester
from pymatgen.core import Structure
from matgl.graph.compute import compute_pair_vector_and_distance

from csp_elites.utils.utils import normalise_between_0_and_1


class BandGapCalculator:
    def __init__(self, normalisation_values: Optional[Tuple[float, float]] = None):
        self.model_wrapper = matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi")

        self.graph_converter = Structure2Graph(
            element_types=self.model_wrapper.model.element_types,
            cutoff=self.model_wrapper.model.cutoff,
        )
        self.normalisation_values = normalisation_values

    def compute(self,
                structure: Structure,
                band_gap_type: torch.Tensor = torch.tensor([3]),
                compute_gradients: bool = False,
                ):
        if compute_gradients:
            model_output, gradients_wrt_positions = self._compute_band_gap_with_gradients(
                structure=structure, band_gap_type=band_gap_type,
            )
        else:
            model_output = self._compute_band_gap_no_gradients(
                structure=structure, band_gap_type=band_gap_type,
            )
            gradients_wrt_positions = None
        model_output = model_output.detach().numpy()
        if self.normalisation_values is not None:
            model_output = normalise_between_0_and_1(model_output, self.normalisation_values)
        return model_output, gradients_wrt_positions

    def _compute_band_gap_no_gradients(self, structure: Structure, band_gap_type: torch.Tensor):
        return self.model_wrapper.predict_structure(
            structure=structure,
            state_feats=band_gap_type
        )

    def _compute_band_gap_with_gradients(self, structure: Structure, band_gap_type: torch.Tensor):
        graph, band_gap_type_default = self.graph_converter.get_graph(structure)
        if band_gap_type is None:
            band_gap_type = torch.tensor(band_gap_type_default)
        graph.ndata["pos"].requires_grad_()
        graph.edata["pbc_offset"].requires_grad_()
        graph.edata["lattice"].requires_grad_()
        bond_vec, bond_dist = compute_pair_vector_and_distance(graph)

        graph.edata["edge_attr"] = self.model_wrapper.model.bond_expansion(bond_dist)

        # adding requires_grad argument edges to access gradients
        graph.edata["edge_attr"].requires_grad_()

        model_output = self.model_wrapper(
            graph, graph.edata["edge_attr"], graph.ndata["node_type"], band_gap_type,
        )

        gradient_wrt_positions = torch.autograd.grad(
            model_output, graph.ndata["pos"], create_graph=True, retain_graph=True,
        )[0].detach().numpy()

        model_output = model_output.detach()
        # model_output_converted = self.model_wrapper.transformer.inverse_transform(model_output)
        return model_output, gradient_wrt_positions


if __name__ == '__main__':
    shear_calculator = BandGapCalculator()

    with MPRester(api_key="4nB757V2Puue49BqPnP3bjRPksr4J9y0") as mpr:
        structure = mpr.get_structure_by_material_id("mp-1840", final=True)
    bg_no_grad, _ = shear_calculator.compute(structure, compute_gradients=False, band_gap_type=torch.tensor([3]))
    bg_with_grad, gradient = shear_calculator.compute(structure, compute_gradients=True, band_gap_type=torch.tensor([3]))
    assert bg_no_grad == bg_with_grad
    print(gradient)
