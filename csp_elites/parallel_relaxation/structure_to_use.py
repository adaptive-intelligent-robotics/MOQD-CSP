import copy

from chgnet.model import CHGNet, CHGNetCalculator
from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor

# if __name__ == '__main__':

chgnet = CHGNet.load()
with MPRester(api_key="4nB757V2Puue49BqPnP3bjRPksr4J9y0") as mpr:
    one_structure = mpr.get_structure_by_material_id("mp-1341203", final=True)
    # two_structure = mpr.get_structure_by_material_id("mp-1341203", final=True)

atoms_for_ref = AseAtomsAdaptor.get_atoms(one_structure)

atoms_for_ref.rattle(0.1)
atoms_to_test = copy.deepcopy(atoms_for_ref)

atoms_for_ref.calc = CHGNetCalculator()
