from pymatgen.core import Lattice, Structure, Molecule

if __name__ == '__main__':

    coords = [[0, 0, 0], [0.75,0.5,0.75]]
    lattice = Lattice.from_parameters(a=3.84, b=3.84, c=3.84, alpha=120,
                                      beta=90, gamma=60)
    struct = Structure(lattice, ["Si", "Si"], coords)

    strucure_2 = Structure(lattice, ["Si", "Si"], coords)
    print(struct == strucure_2)
