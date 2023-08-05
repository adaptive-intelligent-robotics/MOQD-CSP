from enum import Enum


class BandGapEnum(int, Enum):
    PBE = 0
    GLLB_SC = 1
    MSE = 2
    SCAN = 3


class MaterialProperties(str, Enum):
    BAND_GAP = "band_gap"
    ENERGY_FORMATION = "energy_formation"
    SHEAR_MODULUS = "shear_modulus"
    CONSTRAINT_FORCE = "constraint_force"


class StartGenerators(str, Enum):
    RANDOM = "random"
    PYXTAL = "pyxtal"
