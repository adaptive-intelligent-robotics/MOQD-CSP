from enum import Enum


class BandGapEnum(int, Enum):
    PBE = 0
    GLLB_SC = 1
    MSE = 2
    SCAN = 3


class MaterialProperties(str, Enum):
    BAND_GAP = "band_gap"
    SHEAR_MODULUS = "shear_modulus"
    ENERGY = "energy"
    CONSTRAINT_FORCE = "constraint_force"
    # The propperties below are not natively handled in this implementation
    CONSTRAINT_BG = "constraint_band_gap"
    CONSTRAINT_SHEAR = "constraint_shear"
    ENERGY_FORMATION = "energy_formation"

class StartGenerators(str, Enum):
    RANDOM = "random"
    PYXTAL = "pyxtal"
