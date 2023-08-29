import torch
from chgnet.model import CHGNet
from tensorflow.python.client import device_lib
import tensorflow as tf

from csp_elites.property_calculators.band_gap_calculator import BandGapCalculator
from csp_elites.property_calculators.shear_modulus_calculator import ShearModulusCalculator

if __name__ == '__main__':
    energy = CHGNet.load()
    bg = BandGapCalculator()
    shear = ShearModulusCalculator()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    energy.to(device)
    bg.model_wrapper.to(device)
    print(f"graph converting {energy.graph_converter.algorithm}")
    print(f"CHGNet on gpu {next(energy.parameters()).is_cuda}")

    print(f"BG on gpu {next(bg.model_wrapper.parameters()).is_cuda}")
    string = "GPU"
    print(f"Shear on gpu {string in str(device_lib.list_local_devices())}")
    print(tf.config.list_physical_devices('GPU'))
