from typing import Tuple

import numpy as np


def normalise_between_0_and_1(
    values_to_normalise: np.ndarray, normalisation_limits: Tuple[float, float]
):
    return (values_to_normalise - normalisation_limits[0]) / (
        normalisation_limits[1] - normalisation_limits[0]
    )
