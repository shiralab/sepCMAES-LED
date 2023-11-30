from .objectivefunction import ContinuousObjectiveFunction
from .utils import generate_rotation_matrix, get_lambda
import numpy as np


class SharpRidge(ContinuousObjectiveFunction):
    def __init__(
        self,
        dimensionality: int,
        effective_dimensionality: int,
        permissible_range: float = 1e-10,
        seed: int = 0,
    ):
        super().__init__(
            dimensionality=dimensionality,
            effective_dimensionality=effective_dimensionality,
            permissible_range=permissible_range,
            seed=seed,
        )

    def __call__(self, x: np.ndarray) -> float:
        super().__call__(x=x)
        x = x[: self.effective_dimensionality]
        x = x - self.x_opt
        x = x**2
        return float(x[0] + 100 * np.sqrt(np.sum(x[1:])))
