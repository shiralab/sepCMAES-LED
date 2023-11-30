from .objectivefunction import ContinuousObjectiveFunction
from .utils import generate_rotation_matrix
import numpy as np


class Rosenbrock(ContinuousObjectiveFunction):
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
        self.z_coef = np.maximum(1, np.sqrt(self.effective_dimensionality) / 8.0)

    def __call__(self, x: np.ndarray) -> float:
        super().__call__(x=x)
        x = x[: self.effective_dimensionality]
        return self._f(x=x)

    def _z(self, x: np.ndarray) -> np.ndarray:
        return self.z_coef * (x - self.x_opt) + 1

    def _f(self, x: np.ndarray) -> float:
        z = self._z(x)
        return np.sum(100 * (z[:-1] ** 2 - z[1:]) ** 2 + (z[:-1] - 1.0) ** 2)
