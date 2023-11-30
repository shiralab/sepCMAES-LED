from .objectivefunction import ContinuousObjectiveFunction
from .utils import generate_rotation_matrix, get_T_osz
import numpy as np


class Ellipsoid(ContinuousObjectiveFunction):
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
        self.coefficients = 10 ** (
            6 * np.arange(self.effective_dimensionality) / float(self.effective_dimensionality - 1)
        )

    def __call__(self, x: np.ndarray) -> float:
        super().__call__(x=x)
        x = x[: self.effective_dimensionality]
        x = x - self.x_opt
        x = x**2
        return float(np.sum(self.coefficients * x))


class RotatedEllipsoid(ContinuousObjectiveFunction):
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
        self.rotation_matrix = generate_rotation_matrix(
            dim=self.effective_dimensionality, seed=self.seed
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x[: self.effective_dimensionality]
        x = x - self.x_opt
        z = get_T_osz(np.dot(x, self.rotation_matrix))
        z = z**2
        return float(np.sum(self.coefficients * z))
