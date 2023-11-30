from .objectivefunction import ContinuousObjectiveFunction
from .utils import generate_rotation_matrix, get_lambda, get_T_osz
import numpy as np


class AttractiveSector(ContinuousObjectiveFunction):
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
        self.Q = generate_rotation_matrix(dim=self.effective_dimensionality, seed=self.seed)
        self.R = generate_rotation_matrix(dim=self.effective_dimensionality, seed=self.seed + 1)
        self.Lambda = get_lambda(dim=self.effective_dimensionality, alpha=10)
        # self.rotation_matrix = np.dot(self.Q, self.Lambda).dot(self.R)
        self.rotation_matrix = self.Lambda

    def __call__(self, x: np.ndarray) -> float:
        super().__call__(x=x)
        x = x[: self.effective_dimensionality]
        z = np.dot(self.rotation_matrix, (x - self.x_opt))
        s = np.where(z * self.x_opt > 0, 10**2, 1)
        return np.sum((z * s) ** 2)
