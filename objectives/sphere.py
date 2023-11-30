from .objectivefunction import ContinuousObjectiveFunction
import numpy as np

class Sphere(ContinuousObjectiveFunction):
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
        return float(np.sum(x**2))

    def evaluate_n(self, X: np.ndarray) -> np.ndarray:
        X = X[:, : self.effective_dimensionality]
        return np.sum(X**2, axis=1)
