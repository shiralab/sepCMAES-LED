from .objectivefunction import ContinuousObjectiveFunction
import numpy as np


class Ackley(ContinuousObjectiveFunction):
    def __init__(
        self,
        dimensionality: int,
        effective_dimensionality: int,
        permissible_range: float,
        seed: int = 0,
    ):
        super().__init__(
            dimensionality=dimensionality,
            effective_dimensionality=effective_dimensionality,
            permissible_range=permissible_range,
            seed=seed,
        )
        self.coef_a, self.coef_b, self.coef_c = 20.0, 0.2, 2 * np.pi

    def __call__(self, x: np.ndarray) -> float:
        super().__call__(x=x)
        x = x[: self.effective_dimensionality]
        x = x - self.x_opt

        return (
            20
            - 20 * np.exp(-0.2 * np.sqrt(np.mean(x**2)))
            + np.exp(1)
            - np.exp(np.mean(np.cos(2 * np.pi * x)))
        )
