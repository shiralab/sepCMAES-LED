from abc import ABC, abstractmethod
import numpy as np


class ContinuousObjectiveFunction(ABC):
    dimensionality: int
    effective_dimensionality: int
    permissible_range: float
    seed: int

    def __init__(
        self,
        dimensionality: int,
        effective_dimensionality: int,
        permissible_range: float,
        seed: int = 0,
    ):
        self.dimensionality = dimensionality
        self.effective_dimensionality = effective_dimensionality
        self.permissible_range = permissible_range
        self.seed = seed
        self.rnd = np.random.RandomState(self.seed)
        upper, lower = -5, 5
        self.x_opt = (upper - lower) * self.rnd.rand(self.effective_dimensionality) + lower

    def __call__(self, x: np.ndarray) -> float:
        if np.any(np.isnan(x)):
            raise ValueError("sample contains NaN.")

    def is_optimized(self, x: np.ndarray) -> bool:
        return (self(x=x) - self(x=self.x_opt)) < self.permissible_range
