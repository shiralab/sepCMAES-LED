from typing import List
from .objectivefunction import ContinuousObjectiveFunction
import sys
sys.path.append('../')
import objectives
import numpy as np

# public symbols
__all__ = ['ContinuousObjectiveFunctionFactory']


class ContinuousObjectiveFunctionFactory:
    @classmethod
    def get(
        cls,
        name: str,
        dimensionality: int,
        effective_dimensionality: int,
        terminate_condition: float,
        seed: int,
    ) -> ContinuousObjectiveFunction:

        return getattr(objectives, name)(
            dimensionality, effective_dimensionality, terminate_condition, seed
        )


class DistributionInitialValueFactory:
    @classmethod
    def get(cls, name: str, dimensionality: int, seed: int) -> tuple:
        np.random.seed(seed + 100)
        if name in [
            "Ackley",
            "Sphere",
            "Ellipsoid",
            "Rosenbrock",
            "AttractiveSector",
            "SharpRidge",
        ]:
            lower, upper = -5.0, 5.0
            return (upper - lower) * np.random.rand(dimensionality) + lower, 2.0
