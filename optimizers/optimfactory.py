from typing import Optional, Tuple
import numpy as np
import sys
sys.path.append('../')
import optimizers
import random


class OptimizerFactory:
    _domains = {
        "Sphere": (-5, 5),
        "Rosenbrock": (-5, 5),
        "Ellipsoid": (-5, 5),
        "Ackley": (-5, 5),
        "AttractiveSector": (-5, 5),
        "SharpRidge": (-5, 5),
    }

    _optims = {
        "SepCMAES",
        "SepCMAESTPA",
        "SepCMAESModifiedTPA",
        "SepCMAESLED",
        "SepCMAESModifiedTPALED",
    }

    @classmethod
    def _get_domain(cls, name: str, dim: int, seed: int) -> Tuple[np.ndarray, float]:
        np.random.seed(seed + 100)
        random.seed(seed + 100)
        if name in [
            "Sphere",
            "Ellipsoid",
            "Rosenbrock",
            "Ackley",
            "AttractiveSector",
            "SharpRidge",
        ]:
            lower, upper = cls._domains[name][0], cls._domains[name][1]
            return (upper - lower) * np.random.rand(dim) + lower, 2
        else:
            raise NotImplementedError("No such function")

    @classmethod
    def get(cls, method: str, obj_name: str, dim: int, seed: int = 100, lam: Optional[int] = None):
        if method in cls._optims:
            init_mean, init_sigma = cls._get_domain(name=obj_name, dim=dim, seed=seed)
            return getattr(optimizers, method)(mean=init_mean, sigma=init_sigma, lam=lam)
        else:
            raise NotImplementedError("No such method")

    @classmethod
    def _is_continuous(cls, obj_name: str) -> bool:
        return obj_name in cls._domains.keys()
