from typing import *
from .sepcsaled import SepCMAESLED
import numpy as np
import math


class SepCMAESLEDOracle(SepCMAESLED):
    oracle: int

    def __init__(self, mean: np.ndarray, sigma: float, lam: Optional[int] = None):
        super().__init__(mean=mean, sigma=sigma, lam=lam)

    def _update_mask(self):
        v = np.ones(self._dim)
        v[self.oracle :] = 0
        self._v = v
        self._v_sum = np.sum(self._v)

    def _update_step_size(self):
        self._sigma = self._sigma * np.exp(
            (self.c_sigma / self.d_sigma) * ((np.linalg.norm(self.p_sigma) ** 2) / self.oracle - 1)
        )  # line.10
