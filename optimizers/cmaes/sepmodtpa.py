from typing import *
from .sepcsa import SepCMAES
import numpy as np
import math


class SepCMAESModifiedTPA(SepCMAES):
    _sample_idx: int
    _z_t_norm: float
    _m_shift: np.ndarray
    _alpha_s: float

    @property
    def sample_idx(self) -> int:
        return self._sample_idx

    def __init__(self, mean: np.ndarray, sigma: float, lam: Optional[int] = None):
        super().__init__(mean, sigma, lam=lam)
        self._c_sigma = 0.3
        self._d_sigma = np.sqrt(self._dim)
        self._alpha_s = 0.0

        self._sample_idx = -1

    @property
    def d_sigma(self):
        return self._d_sigma

    def _ask(self, z: np.ndarray) -> np.ndarray:
        self._sample_idx += 1
        if (self._sample_idx == 0) and (self._t > 0):
            self._z_t_norm = np.linalg.norm(z)
            y = (
                self._z_t_norm
                * self._m_shift
                / np.sqrt(np.dot(self._m_shift.T, self._C_1).dot(self._m_shift))
            )
        elif (self._sample_idx == 1) and (self._t > 0):
            y = (
                - self._z_t_norm
                * self._m_shift
                / np.sqrt(np.dot(self._m_shift.T, self._C_1).dot(self._m_shift))
            )
        else:
            y = np.dot(self._D, z)  # ~ N(0, C)
        return self._m + self._sigma * y

    def tell(self, solutions: List[Tuple[np.ndarray, float, int]]):
        self._t += 1
        if len(solutions) != self.pop_size:
            raise ValueError("solutions length does not reach pop size")
        solutions.sort(key=lambda s: s[1])
        x_k = np.array([s[0] for s in solutions])  # ~ N(m, σ^2 C)  line.4
        sample_idxs = np.array([s[2] for s in solutions])
        self._tell(x_k=x_k, sample_idxs=sample_idxs)

    def _tell(self, x_k: np.ndarray, sample_idxs: np.ndarray) -> None:
        y_k = (x_k - self._m) / self._sigma  # ~ N(0, C)
        self._y_w = np.sum(y_k.T * self.weights, axis=1)  # line.5

        z_k = np.array([np.dot(self._D_1, y_i) for y_i in y_k])  # ~ N(0, I)
        z_w = np.sum(z_k.T * self.weights, axis=1)  # line.5

        self._update_mask()

        _m = self.mean
        self._m = np.sum(x_k.T * self.weights, axis=1)
        self._update_s_m(np.sign(self._m - _m))

        self._m_shift = self._m - _m

        self._update_evolution_path(z_w=z_w)

        self._update_C(y_k=y_k)  # line.9

        if self._t > 0:
            # update step-size
            rank_x1, rank_x2 = (
                self._ordering_to_rank(np.where(sample_idxs == 0)[0]),
                self._ordering_to_rank(np.where(sample_idxs == 1)[0]),
            )
            self._alpha_s = (1 - self._c_sigma) * self._alpha_s + self._c_sigma * (
                rank_x2 - rank_x1
            ) / (self._lam - 1)
            self._sigma = self._sigma * np.exp(self._alpha_s / self.d_sigma)

        self._eigen_decomposition()  # line.11

        self._sample_idx = -1

    def _get_H_sigma(self):
        return 1 if self._alpha_s < 0.5 else 0

    def _update_step_size(self):
        pass

    def _update_p_sigma(self, z_w: np.ndarray):
        pass

    def _accumulate_mask(self):
        pass

    def _ordering_to_rank(self, ordering: int) -> int:
        rank = np.linspace(0, self._lam - 1, self._lam)
        return rank[ordering]
