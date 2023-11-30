from typing import *
from .sepcsa import SepCMAES
import numpy as np
import math


class SepCMAESTPA(SepCMAES):
    _alpha_s: float = 0.0
    _tpa_alpha: float = 0.5
    _tpa_beta: float = 0.0
    _c_alpha: float = 0.3
    _alpha_dash: float = 0.5
    _y_w: np.ndarray

    def __init__(self, mean: np.ndarray, sigma: float, lam: Optional[int] = None):
        super().__init__(mean, sigma)

    def update_step_size(self, objctive_function):
        f_plus, f_minus = (
            objctive_function(self._m + self._alpha_dash * self._sigma * self._y_w),
            objctive_function(self._m - self._alpha_dash * self._sigma * self._y_w),
        )
        self._n_feval += 2
        alpha_act = -self._tpa_alpha + self._tpa_beta if f_minus < f_plus else self._tpa_alpha
        self._alpha_s = (1 - self._c_alpha) * self._alpha_s + self._c_alpha * alpha_act
        self._sigma = self._sigma * np.exp(self._alpha_s)

    def _update_step_size(self):
        pass

    def _tell(self, x_k: np.ndarray) -> None:
        y_k = (x_k - self._m) / self._sigma  # ~ N(0, C)
        self._y_w = np.sum(y_k.T * self.weights, axis=1)  # line.5

        z_k = np.array([np.dot(self._D_1, y_i) for y_i in y_k])  # ~ N(0, I)
        z_w = np.sum(z_k.T * self.weights, axis=1)  # line.5

        self._update_mask()
        self._accumulate_mask()

        _m = self.mean
        self._m = np.sum(x_k.T * self.weights, axis=1)

        self._update_s_m(np.sign(self._m - _m))

        self._update_evolution_path(z_w=z_w)

        self._update_C(y_k=y_k)  # line.9

        self._eigen_decomposition()  # line.11

    def _update_C(self, y_k: np.ndarray):
        """
        C is restricted to diagonal matrix
        :param x_k: np.ndarray
        """
        rank_one = self.p_c**2  # diagonal elements of p_c p_c^T
        rank_mu = np.sum((y_k**2).T * self.weights, axis=1)  # diagonal elements

        coe1, coe2 = (
            1 / self.mu_cov * self.c_cov,
            (1 - 1 / self.mu_cov) * self.c_cov,
        )

        d_ng = (-coe1 - coe2) * self.diagonal_C + coe1 * rank_one + coe2 * rank_mu

        d_ng_rankmu = -self.diagonal_C + rank_mu
        self._update_s_c(s_delta=np.sign(d_ng_rankmu))
        self._update_gamma(gamma_delta=np.ones(self._dim))
        signals_c = (
            self._s_c**2 / (self._gamma + self._EPS) * (self._beta_hat / (2 - self._beta_hat))
        )
        signals_m = (
            self._s_m**2 / (self._gamma + self._EPS) * (self._beta_hat / (2 - self._beta_hat))
        )
        self.signals = np.maximum(signals_c, signals_m)

        diagC = self.diagonal_C + d_ng
        self._C = np.diag(diagC)

    def _get_H_sigma(self):
        return 1
