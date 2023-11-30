from typing import *
from ._cmaes import _CMAES
import numpy as np
import math


class SepCMAES(_CMAES):
    ## attributes for accumulation
    _s_m: np.ndarray
    _s_c: np.ndarray
    _gamma: np.ndarray
    _beta_hat: float = 0.01

    ## attributes for LED
    _p_v: np.ndarray
    _v: np.ndarray
    _v_sum: float

    ## attributes for sep-CMAES
    _c_cov: float
    _c_sigma: float
    _d_sigma: float
    _c_c: float
    _chi_n: float

    signals: np.ndarray

    def __init__(self, mean: np.ndarray, sigma: float, lam: Optional[int] = None):
        super().__init__(mean=mean, sigma=sigma, lam=lam)

        # attributes for accumulation
        self._s_m = np.zeros(self._dim)
        self._s_c = np.zeros(self._dim)
        self._gamma = np.zeros(self._dim)
        self.signals = np.zeros(self._dim)

        # attributes for LED
        self._p_v = np.zeros(self._dim)
        self._v = np.ones(self._dim)
        self._v_sum = self._dim

    ## attributes for accumulation

    @property
    def beta_hat(self) -> float:
        return self._beta_hat

    @beta_hat.setter
    def beta_hat(self, beta_hat: float):
        self._beta_hat = beta_hat if beta_hat > 0 else 0.01

    ## attributes for LED
    @property
    def v(self) -> np.ndarray:
        return self._v

    @property
    def p_v(self) -> np.ndarray:
        return self._p_v

    @property
    def v_sum(self) -> np.ndarray:
        return self._v_sum

    ## attributes for CMA-ES
    @property
    def c_1(self) -> float:
        return (self._dim + 2) / 3 * self._c_1

    @property
    def c_mu(self) -> float:
        return (self._dim + 2) / 3 * self._c_mu

    def _ask(self, z: np.ndarray) -> np.ndarray:
        y = np.dot(self._D, z)  # ~ N(0, C)
        return self._m + self._sigma * y  # ~ N(m, σ^2 C)

    def _tell(self, x_k: np.ndarray) -> None:
        y_k = (x_k - self._m) / self._sigma  # ~ N(0, C)
        z_k = np.array([np.dot(self._D_1, y_i) for y_i in y_k])  # ~ N(0, I)
        z_w = np.sum(z_k.T * self.weights, axis=1)  # line.5

        self._update_mask()
        self._accumulate_mask()

        _m = self.mean
        self._m = np.sum(x_k.T * self.weights, axis=1)
        self._update_s_m(np.sign(self._m - _m))

        self._update_evolution_path(z_w=z_w)

        self._update_C(y_k=y_k)  # line.9

        self._update_step_size()
        self._eigen_decomposition()  # line.11

    def _update_evolution_path(self, z_w: np.ndarray):
        """
        :param z_w: np.ndarray
        """
        self._update_p_sigma(z_w=z_w)

        H_sigma = self._get_H_sigma()

        coe1, coe2 = 1 - self.c_c, np.sqrt(self.c_c * (2 - self.c_c))
        self._p_c = coe1 * self.p_c + H_sigma * coe2 * np.sqrt(self.mu_w) * np.dot(
            self._D, z_w
        )  # line.8 where B = I

    def _get_H_sigma(self):
        rhs = np.sum(self.p_sigma**2) / ((1 - (1 - self.c_sigma) ** (2 * self._t)))
        lhs = (2 + (4 / (self._dim + 1))) * self._dim
        H_sigma = 1 if rhs < lhs else 0  # line.7
        return H_sigma

    def _update_p_sigma(self, z_w: np.ndarray):
        coe1, coe2 = 1 - self.c_sigma, np.sqrt(self.c_sigma * (2 - self.c_sigma))
        self._p_sigma = coe1 * self.p_sigma + coe2 * np.sqrt(self.mu_w) * z_w  # line.6 where B = I

    def _update_s_m(self, s_delta: np.ndarray):
        coe1, coe2 = 1 - self.beta_hat, np.sqrt(self.beta_hat * (2 - self.beta_hat))
        self._s_m = coe1 * self._s_m + coe2 * s_delta

    def _update_s_c(self, s_delta: np.ndarray):
        coe1, coe2 = 1 - self.beta_hat, np.sqrt(self.beta_hat * (2 - self.beta_hat))
        self._s_c = coe1 * self._s_c + coe2 * s_delta

    def _update_gamma(self, gamma_delta: np.ndarray):
        coe1, coe2 = (1 - self.beta_hat) ** 2, self.beta_hat * (2 - self.beta_hat)
        self._gamma = coe1 * self._gamma + coe2 * gamma_delta

    def _update_step_size(self):
        self._sigma = self._sigma * np.exp(
            (self.c_sigma / self.d_sigma) * (np.linalg.norm(self.p_sigma) / self._chi_n - 1)
        )

    def _eigen_decomposition(self):
        """
        C is restricted to diagonal matrix
        """
        self._D = np.diag(np.sqrt(self.diagonal_C))
        self._D_1 = np.diag(1 / np.sqrt(self.diagonal_C))
        self._C = self._D**2
        self._C_1 = np.diag(1 / self.diagonal_C)

    def _update_C(self, y_k: np.ndarray):
        """
        C is restricted to diagonal matrix
        :param x_k: np.ndarray
        """
        rank_one = self.p_c**2  # diagonal elements of p_c p_c^T
        rank_mu = np.sum((y_k**2).T * self.weights, axis=1)  # diagonal elements

        H_sigma = self._get_H_sigma()
        c_h = self.c_1 * (1 - H_sigma) * self.c_c * (2 - self.c_c)

        diagC = (
            (1 - (self.c_1 - c_h) - self.c_mu) * self.diagonal_C
            + self.c_1 * rank_one
            + self.c_mu * rank_mu
        )
        ng_rank_mu = rank_mu - self.diagonal_C
        self._update_s_c(s_delta=np.sign(ng_rank_mu))
        self._update_gamma(gamma_delta=np.ones(self._dim))
        signals_c = (
            self._s_c**2 / (self._gamma + self._EPS) * (self._beta_hat / (2 - self._beta_hat))
        )
        signals_m = (
            self._s_m**2 / (self._gamma + self._EPS) * (self._beta_hat / (2 - self._beta_hat))
        )
        self.signals = np.maximum(signals_c, signals_m)

        diagC = np.maximum(np.minimum(diagC, 1e20), 1e-20)
        self._C = np.diag(diagC)

    def _update_mask(self):
        v = np.ones(self._dim)
        self._v = v
        self._v_sum = np.sum(self._v)

    def _accumulate_mask(self):
        coe1, coe2 = (1 - self.c_sigma) ** 2, self.c_sigma * (2 - self.c_sigma)
        self._p_v = coe1 * self._p_v + coe2 * self._v
