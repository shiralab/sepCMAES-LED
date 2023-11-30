from typing import Tuple
from ._cmaes import _CMAES
import numpy as np


class CMAES(_CMAES):
    _csa: bool
    _snr_el: Tuple[np.ndarray, np.ndarray]
    _beta_hat: float

    def __init__(self, mean: np.ndarray, sigma: float):
        super().__init__(mean=mean, sigma=sigma)

        self._c_cov = self._c_cov_def

        # element-wise SNR
        self.snr_el = (np.zeros(self._dim), np.zeros(self._dim))
        self._beta_hat = 1 / self._dim

    @property
    def snr_el(self) -> np.ndarray:
        return (self._snr_el[0] ** 2) / (self._snr_el[1] + 1e-8)

    @snr_el.setter
    def snr_el(self, snr_el: Tuple[np.ndarray, np.ndarray]):
        self._snr_el = snr_el

    @property
    def v(self) -> np.ndarray:
        # return self._sigmoid(x=self.snr_el)
        return np.tanh(self.snr_el) + 1e-8

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-1 * (self.snr_el - 1)))

    def _ask(self, z: np.ndarray) -> np.ndarray:
        y = np.dot(self._B, self._D).dot(z)  # ~ N(0, C)
        return self._m + self._sigma * y  # ~ N(m, σ^2 C)

    def _tell(self, x_k: np.ndarray) -> None:
        y_k = (x_k - self._m) / self._sigma  # ~ N(0, C)
        y_w = np.sum(y_k.T * self._weights, axis=1)  # eq. (41)

        diagonal_C_ = self.diagonal_C

        self._m = np.sum(x_k.T * self._weights, axis=1)  # correspond to eq. (42) where _c_m = 1

        self._update_evolution_path(y_w=y_w)

        self._update_C(y_k=y_k)
        self._update_snr_el(s_delta=np.sign(self.diagonal_C - diagonal_C_))

        if self._csa:
            self._sigma = self._sigma * np.exp(
                (self.c_sigma / self.d_sigma) * (np.linalg.norm(self.p_sigma) / self.chi_n - 1)
            )  # eq. (44)

        self._eigen_decomposition()

    def _update_snr_el(self, s_delta: np.ndarray):
        coe1, coe2 = 1 - self._beta_hat, np.sqrt(self._beta_hat * (2 - self._beta_hat))
        self.snr_el = (
            coe1 * self._snr_el[0] + coe2 * s_delta,
            (coe1**2) * self._snr_el[1] + (coe2**2),
        )

    def _update_evolution_path(self, y_w: np.ndarray) -> None:
        coe1, coe2 = 1 - self.c_sigma, np.sqrt(self.c_sigma * (2 - self.c_sigma))
        self.p_sigma = coe1 * self.p_sigma + coe2 * np.sqrt(self._mu_w) * np.dot(
            self._sqrt_C_1, y_w
        )  # eq. (43)

        rhs = np.linalg.norm(self.p_sigma) / np.sqrt(1 - (1 - self.c_sigma) ** (2 * self._t))
        lhs = (1.4 + (2 / (self._dim + 1))) * self.chi_n
        H_sigma = 1 if rhs < lhs else 0

        coe1, coe2 = 1 - self.c_c, np.sqrt(self.c_c * (2 - self.c_c))
        self._p_c = coe1 * self._p_c + H_sigma * coe2 * np.sqrt(self._mu_w) * y_w  # eq. (47)

    def _update_C(self, y_k: np.ndarray) -> None:
        rank_one = np.outer(self._p_c, self._p_c)
        rank_mu = np.sum(np.array([w * np.outer(y, y) for w, y in zip(self._weights, y_k)]), axis=0)

        coe1, coe2 = (
            1 / self._mu_cov * self._c_cov,
            (1 - 1 / self._mu_cov) * self._c_cov,
        )
        self._C = (1 - coe1 - coe2) * self._C + coe1 * rank_one + coe2 * rank_mu

    def _eigen_decomposition(self) -> None:
        eig_values, B = np.linalg.eigh(self._C)
        assert np.all(eig_values > 0)

        self._D = np.diag(np.sqrt(eig_values))
        self._B = B
        self._D_1 = np.diag(1 / np.sqrt(eig_values))
        self._C_1 = np.dot(B, np.diag(1 / eig_values**2)).dot(B.T)
        self._sqrt_C_1 = np.dot(B, np.diag(1 / np.sqrt(eig_values))).dot(B.T)
