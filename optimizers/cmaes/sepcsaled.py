from typing import *
from .sepcsa import SepCMAES
import numpy as np
import math


class SepCMAESLED(SepCMAES):
    _gain_power_min: float
    _gain_power_max: float

    def __init__(self, mean: np.ndarray, sigma: float, lam: Optional[int] = None):
        dim = len(mean)
        self._s_el = np.zeros(dim)
        self._gamma_el = np.zeros(dim)
        super().__init__(mean=mean, sigma=sigma, lam=lam)

        if self.beta_hat == 0.1:
            self._signal_thr = (0.436 + 0.301 * np.log(self._dim)) * (
                0.213 + 0.278 / np.sqrt(self._lam)
            )
        elif self.beta_hat == 0.05:
            self._signal_thr = (0.215 + 0.207 * np.log(self._dim)) * (
                0.167 + 0.445 / np.sqrt(self._lam)
            )
        elif self.beta_hat == 0.01:
            self._signal_thr = (0.194 + 0.098 * np.log(self._dim)) * (
                0.055 + 0.534 / np.sqrt(self._lam)
            )
        elif self.beta_hat == 0.005:
            self._signal_thr = (0.105 + 0.035 * np.log(self._dim)) * (
                0.056 + 1.198 / np.sqrt(self._lam)
            )
        elif self.beta_hat == 0.001:
            self._signal_thr = (0.032 + 0.006 * np.log(self._dim)) * (
                0.031 + 1.657 / np.sqrt(self._lam)
            )
        else:
            raise Exception()

    def _update_mask(self):
        s = self.signals
        max_signal = np.max(s)
        gain_power = (
            max_signal * (self._gain_power_max - self._gain_power_min) + self._gain_power_min
        )
        gain = 10**gain_power

        def sigmoid(x, gain):
            return 1 / (1 + np.exp(-gain * x))

        v = sigmoid(s - self._signal_thr, gain) / sigmoid(1, gain)

        self._v_sum = np.sum(v)
        self._v = v

    def _update_C(self, y_k: np.ndarray):
        super()._update_C(y_k=y_k)

    @property
    def gain_power_min(self) -> float:
        return self._gain_power_min

    @gain_power_min.setter
    def gain_power_min(self, gain_power_min: float):
        self._gain_power_min = gain_power_min

    @property
    def gain_power_max(self) -> float:
        return self._gain_power_max

    @gain_power_max.setter
    def gain_power_max(self, gain_power_max: float):
        self._gain_power_max = gain_power_max

    @property
    def c_1(self) -> float:
        c_1 = 2 / ((self.v_sum + 1.3) ** 2 + self._mu_w)
        return (self.v_sum + 2) / 3 * c_1

    @property
    def c_mu(self) -> float:
        c_mu = np.min(
            [
                1 - self.c_1,
                2 * (self._mu_w - 2 + 1 / self._mu_w) / ((self.v_sum + 2) ** 2 + self._mu_w),
            ]
        )
        return (self.v_sum + 2) / 3 * c_mu

    @property
    def c_sigma(self) -> float:
        return (self._mu_w + 2) / (self.v_sum + self._mu_w + 5)

    @property
    def d_sigma(self) -> float:
        return (
            1 + self.c_sigma + 2 * np.maximum(0, np.sqrt((self._mu_w - 1) / (self.v_sum + 1)) - 1)
        )

    @property
    def c_c(self) -> float:
        return (4 + self._mu_w / self.v_sum) / (self.v_sum + 4 + 2 * self.mu_w / self.v_sum)

    def _tell(self, x_k: np.ndarray) -> None:
        super()._tell(x_k=x_k)

    def _update_step_size(self):
        self._sigma = self._sigma * np.exp(
            (self.c_sigma / self.d_sigma) * (np.sum(self.p_sigma**2) / np.sum(self.p_v) - 1)
        )  # line.10

    def _update_p_sigma(self, z_w: np.ndarray):
        coe1, coe2 = 1 - self.c_sigma, np.sqrt(self.c_sigma * (2 - self.c_sigma))
        self._p_sigma = (
            coe1 * self.p_sigma + coe2 * np.sqrt(self.mu_w) * np.sqrt(self._v) * z_w
        )  # line.6 where B = I

    def _get_H_sigma(self):
        rhs = np.sum(self.p_sigma**2) / (1 - (1 - self.c_sigma) ** (2 * self._t))
        lhs = (2 + (4 / (self.v_sum + 1))) * (np.sum(self._p_v))
        H_sigma = 1 if rhs < lhs else 0  # line.7
        return H_sigma
