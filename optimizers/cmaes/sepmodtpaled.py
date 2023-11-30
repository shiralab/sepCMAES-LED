from typing import *
from .sepmodtpa import SepCMAESModifiedTPA
import numpy as np
import math


class SepCMAESModifiedTPALED(SepCMAESModifiedTPA):
    _warmup: int = 50.0
    _signal_eps: float = 0.0
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
    def c_c(self) -> float:
        return (4 + self._mu_w / self.v_sum) / (self.v_sum + 4 + 2 * self.mu_w / self.v_sum)

    @property
    def d_sigma(self):
        return np.sqrt(self.v_sum)

    @property
    def c_sigma(self) -> float:
        return None

    def _get_H_sigma(self):
        return 1 if self._alpha_s < 0.5 else 0

    def _update_mask(self):
        s = self.signals
        max_signal = np.max(s)
        gain_power = ()
        gain_power = (
            max_signal * (self._gain_power_max - self._gain_power_min) + self._gain_power_min
        )
        gain = 10**gain_power

        def sigmoid(x, gain):
            return 1 / (1 + np.exp(-gain * x))

        v = sigmoid(s - self._signal_thr, gain) / sigmoid(1, gain)
        v = np.clip(v, self._signal_eps, 1)

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

    def _ask(self, z: np.ndarray) -> np.ndarray:
        self._sample_idx += 1
        
        if (self._sample_idx == 0) and (self._t > 0):
            self._z_t_norm = np.linalg.norm(z * self._v)
            _m_shift_v = self._m_shift * self._v
            y = (
                self._z_t_norm
                * self._m_shift
                / np.sqrt(np.dot(_m_shift_v.T, self._C_1).dot(_m_shift_v))
            )
        elif (self._sample_idx == 1) and (self._t > 0):
            _m_shift_v = self._m_shift * self._v
            y = (
                - self._z_t_norm
                * self._m_shift
                / np.sqrt(np.dot(_m_shift_v.T, self._C_1).dot(_m_shift_v))
            )
        else:
            y = np.dot(self._D, z)  # ~ N(0, C)
        return self._m + self._sigma * y