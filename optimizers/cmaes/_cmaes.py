from typing import Optional
import numpy as np
from .._optimizer import _Gaussian


class _CMAES(_Gaussian):
    _weights: np.ndarray
    _mu: int
    _mu_w: float
    _c_sigma: float
    _d_sigma: float
    _p_sigma: np.ndarray
    _p_c: np.ndarray
    _c_c: float
    _mu_cov: float
    _c_cov_def: float
    _c_cov: float
    _chi_n: float
    _csa: bool

    def __init__(self, mean: np.ndarray, sigma: float, lam: Optional[int] = None):
        super().__init__(mean=mean, sigma=sigma)
        self._lam = lam if lam else 4 + int(3 * np.log(self._dim))
        self._mu = int(self.pop_size / 2)

        # CMA weights
        self._weights = self._generate_weights()
        self._mu_w = 1 / np.sum(self.weights**2)

        # step size control
        #   - learning rate for the accumulation of the step-size control
        self._c_sigma = (self._mu_w + 2) / (self._dim + self._mu_w + 5)
        self._d_sigma = (
            1 + self._c_sigma + 2 * np.maximum(0, np.sqrt((self._mu_w - 1) / (self._dim + 1)) - 1)
        )

        # evolution path
        self._p_sigma = np.zeros(self._dim)
        self._p_c = np.zeros(self._dim)
        #   - learning rate for accumulation of the rank-one update
        self._c_c = (4 + self._mu_w / self._dim) / (self._dim + 4 + 2 * self.mu_w / self._dim)
        #   - learning rate for rank-one update
        self._mu_cov = self._mu_w

        # rank-one evolution path
        self._c_c = (4 + self._mu_w / self._dim) / (self._dim + 4 + 2 * self._mu_w / self._dim)
        # rank-one learning rate
        self._c_1 = 2 / ((self._dim + 1.3) ** 2 + self._mu_w)
        # rank-mu learning rate
        self._c_mu = np.min(
            [
                1 - self._c_1,
                2 * (self._mu_w - 2 + 1 / self._mu_w) / ((self._dim + 2) ** 2 + self._mu_w),
            ]
        )

        # E||N(0, I)||
        self._chi_n = np.sqrt(self._dim) * (1 - (1 / (4 * self._dim)) + 1 / (21 * (self._dim**2)))

    def _generate_weights(self) -> np.ndarray:
        """
        CMA weight
        w_i = \frac{\ln(\mu + 1) - \ln(i)}{\sum_{j=1}^{\mu} \ln(\mu + 1) - \ln(j)}
        :return: np.ndarray
        """
        w_i = np.zeros(self.pop_size)
        w_i[: self.mu] = np.array(
            [(np.log(self.mu + 1) - np.log(i)) for i in range(1, self.mu + 1)]
        )
        w_i /= np.sum(w_i)
        return w_i

    @property
    def mu(self) -> int:
        return self._mu

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @property
    def mu_w(self) -> float:
        return self._mu_w

    @property
    def mu_cov(self) -> float:
        return self._mu_cov

    @property
    def c_1(self) -> float:
        return self._c_1

    @property
    def c_mu(self) -> float:
        return self._c_mu

    @property
    def c_sigma(self) -> float:
        return self._c_sigma

    @property
    def d_sigma(self) -> float:
        return self._d_sigma

    @property
    def c_c(self) -> float:
        return self._c_c

    @property
    def chi_n(self) -> float:
        return self._chi_n

    @property
    def p_sigma(self) -> np.ndarray:
        return self._p_sigma

    @property
    def p_c(self) -> np.ndarray:
        return self._p_c
