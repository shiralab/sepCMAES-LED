from abc import ABC, abstractmethod
from typing import *
import numpy as np
import warnings


class _Optimizer(ABC):
    _t: int
    _n_feval: int
    _dim: int
    _lam: int
    _snr: Tuple
    _beta: float
    _error: bool
    _EPS = 1e-300

    def __init__(self) -> None:
        super().__init__()
        self._t = 0
        self._n_feval = 0
        self._error = False

    @abstractmethod
    def ask(self):
        pass

    @abstractmethod
    def tell(self, x_k: np.ndarray):
        pass

    @property
    def num_of_feval(self) -> int:
        return self._n_feval

    @property
    def generations(self) -> int:
        return self._t

    @property
    def dimensionality(self) -> int:
        return self._dim

    @property
    def pop_size(self) -> int:
        return self._lam


class _Gaussian(_Optimizer):
    _m: np.ndarray
    _C: np.ndarray
    _D: np.ndarray
    _B: np.ndarray
    _sqrt_C_1: np.ndarray
    _D_1: np.ndarray
    _C_1: np.ndarray
    _sigma: float

    def __init__(self, mean: np.ndarray, sigma: float) -> None:
        super().__init__()

        self._dim = len(mean)
        self._m = mean
        self._C = np.identity(self._dim)
        self._D = np.identity(self._dim)
        self._B = np.identity(self._dim)
        self._sqrt_C_1 = np.identity(self._dim)
        self._C_1 = np.identity(self._dim)
        self._D_1 = np.identity(self._dim)
        self._sigma = sigma

        # SNR
        self.snr = (np.zeros(self._dim), 0.0)
        self._beta = self._dim ** (-0.5)

    def ask(self) -> np.ndarray:
        self._n_feval += 1
        z = np.random.randn(self._dim)  # ~ N(0, I)
        return self._ask(z=z)  # ~ N(m, σ^2 C)

    def tell(self, solutions: List[Tuple[np.ndarray, float]]):
        self._t += 1
        if len(solutions) != self.pop_size:
            raise ValueError("solutions length does not reach pop size")
        solutions.sort(key=lambda s: s[1])

        x_k = np.array([s[0] for s in solutions])  # ~ N(m, σ^2 C)  line.4
        self._tell(x_k=x_k)

    def _care_C_overflow(self, fn, x: np.ndarray):
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                return fn(x)
            except RuntimeWarning as e:
                if not self._error:
                    print(f"C[max={max(self.diagonal_C)}, min={min(self.diagonal_C)}")
                    self._error = True

                if max(self.diagonal_C) > 1:
                    return np.inf
                else:
                    return -np.inf

    @abstractmethod
    def _ask(self, x_k: np.ndarray):
        pass

    @abstractmethod
    def _tell(self, x_k: np.ndarray):
        pass

    @abstractmethod
    def _eigen_decomposition(self) -> np.ndarray:
        pass

    @property
    def mean(self) -> np.ndarray:
        return self._m.copy()

    @property
    def C(self) -> np.ndarray:
        return self._C.copy()

    @property
    def diagonal_C(self) -> np.ndarray:
        return np.diag(self._C)

    @property
    def m_F(self) -> np.ndarray:
        return self._C_1

    @property
    def C_F(self) -> np.ndarray:
        return self._care_C_overflow(fn=lambda x: (1 / 2) * (x**2), x=self._C_1)

    @property
    def m_sqrt_F(self) -> np.ndarray:
        return self.sqrt_C_1

    @property
    def C_sqrt_F(self) -> np.ndarray:
        return (1 / np.sqrt(2)) * self._C_1

    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def eig_values(self) -> np.ndarray:
        return np.diag(self._D)

    @property
    def condition_number(self) -> float:
        return max(self.eig_values) / min(self.eig_values)
