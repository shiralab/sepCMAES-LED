import numpy as np
from scipy.stats import special_ortho_group


def generate_rotation_matrix(dim: int, seed: int) -> np.ndarray:
    np.random.seed(seed)
    rotation_matrix = special_ortho_group.rvs(dim)
    return rotation_matrix


def get_lambda(dim: int, alpha: float) -> np.ndarray:
    lam = [alpha ** (0.5 * i / (dim - 1)) for i in range(dim)]
    return np.diag(lam)


def get_T_osz(x: np.ndarray) -> np.ndarray:
    x_hat = np.where(x != 0, np.log(np.abs(x)), 0)
    c1 = np.where(x > 0, 10, 5.5)
    c2 = np.where(x > 0, 7.9, 3.1)

    return np.sign(x) * np.exp(x_hat + 0.049 * (np.sin(c1 * x_hat) + np.sin(c2 * x_hat)))


def get_T_asy(x: np.ndarray, beta: float) -> np.ndarray:
    i_1 = np.arange(x.shape[0])
    D_1 = x.shape[0] - 1
    return np.where(x > 0.0, x ** (1 + beta * i_1 / D_1 * np.sqrt(x)), x)


def get_fpen(x: np.ndarray) -> float:
    return np.sum(np.maximum(0, np.abs(x) - 5) ** 2)
