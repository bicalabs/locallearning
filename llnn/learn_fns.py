import numpy as np
from typing import Callable


LearnFn = Callable[[np.array, int, int], np.ndarray]


def linear(y: np.array, a: int, b: int) -> np.ndarray:
    return 1 + a * y - 1 / b


def inc_xor_decr(y: np.array, a: int, b: int) -> np.ndarray:
    return 1 + a * y - 1 / b
