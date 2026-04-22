import numpy as np


def tanh(x, alpha: float = 1.0, derivata: bool = False):
    x = np.clip(x, -50, 50)

    t = np.tanh(alpha * x)

    if derivata:
        return alpha * (1.0 - t * t)
    else:
        return t