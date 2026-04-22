import numpy as np
def leaky_relu(z, derivata: bool = False):
    if derivata:
        return np.where(z > 0, 1, 0.01)

    else:
        return np.where(z > 0, z, 0.01 * z)