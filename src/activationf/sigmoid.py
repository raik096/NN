import numpy as np

### ACTIVATION FUNCTION SIGMA
#.  arg -> alpha is for the slop

def sigmaf(x, alpha=1, derivata: bool = False):

    if derivata: 
        s = sigmaf(x, alpha)
        return s * (1 - s)
    else:
        x_clipped = np.clip(x, -700, 700)
        return 1 / (1 + np.exp(-alpha * x_clipped))