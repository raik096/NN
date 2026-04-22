import numpy as np
from src.activationf.sigmoid import *
from src.activationf.relu import *

def backprop_output(O_u, delta_t):
    DelW=O_u * delta_t

    return DelW