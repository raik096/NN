import numpy as np

def linear(x, m=1, q=0, derivata: bool = False):
    if derivata:
        # La derivata di mx + q è m (di solito 1, però m)
        # Restituiamo un array di 1 della stessa shape di x del valore m
        return np.full_like(x, m)
    else:
        return m * x + q