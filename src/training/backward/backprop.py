import numpy as np
import math as math
from typing import Callable


Array2D = np.ndarray
Array1D = np.ndarray


def delta_k(d: Array1D, x_k: Array1D) -> Array1D: 
    """
    Funzione che calcola il delta k vettore di un pattern, quindi una componente fondamentale per calcolare la backprop sull'hidden layer

    Args:
        d: vettore target
        x_k: vettore output layer
        (opzionale) f_deactiv: derivata della funzione di attivazione, nel caso dell'output layer è linaere quindi ritorno al x_k
        x_j: vettore hidden layer

    Ritorna: 
        dk: vettore dei delta k 
    """
    # Inizializza vettore delta k vuoto
    dk = np.zeros(x_k.size)

    # Aggiorna il vettore deltak con la differenza tra target e previsione * la derivata della funzione lineare quindi 1
    dk = (d - x_k) # * 1 # <- Versione vettorializzata

    """
    for kunit in range(x_k.size):
        # Aggiorna il vettore deltak con la differenza tra target e previsione * la derivata della funzione lineare quindi 1
        dk[kunit] = (d[kunit] - x_k[kunit]) * 1

    #print(f"-----  dk[kunit] = (d[kunit] - x_k[kunit]) * x_k[kunit]) = {d[kunit]} - {x_k[kunit]} * {x_k[kunit]}")
    """
    
    return dk



def delta_j(w_ji: Array2D, x_prev: Array1D, w_next: Array2D, 
            delta_next: Array1D, df_act: Callable) -> Array1D:
    """
    Funzione che calcola il vettore delta_j corrispondente all'hidden layer
    
    Args: 
        x_j: vettore hidden layer
        w_kj: matrice dei pesi tra l'ultimo hidden layer e l'otput layer
        delta_k: vettore dei delta k 
        x_k: vettore output layer
        fd: derivata della funzione di attivazione dell'hidden layer da passare

    Ritorna:
        dj: vettore dei delta j
    """
    # Inizializza vettore delta j vuoto

    n_units = w_ji.shape[1]
    dj = np.zeros(n_units)

    for unit in range(n_units):

        net_j = np.dot(w_ji[1:, unit], x_prev) + w_ji[0][unit] # <- Aggiunge il bias
        sommatoria = np.dot(w_next[unit + 1, :], delta_next) # <- Salta la prima riga dei bias

        #for kunit in range(delta_k.size):
        #    sum_parz += delta_k[kunit] * w_kj[junit, kunit]

        # La f_act ha un parametro aggiuntivo, se True, ritorna la derivata della funzione
        dj[unit] = sommatoria * df_act(net_j, True)


    return dj
    
    
def compute_delta_weights(delta: Array1D, x_prev: Array1D) -> Array2D:
    """
    Calcola la matrice dei delta per i pesi dato il delta e l'input.
    
    Args:
        delta: vettore delta del layer corrente
        x_prev: vettore del layer precedente
    
    Returns:
        delta_w: matrice dei delta pesi (include bias alla riga 0)
    """
    # Aggiungo un valore che consiste nel bias, maggiori informazioni in appunti_utili/ appunti_gestione_batch
    x_prev_biased = np.append(1, x_prev)
    # Outer product tra x_prev e delta
    delta_w = np.outer(x_prev_biased, delta)
    return delta_w


def compute_delta_all_layers(d: Array1D, x_k: Array1D, w_kj2: Array2D, 
                             x_j2: Array1D, w_j2j1: Array2D, x_j1: Array1D, 
                             w_j1i: Array2D, x_i: Array1D, 
                             fd: Callable) -> tuple[Array2D, Array2D, Array2D]:
    
    # 1. OUTPUT LAYER
    dk = delta_k(d, x_k)
    delta_wk = compute_delta_weights(dk, x_j2)
    
    # 2. HIDDEN LAYER 2
    dj2 = delta_j(w_j2j1, x_j1, w_kj2, dk, fd)
    delta_wj2j1 = compute_delta_weights(dj2, x_j1)
    
    # 3. HIDDEN LAYER 1
    dj1 = delta_j(w_j1i, x_i, w_j2j1, dj2, fd)
    delta_wj1i = compute_delta_weights(dj1, x_i)
    
    return delta_wk, delta_wj2j1, delta_wj1i, gradient_norm(dk, dj2, dj1)

def gradient_norm(dk, dj2, dj1):

    d_all = np.concatenate([dk, dj2, dj1])
    return math.sqrt(np.sum(d_all ** 2))