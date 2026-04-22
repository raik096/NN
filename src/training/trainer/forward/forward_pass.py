import numpy as np
from typing import Callable

# Tipi utili per chiarezza
Array2D = np.ndarray
Array1D = np.ndarray



def forward_hidden(x_i: Array1D, w_ji: Array2D, f_act: Callable) -> Array1D:
    """
    Calcola l'output del hidden layer con act. func. sigma con array in ingresso x_i
    
    Args: 
        x_i: vettore input (n_features)
        w_ji: matrice pesi (n_features (contiene il bias), n_hidden)
        f_act: la funzione di attivazione per tutti (e soltanto) gli hidden layer
    Ritorna: 
        vettore attivazioni hidden layer (n_hidden)
    """

    n_hidden_units = w_ji.shape[1]
    # Inizializza vettore risultato
    x_j = np.zeros(n_hidden_units)

    # Salta la prima riga essendo il bias, che non ha un x_i a cui essere moltiplicato
    # il bias viene sommato successivamente il calcolo del prodotto scalare
    net = np.dot(x_i, w_ji[1:]) + w_ji[0] # <- Vettorializzata
    x_j = f_act(net)

    """ Versione non vettorializzata
    #print("inside forward hidden, x_j.size: ", x_j.shape, "w_ji.shape: ", w_ji.shape)
    for junit in range(n_hidden_units):
        
        # Salta la prima riga essendo il bias, che non ha un x_i a cui essere moltiplicato
        # il bias viene sommato successivamente il calcolo del prodotto scalare
        net = np.dot(x_i, w_ji[1:, junit]) + w_ji[0][junit] # <- Aggiunge il bias

        x_j[junit] = relu(net)

    #z_j = np.dot(w_ji.T, x_i)
    #x_j = relu(z_j)

    """
    return x_j


def forward_output(x_j: Array1D, w_kj: Array2D) -> Array1D:
    """
    Calcola l'output layer (lineare).
    
    Args:
        X1: vettore attivazioni hidden layer (n_hidden)
        K: matrice pesi output (n_hidden (compreso di bias), n_outputs)
    
    Ritorna:
        vettore predizioni
    """
    n_outputs = w_kj.shape[1]
    x_k = np.zeros(n_outputs)


    x_k = np.dot(x_j, w_kj[1:]) + w_kj[0] # <- Vettorializzata

    """ Non vettorializzata
    for kunit in range(n_outputs):

        x_k[kunit] = np.dot(x_j, w_kj[1:, kunit]) + w_kj[0][kunit] # <- Aggiunge il bias

    #x_k = np.dot(w_kj.T, x_j)
    """

    return x_k

    
    
def forward_all_layers(x_i: Array1D, w_j1i: Array2D,w_j2j1: Array2D, w_kj2: Array2D, f_act: Callable) -> tuple[Array1D, Array1D, Array1D]:
    
    x_j1 = forward_hidden(x_i, w_j1i, f_act)


    x_j2 = forward_hidden(x_j1, w_j2j1, f_act)


    return forward_output(x_j2, w_kj2), x_j2, x_j1