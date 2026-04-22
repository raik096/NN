import numpy as np
import math as math
from typing import Callable, List, Optional

Array2D = np.ndarray
Array1D = np.ndarray


def delta_k(d: Array1D, x_k: Array1D, net_k: Array1D, f_act_output: Callable = None) -> Array1D:
    """
    Calcola delta per output layer.
    
    IMPORTANTE: Usa l'output POST-attivazione (x_k), non il net!
    
    Args:
        d: target
        x_k: output del layer (POST-attivazione)
        f_act_output: funzione di attivazione (se None, assume lineare)
    """
    error = d - x_k
    
    if f_act_output is None:
        # Funzione lineare derivata = 1
        return error
    else:
        # Per sigmoide: f'(net) = f(net) * (1 - f(net))
        # Dato che x_k = f(net), possiamo usare direttamente
        derivata = f_act_output(net_k, derivata = True)
        return error * derivata


def delta_j(w_next: Array2D, delta_next: Array1D, 
            net_j: Array1D, f_act_hidden: Callable) -> Array1D:
    """
    Delta per hidden layer.
        
    Args:
        x_j: output del layer corrente (POST-attivazione)
        w_next: pesi del layer successivo
        delta_next: delta del layer successivo
        f_act_hidden: funzione attivazione hidden
    """
    error_signal = np.dot(w_next[1:, :], delta_next)
    
    # Per sigmoide: f'(net) = f(net) * (1 - f(net))
    # Dato che x_j = f(net), possiamo usare direttamente x_j
    derivata = f_act_hidden(net_j, derivata = True)
    
    #return error_signal * f_act_hidden(x_j, derivata = True)
    return error_signal * derivata


def compute_delta_weights(delta: Array1D, x_prev: Array1D) -> Array2D:
    """
    Calcola la matrice dei delta per i pesi dato il delta e l'input.
    
    Args:
        delta: vettore delta del layer corrente
        x_prev: vettore del layer precedente
    
    Returns:
        delta_w: matrice dei delta pesi (include bias alla riga 0)
    """
    x_prev_biased = np.append(1, x_prev)
    delta_w = np.outer(x_prev_biased, delta)
    return delta_w


def compute_delta_all_layers_list(
    d: Array1D,
    layer_results_list: List[Array1D],    # <- I risultati del forward  
    layer_net_list: List[Array1D],        # <- Il calcolo del net su ogni layer
    weights_matrix_list: List[Array2D],
    x_pattern: Array1D,
    f_act_hidden: Callable,
    f_act_output: Callable,
    alpha_momentum: float = 0.0,
    max_norm_gradient_for_clipping: float = 10
) -> tuple[List[Array2D], float]:
    """
    Backpropagation usando gli OUTPUT (post-attivazione) invece dei net.
    """
    
    deltas_list = []
    weight_gradients_list = []
    num_layers = len(weights_matrix_list)

    # Scorro la lista del result list fatta come:
    # Layer 1 -> Layer 2 -> Layer 3 ...
    # La leggo al contrario
    for i in range(num_layers - 1, -1, -1):

        curr_output = layer_results_list[i]  # Output POST-attivazione
        curr_net = layer_net_list[i]          
        
        if i == 0:
            prev_output = x_pattern
        else:
            prev_output = layer_results_list[i - 1]
        
        # Calcola delta
        if i == num_layers - 1:
            # Output layer: usa l'output POST-attivazione
            delta = delta_k(d, curr_output, curr_net, f_act_output)
        else:
            w_next = weights_matrix_list[i + 1]
            delta_next = deltas_list[-1]
            delta = delta_j(w_next, delta_next, curr_net, f_act_hidden)
        
        deltas_list.append(delta)
        grad = compute_delta_weights(delta, prev_output)
        
        weight_gradients_list.append(grad)

    weight_gradients_list.reverse()
    
    weight_gradients_list, grad_norm = gradient_norm_with_clipping(
        weight_gradients_list, max_norm_gradient_for_clipping
    )
    
    return weight_gradients_list, grad_norm


def gradient_norm_with_clipping(grad_list: List[Array2D], max_norm: float) -> tuple[List[Array2D], float]:
    """ 
    Calcola la norma globale del gradiente con clipping.
    """
    total_sum = 0.0
    for g in grad_list:
        total_sum += np.sum(g ** 2)
    
    current_norm = math.sqrt(total_sum)
        
    if current_norm > max_norm:
        scaling_factor = max_norm / (current_norm + 1e-6)
        grad_list = [g * scaling_factor for g in grad_list]
        return grad_list, max_norm
    
    return grad_list, current_norm