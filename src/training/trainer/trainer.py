# MAIN PROGETTO ML

import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Dict
from src.nn.nn import NN
from src.training.trainer.forward.forward_pass import *
from src.training.trainer.backward.backprop import compute_delta_all_layers_list
from src.utils import *

Array2D = np.ndarray
Array1D = np.ndarray

class Trainer(ABC):
    """
    La classe ha lo scopo di orchestrare le fasi di training in base ai parametri passati,
    il suo scopo è:
        1) Creare cartella relativa all'istanza di training, con gli storici e i dati interessanti
        2) Orchestrare il forward, update weights, backpropagation

    """

    def __init__ (self,
                 input_size: int,
                 units_list: list[int],
                 n_outputs: int,
                 f_act_hidden: Callable,
                 f_act_output: Callable,
                 learning_rate: float,
                 use_decay: bool,
                 decay_factor: float,
                 decay_step: int,
                 mini_batch_size: int,
                 epochs: int,
                 momentum: bool,
                 alpha_mom: float,
                 max_gradient_norm: float,
                 verbose: bool = False,      # <- Importante da togliere nella grid search
                 lambdal2: float = 1e-4,   
                 validation: bool = False,
                 early_stopping: bool = False,
                 epsilon: float = 0,
                 split: int = 0,
                 patience: int = 0,
                 d_min: float = 0,
                 d_max: float = 1 ):   # <- Caso standard per il monk


        self.f_act_hidden = f_act_hidden
        self.f_act_output = f_act_output
        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.momentum = momentum
        self.alpha_mom = alpha_mom
        self.use_decay = use_decay
        self.decay_factor = decay_factor
        self.decay_step = decay_step
        self.max_gradient_norm = max_gradient_norm
        self.verbose = verbose
        self.lambdal2 = lambdal2
        self.epoch = 0
        self.old_deltas = None
        self.early_stopping = early_stopping
        self.epsilon = epsilon
        self.patience = patience
        self.validation = validation
        self.d_min = d_min
        self.d_max = d_max

        # Inizializza la rete neurale
        self.neuraln = NN(
                    n_inputs = input_size, 
                    units_list = units_list, 
                    n_outputs = n_outputs, 
                    f_act_hidden = f_act_hidden,
                    f_act_output = f_act_output
                )
                
        self.tr_mee_history = [] 
        self.tr_mse_history = []

    @abstractmethod
    def fit(self, 
            tr_x: np.ndarray, 
            tr_d: np.ndarray,
            vl_x: np.ndarray = None,
            vl_d: np.ndarray = None,
            metric_fn: Callable = None,
            metric_mode: str = 'min'):
        """
        Metodo centrale della classe Train. Fa le seguenti cose:
            - Prende come argomenti:
                . Tutti i pattern dei valori in ingresso CUP (500, 14)
                . Tutti i corrispondenti pattern dei valori risultati CUP (500, 4)
                . metric_fn: Funzione che accetta (output, target) e ritorna un float
                    tipo compute_mee per CUP, compute_accuracy per MONK.
                . metric_mode, perché? 
                    1) Nel caso ad esempio della mee abbiamo bisogno di fermarci appena l'mee inizia a salire
                    2) Nel caso ad esempio dell'accuracy o di altre statistiche ci vogliamo fermare quando inizia
                        a scendere

            - Itera le epoche al cui interno si itera sui pattern, per ogni epoca:
                . Avvia il timer
                . Chiama il metodo interno run epoch che restituisce i risultati sull'errore corrente
                . Controlla se il criterio di fermata è soddisfatto, se sì ferma l'iterazione
        
        DEVE ESSERE IMPLEMENTATO DALLE SOTTOCLASSI
        
        """
        pass

    def _run_epoch(self, 
                   input_matrix: np.ndarray, 
                   d_matrix: np.ndarray, 
                   n_patterns: int) -> Dict[str, float]:
        """
        Metodo che nasce con l'esigenza di portare un po' di logica fuori dal train,
        runna una epoca, restitutuendo le informazioni sull'errore.
        Questo metodo gestisce la divisione della logica in base all'esplorazione tramite batch o online
        """

        indices = np.arange(n_patterns)
        np.random.shuffle(indices)

        epoch_mee, epoch_mse, epoch_grad = 0.0, 0.0, 0.0
        final_output = []

        # Crea una lista di batch quindi batch_deltas = [ [dwk], [dwj2], [dwj1]], 
        # in base a quanti sono le matrici dentro la lista dei pesi
        mb_size = self.mini_batch_size
        if mb_size > n_patterns or mb_size < 1:
            # Forza il full batch
            mb_size = n_patterns

        # Per gestire la logica ha bisogno di un accumulatore che ha size 1 se è online, size = n_patterns se è full
        mb_deltas_accumulator = [np.zeros_like(w) for w in self.neuraln.weights_matrix_list]
        patterns_in_current_mb = 0

        if self.momentum and self.old_deltas is None:
            self.old_deltas = [np.zeros_like(w) for w in self.neuraln.weights_matrix_list]
        
        # Scorre tutti gli indici dividendo il comportamento in base a se è batch oppure online
        for i, idx in enumerate(indices):

            x_pattern, d_pattern = input_matrix[idx], d_matrix[idx]

            layer_results, layer_nets = self.neuraln.forward_network(
                x_pattern, self.f_act_hidden, self.f_act_output
            )

            final_output.append(layer_results[-1])

            # L'asterisco serve per raggruppare in lista tutti i risultati
            # tranne, in questo caso, l'ultimo. Essendo specificato.
            # quindi deltas = [dwk, dwj2, dwj1], nel caso di una rete con 2 hidden layer
            deltas, grad_norm = compute_delta_all_layers_list(
                d = d_pattern,
                layer_results_list = layer_results,
                layer_net_list = layer_nets, 
                weights_matrix_list = self.neuraln.weights_matrix_list,
                x_pattern = x_pattern,
                f_act_hidden = self.f_act_hidden,
                f_act_output = self.f_act_output,
                max_norm_gradient_for_clipping = self.max_gradient_norm
            )
                        
            epoch_grad += grad_norm

            # Accumula i delta prima dell'aggiornamento dei pesi
            for layer_idx in range(len(mb_deltas_accumulator)):
                mb_deltas_accumulator[layer_idx] += deltas[layer_idx]
                
            patterns_in_current_mb += 1 

            # Se ho raggiunto la size del minibatch o ho raggiunto la fine del dataset
            if patterns_in_current_mb == mb_size or i == n_patterns - 1:

                # Media dei gradienti -> d / 1 se online
                avg_deltas = [d / patterns_in_current_mb for d in mb_deltas_accumulator]
            
                # Se si usa il momentum, nel primo run_epoch somma 0
                if self.momentum:
                    for layer_idx in range(len(avg_deltas)):
                        avg_deltas[layer_idx] += (self.alpha_mom * self.old_deltas[layer_idx])
                
                self.neuraln.update_weights(avg_deltas, eta = self.learning_rate, lambda_l2 = self.lambdal2)
                
                # Salva per momentum prossima epoca
                if self.momentum: 
                    self.old_deltas = avg_deltas

                mb_deltas_accumulator = [np.zeros_like(w) for w in self.neuraln.weights_matrix_list]
                patterns_in_current_mb = 0 

        # Se si usa il decadimento dinamico del learning rate
        if self.use_decay and self.epoch > 0 and self.epoch % self.decay_step == 0:
            self.learning_rate *= self.decay_factor        
        
        d_ordered = d_matrix[indices]

        epoch_mee = denorm_mean_euclidean_error(final_output, d_ordered, self.d_max, self.d_min)
        epoch_mse = denorm_mean_squared_error(final_output, d_ordered, self.d_max, self.d_min)

        #epoch_mee = mean_euclidean_error(final_output, d_ordered)
        #epoch_mse = mean_squared_error(final_output, d_ordered)

        return {
            'mee_tr': epoch_mee,
            'mse_tr': epoch_mse,
            'grad_norm': epoch_grad / n_patterns
        }
    
