import numpy as np
from typing import Callable
from src.activationf.linear import linear
from src.activationf.sigmoid import sigmaf
from src.activationf.relu import relu
from src.activationf.leaky_relu import leaky_relu
from src.activationf.tanh import tanh

Array2D = np.ndarray
Array1D = np.ndarray

class NN:
    def __init__(self,
                n_inputs: int,
                units_list: list[int],
                n_outputs: int,
                f_act_hidden: Callable,
                f_act_output: Callable):
        
        self.n_inputs = n_inputs
        self.units_list = list(units_list)
        self.n_outputs = n_outputs  
        self.f_act_hidden = f_act_hidden
        self.f_act_output = f_act_output
        
        self.weights_matrix_list: list[Array2D] = []
        self._generate_weights_matrix_list()

        # MODIFICA: Salviamo sia gli output che i net (pre-attivazione)
        num_layers = len(self.units_list) + 1
        self.layer_results_list: list[Array1D] = [None] * num_layers
        self.layer_net_list: list[Array1D] = [None] * num_layers 

    def run_nn(self, input):
        """
        Runna l'istanza delle rete neurale con lo stato attuale dei pesi memorizzati in memoria,
        ritorna una lista di risultati in numpy array
        """
        final_output = [] 
        for pattern in input:
        # Forward pass per singolo pattern
            layer_results, _ = self.forward_network(
                pattern, 
                self.f_act_hidden, 
                self.f_act_output
            )
            final_output.append(layer_results[-1])

        return np.array(final_output)

    def update_weights(self, delta_list: list[Array2D], eta, lambda_l2=1e-4):
        """
        Aggiorna i pesi con gradiente + regolarizzazione L2.

        lambda_l2: coefficiente di regolarizzazione (>= 0).
        """

        self.weights_matrix_list = [
            (w + eta * d) - eta * lambda_l2 * w
            for (w, d) in zip(self.weights_matrix_list, delta_list)
        ]

    def forward_network(self, x_pattern: Array1D, fun_act_hidden: Callable, 
                       fun_act_output: Callable) -> tuple[list[Array1D], list[Array1D]]:
        """
        Forward pass che salva sia output che net values.
        
        Returns:
            tuple: (layer_results_list, layer_net_list)
                - layer_results_list: output dopo attivazione per ogni layer
                - layer_net_list: net input (prima attivazione) per ogni layer
        """
        current_input = x_pattern
        
        for i, weights in enumerate(self.weights_matrix_list):
            # Calcola il net, e qui l'unica volta che lo calcola
            # net è un vettore dei net di quello strato
            net = np.dot(current_input, weights[1:]) + weights[0]
            
            # Salvo il net che viene aggiunto alla lista
            # per layer, quindi il net è unificato per tutto lo strato,
            # non più il net che viene calcolato da capo per ogni unità
            self.layer_net_list[i] = net
            
            # Controlla se è l'ultimo layer
            is_output_layer = (i == len(self.weights_matrix_list) - 1)
            
            if is_output_layer:
                output = fun_act_output(net)
            else:
                output = fun_act_hidden(net)
            
            # Salva l'output (post-attivazione)
            self.layer_results_list[i] = output
            current_input = output

        # Ritorna le liste
        return self.layer_results_list, self.layer_net_list

    def _generate_weights_matrix_list(self):
        """
        Lavora con self.units_list: una lista di interi corrispondenti al numero di unità,
        il numero di unità è in base a quanto è lunga questa lista di interi, es:
            - lista vuota, allora la rete ha 0 hidden layer
            - lista con un n: int, allora la rete ha un hidden layer grande n
            - ...
        Ritorna una lista di matrici di pesi
        """

        layer_sizes = [self.n_inputs] + self.units_list + [self.n_outputs]

        for i in range(len(layer_sizes) - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i+1] 

            if self.f_act_output == linear:
                # Buono per ReLu
                weights = np.random.randn(n_in, n_out)* np.sqrt(2.0 / n_in)
                #weights = self._add_bias(weights)

            elif (self.f_act_output == sigmaf or self.f_act_output == tanh) and (self.f_act_hidden == relu or self.f_act_hidden == leaky_relu):

                if i == len(layer_sizes) - 2:
                    # Glorot/Xavier uniform: range = ±sqrt(6 / (n_in + n_out))
                    limit = np.sqrt(6.0 / (n_in + n_out))
                    weights = np.random.uniform(-limit, limit, (n_in, n_out))

                # ---- Hidden layers (He initialization) ----
                else:
                    # He uniform: range = ±sqrt(6 / n_in)
                    he_limit = np.sqrt(6.0 / n_in)
                    weights = np.random.uniform(-he_limit, he_limit, (n_in, n_out))

                    #weights = self._add_bias(weights)

            else:
            
                # Buono per SIGMOIDE
                limit = np.sqrt(6 / (n_in + n_out))
                weights = np.random.uniform(-limit, limit, (n_in, n_out))
        
            
            weights = self._add_bias(weights)

            
            
            self.weights_matrix_list.append(weights)




    def _add_bias(self, x: Array2D) -> Array2D:
        """ 
        Aggiunge il bias, in testa alla matrice dei pesi, 
        come vettore di 1 lungo quanti i nodi nel layer di destinazione
        """

        bias_row = np.ones((1, x.shape[1]))
        return np.concatenate((bias_row, x), axis=0)