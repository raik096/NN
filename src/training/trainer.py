# MAIN PROGETTO ML

import numpy as np
import time
from src.nn.nn import NN
from src.training.forward.forward_pass import *
from src.training.backward.backprop import *
from src.utils import visualization as vs


# Tipi utili per chiarezza
Array2D = np.ndarray
Array1D = np.ndarray


class Trainer:
    """
    La classe ha lo scopo di orchestrare le fasi di training in base ai parametri passati,
    il suo scopo è:
        1) Creare cartella relativa all'istanza di training, con gli storici e i dati interessanti
        2) Orchestrare il forward, update weights, backpropagation

    """

    def __init__ (self,
                 input_size: int,
                 n_hidden1: int,
                 n_hidden2: int,
                 n_outputs: int,
                 f_act: Callable,
                 learning_rate: float,
                 batch: bool,
                 epochs: int,
                 epsilon: float,
                 patience: int):
        

        self.f_act = f_act
        self.batch = batch
        self.epochs = epochs
        self.epsilon = epsilon
        self.patience = patience
        #self.stopping_criteria = stopping_criteria

        # Inizializza la rete neurale
        self.neuraln = NN(
                        input_size, 
                        n_hidden1, 
                        n_hidden2, 
                        n_outputs, 
                        f_act,
                        learning_rate)
        
        self.mee_error_history = [] # <- Quella per la CUP, meno sensibile agli outliers
        self.mse_error_history = []
        # self.total_error_array = []


    def train(self, input_matrix, d_matrix):

        patterns = input_matrix.shape[0]
        gradient_misbehave = 0
        patience = 3
        epoch = 0
        # Inizializza il vettore dove ogni elemento è la sommatoria degli errori dei pattern per epoca
        self.mee_error_history = []
        self.mse_error_history = []
        # self.total_error_array = np.zeros(self.epochs)  ## 



        #total_error_array = np.zeros(self.epochs)

        prev_gradient_norm_epoch = None
        start_time = time.perf_counter()
        print(f"Inizio training...")


        # -------------- CICLO EPOCHS ------------------
        
        # Aggiunge lo STOPPING CRITERIA basato sulla norma del gradiente, che se non scende di molto
        # oltre un certo numero di epoche (= patience) allora ritorna il risultato
        while gradient_misbehave < patience and epoch < self.epochs:
        #for epoch in range(self.epochs):
            epoch += 1
            mee_pattern_error = 0.0
            mse_pattern_error = 0.0
            gradient_norm_epoch = 0.0
            # total_error = np.zeros(self.neuraln.w_kj2.shape[1]) 
            # MSE_k = np.zeros(self.neuraln.w_kj2.shape[1])

            # SHUFFLE (se Online)
            indices = np.arange(patterns)
            if not self.batch:
                np.random.shuffle(indices)

            # BATCH
            if self.batch == True:
                delta_wk_epoch = np.zeros_like(self.neuraln.w_kj2)
                delta_wj2j1_epoch = np.zeros_like(self.neuraln.w_j2j1)
                delta_wj1i_epoch = np.zeros_like(self.neuraln.w_j1i)
            
            # -------------- INIZIO CICLO PATTERN ------------------
            # Sia online o batch, calcola i delta sul pattern
            for idx_pattern in indices:
                
                x_i = input_matrix[idx_pattern]
                d = d_matrix[idx_pattern]

                self.neuraln.forward(input_matrix[idx_pattern])
                    
                mee_pattern_error += (np.sum((d - self.neuraln.x_k) ** 2)) ** 0.5
                mse_pattern_error += (np.sum((d - self.neuraln.x_k) ** 2))

                deltas = compute_delta_all_layers(
                                                d,
                                                self.neuraln.x_k,
                                                self.neuraln.w_kj2, 
                                                self.neuraln.x_j2,
                                                self.neuraln.w_j2j1,
                                                self.neuraln.x_j1,
                                                self.neuraln.w_j1i,
                                                x_i,
                                                self.f_act
                                                )
    
                    #print(delta_wk)
                    # MSE_k += (d - self.neuraln.x_k) ** 2

                dwk, dwj2, dwj1, grad_norm = deltas
                gradient_norm_epoch += grad_norm
                
                # ONLINE: Aggiorna i pesi ad ogni pattern
                if self.batch == False:
                    self.neuraln.update_weights(dwk, dwj2, dwj1)

                # BATCH: Accumula i delta su tutti i pattern, li aggiorna ad ogni epoca
                if self.batch == True:
                    delta_wk_epoch += dwk
                    delta_wj2j1_epoch += dwj2
                    delta_wj1i_epoch += dwj1

            # -------------- FINE CICLO PATTERN ------------------

            if self.batch == True:

                # (?)
                delta_wk_epoch    /= patterns
                delta_wj2j1_epoch /= patterns
                delta_wj1i_epoch  /= patterns  
                
                self.neuraln.update_weights(delta_wk_epoch, 
                                            delta_wj2j1_epoch, 
                                            delta_wj1i_epoch)
                

            mean_epoch_mee_error = mee_pattern_error / patterns
            mean_epoch_mse_error = mse_pattern_error / patterns

            self.mee_error_history.append(mean_epoch_mee_error)
            self.mse_error_history.append(mean_epoch_mse_error)

            # dà un'idea dell'errore medio per ogni neurono di output,
            #MSE_k = np.sqrt(MSE_k) / patterns
            #MSE_tot = 0
            #for i in range(self.neuraln.w_kj2.shape[1]):
            #    MSE_tot += MSE_k[i]
            #total_error = MSE_tot / self.neuraln.w_kj2.shape[1]
            #self.total_error_array[epoch] = total_error
            
            current_grad_norm = gradient_norm_epoch / patterns

            if prev_gradient_norm_epoch is not None:

                diff = abs(current_grad_norm - prev_gradient_norm_epoch) / (prev_gradient_norm_epoch + 1e-10)
                if diff < self.epsilon:
                    gradient_misbehave += 1
                else:
                    gradient_misbehave = 0

            prev_gradient_norm_epoch = current_grad_norm

            print("|| epooch n° ", epoch, ", total mee error: ", mean_epoch_mee_error, " ||")


        tempo_di_training = time.perf_counter() - start_time


        print(" Total mee error: ", mean_epoch_mee_error) 
        print(" Total mse error: ", mean_epoch_mse_error) 
        print(" Tempo di training: ", tempo_di_training)



        print("\n--- Training Completato ---\n")
                
        # Per PLOT
        vs.plot_errors(self, tempo_di_training)
        # self.plot_errors_separate(epoche) # Se preferisci grafici separati

