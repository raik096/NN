import numpy as np
from src.training.forward.forward_pass import  forward_all_layers
from src.activationf.relu import relu

from typing import Callable
# Tipi utili per chiarezza
Array2D = np.ndarray
Array1D = np.ndarray

def add_bias(x: Array2D) -> Array2D:
    """ 
    Aggiunge il bias, in testa alla matrice dei pesi, 
    come vettore di 1 lungo quanti i nodi nel layer di destinazione
    """
    # Vettore di 1
    bias_row = np.ones((1, x.shape[1]))

    return np.concatenate((bias_row, x), axis=0)

"""
Cose da considerare:

    - tipi dei tensori:
    Impostare il tipo dei tensori in cui caricare i pesi del modello.
    float32 → più preciso, più lento, più memoria
    bfloat16 → meno memoria, più veloce, quasi stessa qualità per i LLM

"""

class NN:
    """
    Classe della rete neurale di default feedforward a 2 hidden layers. 
    In futuro proveremo a renderlo parametrizzabile sugli hidden layers.
    Input Layer -> Hidden Layer 1 -> Hidden Layer 2 -> Output Layer

    pro:
        Ogni istanza della classe NN mantiene in memoria i propri valori legati ai vettori risultati
        e alle matrici dei pesi

    Mantiene in memoria:
        - Le matrici dei pesi
        - Vettore risultato x_k
        - Vettore risultato x_j2
        - Vettore risultato x_j1
    """

    def __init__(self,
                 n_inputs: int,
                 n_hidden1: int,
                 n_hidden2: int,
                 n_outputs: int,
                 f_act: Callable,
                 learning_rate: float):
        
        """
        Costruttore della rete neurale 
        ad ora l'unico input del costruttore obbligatorio è n_inputs.
        Da aggiungere iperparam:
            momentum, thikonov, margini inizializzazione dei pesi
            cose non legate alla backpropagation.

        Args:
            n_inputs      : Numero di feature di input
            n_hidden1     : Numero di neuroni nel primo hidden layer
            n_hidden      : Numero di neuroni nel secondo hidden layer
            n_outputs     : Numero di neuroni nell'output layer
            learning_rate : eta
        """

        self.n_inputs = n_inputs  
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_outputs = n_outputs  
        self.f_act = f_act          # Legata agli hidden layer
        self.learning_rate = learning_rate
        
        self.w_j1i = np.random.randn(self.n_inputs, n_hidden1) * np.sqrt(2.0 / n_inputs)
        self.w_j2j1 = np.random.randn(n_hidden1, n_hidden2) * np.sqrt(2.0 / n_hidden1)
        self.w_kj2 = np.random.randn(n_hidden2, n_outputs) * np.sqrt(2.0 / n_hidden2)

        self.w_j1i = add_bias(self.w_j1i)
        self.w_j2j1 = add_bias(self.w_j2j1)
        self.w_kj2 = add_bias(self.w_kj2)

        self.x_j1  =  0
        self.x_j2  =  0
        self.x_k   =  0

    # Adesso prende come argomento matrici a cui poi applica quello che deve applicare

    def update_weights(self,
                    delta_wk: Array2D,
                    delta_wj2j1: Array1D,
                    delta_wj1i: Array1D):
        
        # ONLINE
        # --- Aggiornamento Pesi Output (w_kj2) - Versione vettorializzata ---
        
        # Prende tutta la riga 0 e viene sommata con il vettore dk moltiplicato per lo scalare lear.rate
        # Aggiorna il bias
        #self.w_kj2[0] += self.learning_rate * dk # Versione vettorializzata

        # Il metodo outer restituisce una matrice con shape (self.x_j2, dk), e deve coincidere con
        #   la matrice che sta aggiornando, in questo caso saltando la riga dei bias precedentemente
        #   aggiornata.
        #self.w_kj2[1:, :] += self.learning_rate * np.outer(self.x_j2, dk)

        self.w_kj2 += self.learning_rate * delta_wk

        # --- Aggiornamento Pesi Output (w_kj2) - Versione non vettorializzata ---

        """
        for kunit in range(self.w_kj2.shape[1]):

            # Aggiorna Bias (Riga 0)
            self.w_kj2[0][kunit] += self.learning_rate * dk[kunit] 

            for junit in range(self.w_kj2.shape[0] - 1): 
                print("shape dk: ", dk.shape, "shape x_j2 :", self.x_j2.shape)
                self.w_kj2[junit + 1, kunit] += self.learning_rate * dk[kunit] * self.x_j2[junit]
        """ 

        # --- Aggiornamento Pesi Hidden 2 (w_j2j1) - Versione vettorializzata ---

        self.w_j2j1 += self.learning_rate * delta_wj2j1 
 

        # --- Aggiornamento Pesi Hidden 2 (w_j2j1) - Versione non vettorializzata ---
        
        """
        for j2unit in range(self.w_j2j1.shape[1]):
            self.w_j2j1[0][j2unit] += self.learning_rate * dj2[j2unit] 

            for j1unit in range(self.w_j2j1.shape[0] - 1):
                self.w_j2j1[j1unit + 1, j2unit] += self.learning_rate * dj2[j2unit] * self.x_j1[j1unit]
        """

        # --- Aggiornamento Pesi Hidden 1 (w_j1i) Versione vettorializzata ---

        self.w_j1i += self.learning_rate * delta_wj1i

        # --- Aggiornamento Pesi Hidden 1 (w_j1i) Versione non vettorializzata ---

        """
        for j1unit in range(self.w_j1i.shape[1]):
            self.w_j1i[0][j1unit] += self.learning_rate * dj1[j1unit] 
            
            for iunit in range(self.w_j1i.shape[0] - 1):
                self.w_j1i[iunit + 1, j1unit] += self.learning_rate * dj1[j1unit] * x_pattern[iunit]

        """

    def forward(self, x_pattern: Array1D) -> tuple[Array1D, Array1D, Array1D]:
        """
        Forward pass su tutta la rete per un **singolo pattern!**

        Args:
            x_pattern: Vettore di input
        
        Returns:
            x_j1  = Vettore risultato primo hidden layer
            x_j2  = Vettore risultato secondo hidden layer
            x_k   = Vettore risultato output layer
        """

        #x_biased = self.add_bias(x_pattern)

        self.x_k, self.x_j2, self.x_j1 = forward_all_layers(
            x_pattern,
            self.w_j1i,
            self.w_j2j1,
            self.w_kj2,
            self.f_act
        )

        return self.x_k, self.x_j2, self.x_j1
