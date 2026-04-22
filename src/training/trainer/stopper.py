import numpy as np
from src.utils import *


class EarlyStopper():
    """
    Mantiene le info per l'early stopping,
    attualmente adesso solo attivabile se la validation è attiva
    sullo score della validation. 
    
    Deve inoltre gestire il caso di:

        - MONK = ACCURACY (da massimizzare, si ferma quando inizia a scendere)
        - CUP = MEE (da minimizzare, si ferma quando inizia a salire)
    
    Il costruttore prende come argomenti:

        - patience: massimi comportamenti sbagliati dopo il quale ferma il training,
            aumentarlo potrebbe migliorare a uscire da un minimo locale
        - min_delta: l'epsilon, ovvero la tolleranza massima sul
    """

    def __init__(self,
                 patience,
                 min_delta: int, 
                 mode: int):


        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = np.inf if mode == 'min' else -np.inf # piccolo trick
        self.best_weights = None
        self.stop_training = False



    def __call__(self, current_score, model_weights):
        """
        call praticamente rende l'instanza della classe una funzione chiamabile passando i parametri,
        molto fico.

        In generale questo metodo implementa la logica necessaria per capire se il validation error
        sta migliorando oppure sta peggiorando in base al tipo di metrica utilizzata

        """
        # Logica per minimizzare l'mee
        if self.mode == 'min':
            "Nel caso del min, current_score sarà l'errore sul validation set"
            improved = current_score < (self.best_score - self.min_delta)

        # Logica per massimizzare l'accuracy
        elif self.mode == 'max':

            # Se ha classificato correttamente 
            improved = current_score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = current_score
            self.counter = 0

            #save_model() # SALVA IL MODELLO
            self.best_weights = model_weights

        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_training = True
        
        return self.stop_training