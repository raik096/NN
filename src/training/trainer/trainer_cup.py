from src.training.trainer.trainer import Trainer
from src.training.trainer.stopper import EarlyStopper
from typing import Callable
from src.utils import *

import numpy as np
import time



class TrainerCup(Trainer):
    """
    Sottoclasse di Trainer.
    Mantiene traccia dei parametri necessari al:
        - Train della cup (MSE)
        - Validation (K-FOLD)
        - Early Stopping su validation error
    """

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)

        self.vl_mee_history = []
        self.vl_mse_history = []
        self.trigger_epoch = 0 

    def fit(self, 
            tr_x: np.ndarray, 
            tr_d: np.ndarray,
            vl_x: np.ndarray = None,
            vl_d: np.ndarray = None,
            fold_id: int = None,
            metric_fn: Callable = None):

        n_patterns = tr_x.shape[0]
        start_time = time.perf_counter()

        # Gestisce la logica dell'early stopping
        # e si assicura di salvare la matrice dei pesi
        # appena capisce che si deve fermare, l'early stopping
        # per adesso è attiva solo nel caso sia attiva la validation
        should_stop = EarlyStopper(
            patience = self.patience,
            min_delta = self.epsilon,
            mode="min"
        )

        tr_epoch_results = None
        
        # ! AVVIO CICLO DI EPOCHE ! #
        for epoch in range(1, self.epochs + 1):
            
            self.epoch = epoch 
            tr_epoch_results = self._run_epoch(tr_x, tr_d, n_patterns)

            self.tr_mee_history.append(tr_epoch_results["mee_tr"])
            self.tr_mse_history.append(tr_epoch_results["mse_tr"])


            if self.validation and vl_x is not None:
                
                # Al suo interno viene presa la media del risultato
                # su tutti i pattern
                vl_epoch_results = self._run_epoch_vl(vl_x, vl_d, metric_fn)

                self.vl_mee_history.append(vl_epoch_results["mee_vl"])
                self.vl_mse_history.append(vl_epoch_results["mse_vl"])

                vl_score = vl_epoch_results["vl_score"]
                
                # Se should stop è vero allora l'Early Stopper ha superato la patience
                if should_stop(vl_score, self.neuraln.weights_matrix_list):
                    self.trigger_epoch = epoch
                    break
                    

            if self.verbose and epoch % 15 == 0:
                print("||[CUP] epoch n° ", epoch, ", total mse error with training patterns: ", tr_epoch_results["mse_tr"], " ||")

        #if fold_id is not None:
            #save_model(self, fold_id)
        if self.trigger_epoch == 0:
            self.trigger_epoch = self.epochs

        if self.verbose: 
            print(" Final mee error: ", tr_epoch_results["mee_tr"])
            print(" Final mse error: ", tr_epoch_results["mse_tr"])
            print(" Tempo di fitting: ", time.perf_counter() - start_time)
            print("\n--- Fitting Completato ---\n")

            if self.validation: 
                plot_errors_with_validation_error(self, time.perf_counter() - start_time)
            else:
                plot_errors(self, time.perf_counter() - start_time)

        return (self.tr_mee_history[-1], 
                self.tr_mse_history[-1], 
                self.vl_mee_history[-1] if vl_x is not None else 0.0,
                self.vl_mse_history[-1] if vl_x is not None else 0.0)

        
        
    def _run_epoch_vl(self, vl_x, vl_d, metric_fn: Callable = None):
        n_patterns = vl_x.shape[0]
        # epoch_mee_vl, epoch_mse_vl = [], []
        vl_final_output_array = []

        for pattern in range(n_patterns):
            vl_layer_results, _ = self.neuraln.forward_network(vl_x[pattern], self.f_act_hidden, self.f_act_output)
            vl_final_output_array.append(vl_layer_results[-1].tolist())

            # Calcola accuracy per monitoring
        epoch_mee_vl = mean_euclidean_error(vl_final_output_array, vl_d)
        epoch_mse_vl = mean_squared_error(vl_final_output_array, vl_d)
        vl_score = epoch_mse_vl


        return {
            'vl_score': vl_score,
            'mee_vl': epoch_mee_vl,
            'mse_vl': epoch_mse_vl
            #'accuracy_vl': 0.0
        }
