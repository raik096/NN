from src.training.trainer.trainer import Trainer
from src.training.trainer.stopper import EarlyStopper
import time
#from src.utils.visualization import plot_monk
from src.activationf import *
from src.utils import *




class TrainerMonk(Trainer):
    """
    Sottoclasse di Trainer.
    Mantiene traccia dei parametri necessari al:
        - Train del monk
        - Test set ad ogni epoca
        - Confusion Matrix
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.tr_accuracy_history = []
        self.vl_accuracy_history = []
        self.ts_accuracy_history = []

        self.config = kwargs


    def fit(self, tr_x, tr_d, vl_x = None, vl_d = None, ts_x = None, ts_d = None):
        """
        Il fit del monk 
        """

        n_patterns = tr_x.shape[0]
        start_time = time.perf_counter()
        
        stopper = EarlyStopper(
            patience = self.patience,
            min_delta = self.epsilon,
            mode = "max"
        )

        best_vl_accuracy = 0.0

        # ! AVVIO CICLO DI EPOCHE ! #
        for epoch in range(1, self.epochs + 1):
            
            self.epoch = epoch
            matched_tr = 0
            matched_ts = 0

            tr_epoch_results = self._run_epoch(tr_x, tr_d, n_patterns)
            self.tr_mse_history.append(tr_epoch_results["mse_tr"])

            # Analizza l'accuracy sul training set
            tr_acc = self._compute_accuracy_internal(tr_x, tr_d)
            self.tr_accuracy_history.append(tr_acc)

            # Analizza l'accuracy sul validation set e capisce se si deve fermare
            if self.validation and vl_x is not None and vl_d is not None: 
                vl_acc = self._compute_accuracy_internal(vl_x, vl_d)
                self.vl_accuracy_history.append(vl_acc)

                if vl_acc > best_vl_accuracy:
                    best_vl_accuracy = vl_acc

                if self.early_stopping:
                    if stopper(vl_acc, self.neuraln.weights_matrix_list):

                        self.neuraln.weights_matrix_list = stopper.best_weights
                        break

            # Analizza l'accuracy sul test set
            if ts_x is not None and ts_d is not None:
                ts_acc = self._compute_accuracy_internal(ts_x, ts_d)
                self.ts_accuracy_history.append(ts_acc)

            if self.verbose and epoch % 10 == 0:
                print(f"||[MONK] Epoch {epoch} | TR MSE: {tr_epoch_results['mse_tr']:.4f} | TR ACC: {tr_acc:.4f} ||")


        if self.verbose: 
            training_time = time.perf_counter() - start_time
            print(" Final mse error: ", tr_epoch_results["mse_tr"])
            print(" Accuracy finale su training set: ", self.tr_accuracy_history[-1]*100,"%")

            if ts_x is not None and ts_d is not None:
                print(" Accuracy finale su test set: ", self.ts_accuracy_history[-1]*100,"%")
            print(" Tempo di fitting: ", training_time)
            print("\n--- Fitting Completato ---\n")

            plot_monk(
                            tr_mse_history=self.tr_mse_history, 
                            vl_mse_history=[],
                            tr_accuracy_history=self.tr_accuracy_history, 
                            vl_accuracy_history=self.vl_accuracy_history,
                            ts_accuracy_history=self.ts_accuracy_history, 
                            training_time=training_time,
                            config=self.config
            )

        return (tr_epoch_results["mse_tr"], best_vl_accuracy)
                #self.vl_mee_history[-1] if self.validation else 0.0,
                #self.vl_mse_history[-1] if self.validation else 0.0,
                #final_vl_accuracy if self.validation else 0.0)


    def _compute_accuracy_internal(self, x, d):
        "Helper per simplificare calcolo ridondante"
        matched = 0
        for i, pattern in enumerate(x):
            layer_results, _ = self.neuraln.forward_network(pattern, self.f_act_hidden, self.f_act_output)

            if self.f_act_output == tanh:
                prediction = 1.0 if layer_results[-1] >= 0.0 else 0.0
            else: 
                prediction = 1.0 if layer_results[-1] >= 0.5 else 0.0

            if prediction == d[i]:
                matched += 1
        
        return matched / x.shape[0]
    

