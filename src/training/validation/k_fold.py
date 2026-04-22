import numpy as np
from src.training.trainer.trainer_cup import TrainerCup
from src.utils.compute_error import mean_euclidean_error, mean_squared_error

def k_fold_split(x_i, d, folds, validation_fold):

    # Divide l'array in "folds" parti uguali
    x_i_splitted = np.array_split(x_i, folds)
    d_splitted   = np.array_split(d, folds)

    vl_input   = x_i_splitted[validation_fold]
    vl_targets = d_splitted[validation_fold]

    # Il training set diventa la concatenazione di tutte le parti
    tr_input = np.concatenate(
        [sub for i, sub in enumerate(x_i_splitted) if i != validation_fold]
    )

    tr_targets = np.concatenate(
        [sub for i, sub in enumerate(d_splitted) if i != validation_fold]
    )

    return tr_input, tr_targets, vl_input, vl_targets


def run_k_fold_cup(x_full, d_full, k_folds, model_config, x_test_internal = None, verbose=True):
    """
    Esegue K-Fold e restituisce:
    - Statistiche Scalari: Mean MSE, Mean MEE (con std dev)
    - Curve Medie: Per i grafici (solo MSE, o anche MEE se vuoi)
    """
    
    all_tr_mse_histories = []
    all_vl_mse_histories = []
    all_tr_mee_histories = []
    all_vl_mee_histories = []

    # Liste per le statistiche scalari finali
    vl_mee_finals = []
    vl_mse_finals = []
    tr_mse_finals = []
    tr_mee_finals = []

    test_internal_history_output = []


    for i in range(k_folds):
        if verbose: print(f"--- Fold {i+1}/{k_folds} ---")
            
        tr_input, tr_target, vl_input, vl_target = k_fold_split(x_full, d_full, k_folds, i)
        
        # Ad ogni k viene creato un nuovo trainer da 0
        k_trainer = TrainerCup(input_size=tr_input.shape[1], **model_config)
        
        # Fit restituisce i valori finali per ogni k-fold, ma noi accediamo alle liste interne per precisione
        k_trainer.fit(
            tr_x=tr_input, tr_d=tr_target, 
            vl_x=vl_input, vl_d=vl_target, 
            fold_id=i if verbose else None
        )

        all_tr_mse_histories.append(k_trainer.tr_mse_history)
        all_vl_mse_histories.append(k_trainer.vl_mse_history)
        all_tr_mee_histories.append(k_trainer.tr_mee_history)
        all_vl_mee_histories.append(k_trainer.vl_mee_history)

        vl_mse_finals.append(k_trainer.vl_mse_history[-1])
        vl_mee_finals.append(k_trainer.vl_mee_history[-1])
        tr_mse_finals.append(k_trainer.tr_mse_history[-1])
        tr_mee_finals.append(k_trainer.tr_mee_history[-1])
        #epochs_reached.append(len(k_trainer.tr_mse_history))

        if x_test_internal is not None:
            res, _ = k_trainer.neuraln.forward_network(x_test_internal, k_trainer.f_act_hidden, k_trainer.f_act_output)
            test_internal_history_output.append(res[-1])
    
    
# Restituisce direttamente le liste di liste
    return {
        "vl_mean_mse": np.mean(vl_mse_finals),
        "vl_std_mse": np.std(vl_mse_finals),
        "tr_mean_mse": np.mean(tr_mse_finals),

        "vl_mean_mee": np.mean(vl_mee_finals),
        "vl_std_mee": np.std(vl_mee_finals),
        "tr_mean_mee": np.mean(tr_mee_finals),

        
        "all_tr_history_mse": all_tr_mse_histories,
        "all_vl_history_mse": all_vl_mse_histories,
        
        "all_tr_history_mee": all_tr_mee_histories,
        "all_vl_history_mee": all_vl_mee_histories,

        "epoch_reached": k_trainer.trigger_epoch,

        "test_internal_history_output": test_internal_history_output
    }