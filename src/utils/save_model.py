import json
import os
from datetime import datetime
import numpy as np


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.training.trainer.trainer import Trainer

def make_json_serializable(obj):
    """
    Converte ricorsivamente un oggetto in una versione serializzabile da JSON.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    if callable(obj):
        return obj.__name__  # Restituisce 'sigmoid', 'relu', ecc.
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


def save_model(trainer: 'Trainer', fold_id=None):
    """
    Salva il modello completo estraendo i dati direttamente dal Trainer.
    
    Args:
        trainer: L'istanza del Trainer contenente la rete e i parametri.
        fold_id: (Opzionale) Indice del fold o identificativo del run.
    """
    
    weights_list = trainer.neuraln.weights_matrix_list
    layer_units = trainer.neuraln.units_list
    

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    fold_suffix = f"_Fold_{fold_id}" if fold_id is not None else ""
    folder_name = f"{timestamp}{fold_suffix}"
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../results/models"))
    save_dir = os.path.join(base_dir, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    weights_filename = "weights.txt"
    weights_path = os.path.join(save_dir, weights_filename)
    
    with open(weights_path, "w") as f:
        for i, w in enumerate(weights_list):
            f.write(f"--- Layer {i} to {i+1} Weights ---\n")
            # Salva la matrice riga per riga
            for row in w:
                f.write(",".join(map(str, row.tolist())) + "\n")
            f.write("\n")

    architecture = []
    
    architecture.append({
        "layer_idx": 0,
        "type": "input",
        "units": layer_units[0]
    })

    for i in range(1, len(layer_units)):
        is_output = (i == len(layer_units) - 1)
        act_fn = trainer.f_act_output if is_output else trainer.f_act_hidden
        
        architecture.append({
            "layer_idx": i,
            "type": "output" if is_output else "hidden",
            "units": layer_units[i],
            "activation": act_fn.__name__
        })

    trainer_params = {}
    exclude_keys = ['neuraln', 'tr_mee_history', 'tr_mse_history', 'vl_mee_history', 'vl_mse_history', 'old_deltas']
    
    for key, val in vars(trainer).items():
        if key not in exclude_keys:
            trainer_params[key] = make_json_serializable(val)

    results_summary = {
        "final_tr_mee": trainer.tr_mee_history[-1] if trainer.tr_mee_history else None,
        "final_vl_mee": trainer.vl_mee_history[-1] if trainer.vl_mee_history else None,
    }

    full_config = {
        "architecture": architecture,
        "hyperparameters": trainer_params,
        "results": results_summary
    }

    # Salvataggio JSON
    config_filename = "model_config.json"
    config_path = os.path.join(save_dir, config_filename)
    
    with open(config_path, "w") as f:
        json.dump(full_config, f, indent=4)

    print(f"Modello salvato in:\n   {save_dir}")
    return save_dir