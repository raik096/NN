import json
import numpy as np
import os

def load_model(weights_filename, architecture_filename):
    """
    Carica i pesi e la configurazione del modello salvati da save_model.
    Restituisce:
        - loaded_weights: dict {layer_name: np.ndarray}
        - loaded_architecture: dict (architettura + trainer_params)
    """

    loaded_weights = {}
    with open(weights_filename, "r") as f:
        lines = f.readlines()

    current_layer = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.endswith(":"):
            current_layer = line[:-1]
            loaded_weights[current_layer] = []
        else:
            # parse numeri separati da virgola
            loaded_weights[current_layer].append(
                list(map(float, line.split(",")))
            )

    for key in loaded_weights:
        loaded_weights[key] = np.array(loaded_weights[key])

    with open(architecture_filename, "r") as f:
        loaded_json = json.load(f)

    loaded_architecture = loaded_json.get("architecture", {})
    trainer_params = loaded_json.get("trainer_params", {})

    return loaded_weights, loaded_architecture, trainer_params