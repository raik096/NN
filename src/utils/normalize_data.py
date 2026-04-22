def normalize_data(data):
    # Calcola min e max per ogni colonna
    min_val = data.min(axis=0)
    max_val = data.max(axis=0)
    
    denom = max_val - min_val

    # Se una colonna ha tutti i valori uguali, denom sarà 0.
    # Sostituiamo gli 0 con 1 per evitare la divisione per zero (il numeratore sarà comunque 0)
    denom[denom == 0] = 1.0
    data_norm = (data - min_val) / denom
    return data_norm, min_val, max_val

