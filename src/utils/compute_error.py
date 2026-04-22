import numpy as np

def _normalize_shapes(outputs, targets):
    
    #print("\n\noutputs: ", outputs, "\n\ntargets: ", targets)

    outputs = np.asarray(outputs)
    #targets = np.asarray(targets)
    #outputs = np.array(outputs)
    targets = np.array(targets)

    


    #print("\n\noutputs: ", outputs, "\n\ntargets: ", targets)

    if outputs.ndim == 1:
        outputs = outputs.reshape(-1, 1)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)

    if outputs.shape != targets.shape:
        raise ValueError(f"Shape mismatch: {outputs.shape} vs {targets.shape}")

    return outputs, targets


def mean_euclidean_error(outputs, targets):
    outputs, targets = _normalize_shapes(outputs, targets)
    errors = np.linalg.norm(outputs - targets, axis=1)
    return np.mean(errors)

def denorm_mean_euclidean_error(outputs, targets, dmax, dmin):
    outputs, targets = _normalize_shapes(outputs, targets)
    outputs = outputs * (dmax - dmin) + dmin
    targets = targets * (dmax - dmin) + dmin
    errors = np.linalg.norm(outputs - targets, axis=1)
    return np.mean(errors)

def denorm_mean_squared_error(outputs, targets, dmax, dmin):
    outputs, targets = _normalize_shapes(outputs, targets)
    outputs = outputs * (dmax - dmin) + dmin
    targets = targets * (dmax - dmin) + dmin
    squared_errors = np.sum((outputs - targets) ** 2, axis=1)
    return np.mean(squared_errors)


def mean_squared_error(outputs, targets):
    outputs, targets = _normalize_shapes(outputs, targets)
    squared_errors = np.sum((outputs - targets) ** 2, axis=1)
    return np.mean(squared_errors)

def mean_squared_error(outputs, targets):
    outputs, targets = _normalize_shapes(outputs, targets)
    squared_errors = np.sum((outputs - targets) ** 2, axis=1)
    return np.mean(squared_errors)