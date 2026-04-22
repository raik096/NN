import numpy as np


def hold_out_validation_stratified(x_i, d, split: int, random_state=None):
    rng = np.random.RandomState(random_state)

    test_frac = split / 100.0
    x_train_list = []
    y_train_list = []
    x_val_list = []
    y_val_list = []

    for label in np.unique(d):

        class_idx = np.where(d == label)[0]
        rng.shuffle(class_idx)

        n = len(class_idx)
        n_val = int(round(n * test_frac))

        val_idx = class_idx[:n_val]
        train_idx = class_idx[n_val:]

        x_val_list.append(x_i[val_idx])
        y_val_list.append(d[val_idx])
        x_train_list.append(x_i[train_idx])
        y_train_list.append(d[train_idx])

    x_train = np.vstack(x_train_list)
    y_train = np.concatenate(y_train_list)
    x_val = np.vstack(x_val_list)
    y_val = np.concatenate(y_val_list)

    train_perm = rng.permutation(len(y_train))
    val_perm = rng.permutation(len(y_val))

    return (x_train[train_perm], y_train[train_perm],
            x_val[val_perm], y_val[val_perm])