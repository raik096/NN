
def hold_out_validation(x_i, d, split: int):
    n_total = x_i.shape[0]
    n_keep = int(round(n_total - n_total * split / 100.0))

    x_i_remaining = x_i[:n_keep]
    d_remaining = d[:n_keep]

    validation_set = x_i[n_keep:]
    validation_d = d[n_keep:]

    return x_i_remaining, d_remaining, validation_set, validation_d