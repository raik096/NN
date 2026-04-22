def traindata_split_in_ts_vs(traindata, split):
    #---hold out una percentuale uguale a config.SPLIT/100 es. 20/100

    n_total = int(traindata.shape[0])
    n_keep = int(round(n_total-n_total*split/100.))
    validation_i = traindata[n_keep:,:]                                       #<- i rimanenti valori vengono caricati in una array per la validation
    x_i_remaining = traindata[:n_keep,:]

    return x_i_remaining, validation_i