import pandas as pd

# carica i dati necessari al primo input layer e output layer, prende come argomento il path del file TR

def load_data(path) -> tuple[pd.DataFrame, pd.DataFrame] :

    df_input_layer = pd.read_csv(path, header=None, skiprows=7, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    df_output_layer = pd.read_csv(path, header=None, skiprows=7, usecols=[13, 14, 15, 16])

    #print(df_input_layer, df_output_layer)

    return df_input_layer, df_output_layer


import pandas as pd


def load_monks_data(path) -> tuple[pd.DataFrame, pd.Series]:
    # Carica il file: il formato MONK ha spazi bianchi come separatori
    # La colonna 0 è il target, le colonne 1-6 sono le feature, la colonna 7 è l'ID (da ignorare)
    data = pd.read_csv(path, sep=r'\s+', header=None)

    # Target (0 o 1)
    y = data.iloc[:, 0].astype(float)
    y.name = 'target'

    # Feature grezze (colonne dalla 1 alla 6)
    X_raw = data.iloc[:, 1:7]
    X_raw.columns = [f'feature_{i}' for i in range(1, 7)]

    # ONE-HOT ENCODING
    # È fondamentale perché le feature sono categoriche (es. 1, 2, 3)
    # get_dummies trasforma la feature "1" nel vettore [1, 0, 0]
    X = pd.get_dummies(X_raw, columns=X_raw.columns)

    return X.astype(float), y