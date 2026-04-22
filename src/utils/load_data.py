import pandas as pd

# carica i dati necessari al primo input layer e output layer, prende come argomento il path del file TR

def load_data(path) -> tuple[pd.DataFrame, pd.DataFrame] :

    try:
        df_all = pd.read_csv(path, skiprows=7, header=None)
        df_input_layer = df_all.iloc[:, 1:13]
        if df_all.shape[1] > 13:
            df_output_layer = df_all.iloc[:, 13:17]
        else:
            df_output_layer = None
        #print(df_input_layer, df_output_layer)

        return df_input_layer, df_output_layer
    except Exception as e:
            return None, None


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