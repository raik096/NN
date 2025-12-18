from src.training import trainer
from src.training.trainer import Trainer

from src.utils.load_data import load_data

import config

# Carica dati
x_i, d = load_data(config.PATH_DT)
x_i = x_i.to_numpy()
d = d.to_numpy()

# Inizializza la classe Trainer:
#  - crea la rete neurale
#  - traccia gli errori
#  - implementa metodo che esegue il training

trainer = Trainer(x_i.shape[1],
                  config.N_HIDDENL1,
                  config.N_HIDDENL2,
                  config.N_OUTPUTS,
                  config.FUN_ACT,
                  config.LEARNING_RATE,
                  config.BATCH,
                  config.EPOCHS,
                  config.EPSILON,
                  config.PATIENCE)

# Avvia training
trainer.train(x_i, d)
