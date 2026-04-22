from src.training.trainer.trainer_monk import TrainerMonk
from src.activationf.sigmoid import sigmaf
from src.utils import load_monks_data
import numpy as np


# 1. CARICAMENTO DATI TRAIN
x_train, d_train = load_monks_data("data/monk/train_data/monks-2.train")
x_train = x_train.to_numpy().astype(np.float64)
d_train = d_train.to_numpy().astype(np.float64)

x_test, d_test = load_monks_data("data/monk/test_data/monks-2.test")
x_test = x_test.to_numpy().astype(np.float64)
d_test = d_test.to_numpy().astype(np.float64)


n_patterns = len(x_train)

# 2. CONFIGURAZIONE OTTIMALE PER MONK-1
trainer = TrainerMonk(
    input_size = x_train.shape[1],  
    units_list = [4],          
    n_outputs = 1,
    f_act_hidden = sigmaf,
    f_act_output=sigmaf,    
    learning_rate=0.3,       # MONK disattivo
    use_decay=False,
    decay_factor=0.0,
    decay_step=0,
    mini_batch_size = 25,     # 1: ONLINE, N_PATTERNS: FULL BATCH
    epochs= 300,
    momentum=False,
    alpha_mom=0.3,
    max_gradient_norm=5,
    verbose=True,
    lambdal2=0              
)

print("Inizio Training...")
trainer.fit(x_train, d_train, x_test, d_test)
print("Fine Training.")
