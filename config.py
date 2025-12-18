from src.activationf.relu import relu
from src.activationf.sigmoid import sigmaf

# ============================
#          NN config
# ============================


# ======== PATHS DATA ========
PATH_DT = "data/training_data/ML-CUP25-TR.csv"


# ======== UNITS SIZE ========
N_HIDDENL1 = 64
N_HIDDENL2 = 32
N_OUTPUTS = 4


# ======= ACTIVATION F =======
FUN_ACT = relu


# ====== LEARNING RATE =======
#LEARNING_RATE = 0.0002   #<- Funziona in batch ma da applicare il criterio di stop sennò scavalca il minimo
LEARNING_RATE = 0.000025

# ========= BATCH ============
BATCH = False

# ======== N EPOCHS ==========
EPOCHS = 10000


# gradient_norm < EPSILON, quindi il gradiente non cresce abbastanza,
# serve per lo stopping criteria come limite inferiore, in percentuale
# ======== EPSILON ==========
EPSILON = 0.001

# Dopo quante epoche in cui non cresce il gradiente mi fermo
# ======= PATIENCE ==========
PATIENCE = 3