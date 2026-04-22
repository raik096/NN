from src.activationf.sigmoid import sigmaf
from src.activationf.linear import linear
from src.activationf.leaky_relu import leaky_relu

# ======== PATHS DATA ========
PATH_DT = "data/cup/training_data/ML-CUP25-TR.csv"

# ======== ARCHITECTURE ========
UNITS_LIST = [32, 64, 16]  
N_OUTPUTS = 4

# ======= ACTIVATION F =======
FUN_ACT_HIDDEN = leaky_relu
FUN_ACT_OUTPUT = linear

# ====== LEARNING RATE =======
LEARNING_RATE = 0.0005   #<- Funziona in batch ma da applicare il criterio di stop sennò scavalca il minimo
# Prova: 0.1, 0.05, 0.01 se non converge

# ========= DECAY ============
USE_DECAY = True
DECAY_FACTOR = 0.99
DECAY_STEP = 1000

# ========= BATCH ============
BATCH = True  

# ======== N EPOCHS ==========
EPOCHS = 15000

# ======== EARLY STOPPING ====
EARLY_STOPPING = True 
EPSILON = 0.00005
PATIENCE = 15

# ========= MOMENTUM =========
MOMENTUM = True  
ALPHA_MOM = 0.5  # ← 0.9 è standard

# ========= VALIDATION =======
RUN_VALIDATION = True
RUN_HOLD_OUT_VALIDATION = True
SPLIT = 20  

# ======= REGULARIZATION =====
LAMBDA = 0.0  # ← Disattiva per ora

# ===== GRADIENT CLIPPING ====

MAX_GRADIENT_NORM = 5  

VERBOSE = True

FOLDS = 5