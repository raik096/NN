from src.activationf.sigmoid import sigmaf
from src.activationf.leaky_relu import leaky_relu

# ======== PATHS DATA ========
PATH_DT = "data/monk/train_data/monks-1.train"
MONK = True

# ======== ARCHITECTURE ========
UNITS_LIST = [4]  
N_OUTPUTS = 1

# ======= ACTIVATION F =======
FUN_ACT_HIDDEN = sigmaf
FUN_ACT_OUTPUT = sigmaf

# ====== LEARNING RATE =======
LEARNING_RATE = 0.1  
# Prova: 0.1, 0.05, 0.01 se non converge

# ========= DECAY ============
USE_DECAY = False  
DECAY_FACTOR = 0.99
DECAY_STEP = 10

# ========= BATCH ============
BATCH = True  

# ======== N EPOCHS ==========
EPOCHS = 10

# ======== EARLY STOPPING ====
EARLY_STOPPING = False
EPSILON = 0.1  # Più tollerante per MSE
PATIENCE = 15

# ========= MOMENTUM =========
MOMENTUM = False
ALPHA_MOM = 0.9  # ← 0.9 è standard

# ========= VALIDATION =======
RUN_VALIDATION = True
RUN_HOLD_OUT_VALIDATION = True
SPLIT = 20 

# ======= REGULARIZATION =====
LAMBDA = 0.0  # ← Disattiva per ora, MONK è semplice

# ===== GRADIENT CLIPPING ====

MAX_GRADIENT_NORM = 10.0  

VERBOSE = True

# ===== K-FOLD ====
FOLDS = 2