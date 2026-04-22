from src.activationf.sigmoid import sigmaf
from src.activationf.leaky_relu import leaky_relu

# ======== PATHS DATA ========
PATH_DT = "data/monk/train_data/monks-1.train"
PATH_ST = "data/monk/test_data/monks-1.train"
N_OUTPUTS = [1]

# =============================
        #####    #####
       ##       ##   
       ##  ###   ##### 
       ##   ##       ##
        #####    #####  params 

# ======== ARCHITECTURE ========
# Lista di unità, il numero di layer
# viene inferito dal lenght, il numero dal
# valore degli elementi.

UNITS_LIST = [
                [3],  # best
                [4],
                #[3, 3],
                #[4,4]
            ]


# ======= ACTIVATION F =======
# Le funzioni di attivazione:
# il fun act hidden è comune a tutti gli hidden units
# il fun act output è comune a tutti gli output units
FUN_ACT_HIDDEN = [
                    sigmaf,
                    #leaky_relu
                ]
FUN_ACT_OUTPUT = [
                    sigmaf,
                    #tanh
                ]

# ====== LEARNING RATE =======
# Più comunemente conosciuto come l'eta
LEARNING_RATE = [
                    0.1
                    #0.05,
                    #0.01,
                ]  

# ========= DECAY ============
# Il decay_factor moltiplica l'eta ogni decay_step
USE_DECAY = [
                False,
                #True,
            ]
DECAY_FACTOR = [
                0.99,
            ]
DECAY_STEP = [  
                10,
            ]   

# ========= MINI BATCH ============
# Il mini batch può essere compreso tra
# 1 e numero di pattern, rispettivamente:
# online e full batch
MINI_BATCH_SIZE = [
                #len(tr_input) # <- Gestito dentro il main
                ]  

# ======== N EPOCHS ==========
# Numero di epoche massime
EPOCHS = [
            200
        ]

# ======== EARLY STOPPING ====
# Epsilon indica la tolleranza sul valore del gradiente
# epsilon più piccolo indica meno tolleranza, 
# dopo "patience" epoche il train si ferma 
EARLY_STOPPING = [False]
EPSILON = [0.1]  
PATIENCE = [15]

# ========= MOMENTUM =========
# Segue il segno di iterazioni consecutive,
# stabilizzando la direzione
MOMENTUM = [False]
ALPHA_MOM = [0.9]  

# ========= VALIDATION =======
RUN_VALIDATION = [True]
RUN_HOLD_OUT_VALIDATION = True
SPLIT = 20 

# ======= REGULARIZATION =====
# Previene l'overfitting
LAMBDAL2 = [0]

# ===== GRADIENT CLIPPING ====

MAX_GRADIENT_NORM = [10.0]

VERBOSE = [True]

# ===== K-FOLD ====
FOLDS = 2