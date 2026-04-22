from src.activationf.sigmoid import sigmaf
from src.activationf.linear import linear
from src.activationf.leaky_relu import leaky_relu

# ======== PATHS DATA ========
PATH_DT = "data/cup/training_data/ML-CUP25-TR.csv"
PATH_TS = "data/cup/test_data/ML-CUP25-TS.csv"

N_OUTPUTS = 4


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
           #[128, 64, 32], 
            #[400, 200],
            #[2400, 1200],
            #[100, 200, 200, 100],
            #[150, 300, 150],
            #[300, 150, 300],
            #[500, 250, 125],
            #[800, 400],
            [700, 350],
            [600, 300],  #...
            [500, 250]
            #[300, 150],
            #[250, 100],
            #[250, 100, 50]
            #[150, 75]
            #[200, 100],
            #[256, 128, 64]
            #[40,20]
            ]


# ======= ACTIVATION F =======
# Le funzioni di attivazione:
# il fun act hidden è comune a tutti gli hidden units
# il fun act output è comune a tutti gli output units
FUN_ACT_HIDDEN = [
                    leaky_relu,
                    #leaky_relu
                ]
FUN_ACT_OUTPUT = [
                    linear,
                ]

# ====== LEARNING RATE =======
# Più comunemente conosciuto come l'eta
LEARNING_RATE = [
                    #0.01, 
                    #0.0025,
                    #0.008,
                    #0.015,
                    #0.01.  #mmm
                    #0.1,
                    #0.05,
                    0.02,   #...
                    #0.03
                    #0.001,
                    #0.005
                ]  

# ========= DECAY ============
# Il decay_factor moltiplica l'eta ogni decay_step
USE_DECAY = [
                #False,
                True,
            ]
DECAY_FACTOR = [
                #0.99,
                0.95
            ]
DECAY_STEP = [  
                #25,
                #50, #nice
                #75,
                200 
            ]   

# ========= MINI BATCH ============
# Il mini batch può essere compreso tra
# 1 e numero di pattern, rispettivamente:
# online e full batch
MINI_BATCH_SIZE = [
                #64,
                #len(tr_input) # <- Gestito dentro il main
                ]  

# ======== N EPOCHS ==========
# Numero di epoche massime
EPOCHS = [
            8000
        ]

# ======== EARLY STOPPING ====
# Epsilon indica la tolleranza sul valore del gradiente
# epsilon più piccolo indica meno tolleranza, 
# dopo "patience" epoche il train si ferma 
EARLY_STOPPING = [True]
EPSILON = [1e-11]  
PATIENCE = [100]

# ========= MOMENTUM =========
# Segue il segno di iterazioni consecutive,
# stabilizzando la direzione
MOMENTUM = [
            #False,
            True
            ]
ALPHA_MOM = [
                #0.0,
                #0.5,
                0.9, #mmm
                #0.7
    ]  

# ========= VALIDATION =======
RUN_VALIDATION = [True]
RUN_HOLD_OUT_VALIDATION = [True]
SPLIT = [20] 

# ======= REGULARIZATION =====
# Previene l'overfitting
LAMBDAL2 = [
            1e-06,
            #0,
            #0.00001, #...
            #0.001.  #mmm
]

# ===== GRADIENT CLIPPING ====

MAX_GRADIENT_NORM = [100] 

VERBOSE = [True]

# ===== K-FOLD ====
FOLDS = 3









"""
════════════════════════════════════════════════════════════
 🏆  BEST CONFIGURATION FOUND
════════════════════════════════════════════════════════════
 📊  BEST Mean MSE (Grid)     : 0.107380
────────────────────────────────────────────────────────────
 • alpha_mom                 :  0.9
 • decay_factor              :  0.95
 • decay_step                :  25
 • early_stopping            :  True
 • epochs                    :  400
 • epsilon                   :  1e-06
 • f_act_hidden              :  leaky_relu
 • f_act_output              :  linear
 • lambdal2                  :  0
 • learning_rate             :  0.02
 • max_gradient_norm         :  100
 • mini_batch_size           :  64
 • momentum                  :  True
 • n_outputs                 :  4
 • patience                  :  30
 • split                     :  20
 • units_list                :  [300, 150]
 • use_decay                 :  True
 • validation                :  True
 • verbose                   :  False
════════════════════════════════════════════════════════════
"""