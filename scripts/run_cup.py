import numpy as np
import pandas as pd
from config import gs_cup_config
from src.utils import * 
from src.activationf import *
from src.training.grid_search import GridSearch
from src.training.trainer.trainer_cup import TrainerCup
from src.training.validation.hold_out import hold_out_validation


# =============================================================================
# CARICAMENTO
# =============================================================================
print("Caricamento dati CUP...")


x_i, d = load_data(gs_cup_config.PATH_DT)
x_i = x_i.to_numpy().astype(np.float64)
d = d.to_numpy().astype(np.float64)

# Normalizza
x_i, x_min, x_max = normalize_data(x_i)
d, d_min, d_max = normalize_data(d)

x_i_final = x_i
d_final = d

x_i, d, x_i_test, d_test = hold_out_validation(x_i, d, 15) 

target_range = d_max - d_min
avg_target_range = np.mean(target_range)

# =============================================================================
# GRID SEARCH 
# =============================================================================
gs = GridSearch(
    units_list      = gs_cup_config.UNITS_LIST,          
    n_outputs       = [gs_cup_config.N_OUTPUTS],          
    f_act_hidden    = gs_cup_config.FUN_ACT_HIDDEN,    
    f_act_output    = gs_cup_config.FUN_ACT_OUTPUT,     
    mini_batch_size = [
                       x_i.shape[0] - (x_i.shape[0]/3), 
                        (x_i.shape[0] - (x_i.shape[0]/3))/2,
                       ], 
    learning_rate   = gs_cup_config.LEARNING_RATE,     
    use_decay       = gs_cup_config.USE_DECAY,            
    decay_factor    = gs_cup_config.DECAY_FACTOR,
    decay_step      = gs_cup_config.DECAY_STEP,
    momentum        = gs_cup_config.MOMENTUM,
    alpha_mom       = gs_cup_config.ALPHA_MOM,
    lambdal2        = gs_cup_config.LAMBDAL2,
    epochs          = gs_cup_config.EPOCHS,
    early_stopping  = gs_cup_config.EARLY_STOPPING,
    epsilon         = gs_cup_config.EPSILON,
    patience        = gs_cup_config.PATIENCE,
    max_gradient_norm = gs_cup_config.MAX_GRADIENT_NORM,
    
    split=[gs_cup_config.SPLIT], 
    verbose=gs_cup_config.VERBOSE, 
    validation=[True]
)

print("\nAvvio Grid Search con K-Fold interno...")

best_config, best_score_gs, best_epoch, tr_history_error, vl_history_error = gs.run_for_cup_with_kfold(
        x_i, d,
        k_folds=gs_cup_config.FOLDS,
        d_max=d_max,
        d_min=d_min,  # <- Servono per la denormalizzazione
        )  # <- METRICHE TR ( = TR) E VL

print_config(best_config, best_score_gs, "Mean MSE")

# =============================================================================
# FINAL ASSESSMENT 
# =============================================================================
print(f"AVVIO FINAL ASSESSMENT")

# Cambio configurazione per Train intenso
final_config = best_config.copy()
final_config['epochs'] = best_epoch  # Deve essere la migliore
final_config['early_stopping'] = False
final_config['validation'] = True
final_config['verbose'] = True   
final_config['d_max'] = d_max
final_config['d_min'] = d_min

# Istanza del Trainer, che instanzia una rete neurale in base ai parametri passati
trainer_final = TrainerCup(
    input_size = x_i.shape[1],
    **final_config
)

# Addestramento della NN
tr_mee_history_error, tr_mse_history_error, _, _ = trainer_final.fit(x_i, d, vl_x=x_i_test, vl_d=d_test) # <- METRICHE DEL TR ( = TR + VL)
# Run della NN dopo il fit
final_out = trainer_final.neuraln.run_nn(x_i_test)
# Denormalizzaione degli output
mee_final_test_denorm = denorm_mean_euclidean_error(          # <- METRICHE DEL TEST SET INTERNO
    final_out,    # Output rete
    d_test,       # Target rete
    d_max, d_min # Range per denormalizzare Output
    )

save_model(trainer_final)
print("\n\n|||| TS MEE FINAL: ", mee_final_test_denorm, "|||\n\n")

# =============================================================================
# FINAL TRAIN MODEL
# =============================================================================

final_countdown = False

if final_countdown: 

    x_blind_test, _ = load_data(gs_cup_config.PATH_TS)
    x_blind_test = x_blind_test.to_numpy().astype(np.float64)

    x_blind_test = (x_blind_test - x_min) / (x_max - x_min)

    ultra_final_config = best_config.copy()
    ultra_final_config['epochs'] = best_epoch
    ultra_final_config['patience'] = best_epoch
    ultra_final_config['validation'] = False

    ultra_final_config['d_max'] = d_max
    ultra_final_config['d_min'] = d_min

    ultra_trainer_final = TrainerCup(
        input_size=x_i_final.shape[1],
        **ultra_final_config
    )

    ultra_trainer_final.fit(x_i_final, d_final)

    ultra_final_output = ultra_trainer_final.neuraln.run_nn(x_blind_test)
    ultra_final_output = ultra_final_output * (d_max - d_min) + d_min


    filename = "from_stone_to_dust_ML-CUP25-TS.csv"
    data = datetime.now().strftime("%d %m %Y")

    header = [
        "# Elmi Leonardo, Lazzari Andres\n",
        "# From_Stone_To_Dust\n",
        "# ML-CUP25 v1\n",
        f"# {data}\n"
    ]
    
    ids = np.arange(1, 1001)
    df = pd.DataFrame(ultra_final_output)
    df.insert(0, 'id', ids)

    with open(filename, 'w') as f:
        f.write("\n") 
        f.writelines(header)
        df.to_csv(f, header=False, index=False)

    print(f"File generato from_stone_to_dust: {filename}")