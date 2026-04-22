import numpy as np
from config import gs_monk_config
from src.training.trainer.trainer_monk import TrainerMonk
from src.training.grid_search import GridSearch
from src.activationf import *
from src.utils import *
from src.training.validation.stratified_split import hold_out_validation_stratified

# =============================================================================
# CARICAMENTO DATI
# =============================================================================
x_i, d = load_monks_data("data/monk/train_data/monks-3.train")
x_i = x_i.to_numpy().astype(np.float64)
d = d.to_numpy().astype(np.float64)

tr_input, tr_target, vl_input, vl_target = hold_out_validation_stratified(x_i, d, gs_monk_config.SPLIT)

grid_split_param = gs_monk_config.SPLIT if isinstance(gs_monk_config.SPLIT, list) else [gs_monk_config.SPLIT]
# =============================================================================
# GRID SEARCH
# =============================================================================
gs = GridSearch(
    units_list=gs_monk_config.UNITS_LIST,
    n_outputs=gs_monk_config.N_OUTPUTS,
    f_act_hidden=gs_monk_config.FUN_ACT_HIDDEN,
    f_act_output=gs_monk_config.FUN_ACT_OUTPUT,

    learning_rate=gs_monk_config.LEARNING_RATE, 

    use_decay=gs_monk_config.USE_DECAY,
    decay_factor=gs_monk_config.DECAY_FACTOR,
    decay_step=gs_monk_config.DECAY_STEP,

    mini_batch_size=[len(tr_input)], 

    epochs=gs_monk_config.EPOCHS,

    early_stopping=gs_monk_config.EARLY_STOPPING,
    patience=gs_monk_config.PATIENCE,
    epsilon=gs_monk_config.EPSILON,

    momentum=gs_monk_config.MOMENTUM,
    alpha_mom=gs_monk_config.ALPHA_MOM,
    
    max_gradient_norm=gs_monk_config.MAX_GRADIENT_NORM,
    split=grid_split_param,
    
    verbose=[False],
    validation=[True],
    
    lambdal2=gs_monk_config.LAMBDAL2
)

print("\nAvvio Grid Search...")
best_config, best_acc_gs = gs.run_for_monk_holdout(tr_input, tr_target, vl_input, vl_target)

print(f"MIGLIOR CONFIGURAZIONE TROVATA (Val Acc: {best_acc_gs:.2%})")

# =============================================================================
# FINAL RETRAINING
# =============================================================================

print("\nCaricamento Test Set...")

x_test, d_test = load_monks_data("data/monk/test_data/monks-3.test")
x_test = x_test.to_numpy().astype(np.float64)
d_test = d_test.to_numpy().astype(np.float64)
has_test_set = True
print(f"Test Set caricato: {x_test.shape[0]} patterns")


def evaluate_configuration_with_restarts(label, config, x_tr, y_tr, x_ts, y_ts, n_trials=10):
    print(f"ASSESSMENT: {label} (su {n_trials} inizializzazioni)")
    print(
        f"  Config: L2={config.get('lambdal2', 0)}, Eta={config.get('learning_rate')}, Batch={config.get('mini_batch_size')}")

    mses, accs_tr, accs_ts = [], [], []

    for i in range(n_trials):

        # Reinizializza
        trainer = TrainerMonk(input_size=x_tr.shape[1], **config)
        trainer.verbose = True

        # Train
        final_mse, _ = trainer.fit(x_tr, y_tr, ts_x=x_ts, ts_d=y_ts)
        print(final_mse)

        # Calcoliamo l'accuracy di TRAIN chiamando direttamente la funzione interna
        final_acc_tr = trainer._compute_accuracy_internal(x_tr, y_tr)

        mses.append(final_mse)
        accs_tr.append(final_acc_tr)

        if x_ts is not None:
            acc_ts = trainer._compute_accuracy_internal(x_ts, y_ts)
            accs_ts.append(acc_ts)

    print(f"    Completato {n_trials} run.                 ")

    # Calcolo statistiche
    mean_mse, std_mse = np.mean(mses), np.std(mses)
    mean_tr, std_tr = np.mean(accs_tr), np.std(accs_tr)

    print(f"   MSE (Train):       {mean_mse} ± {std_mse}")
    print(f"   Accuracy (Train):  {mean_tr} ± {std_tr}")

    mean_ts, std_ts = 0.0, 0.0
    if x_ts is not None:
        mean_ts, std_ts = np.mean(accs_ts), np.std(accs_ts)
        print(f"   Accuracy (TEST):   {mean_ts} ± {std_ts}")

    return mean_mse, mean_tr, mean_ts


N_TRIALS = 20  # Numero di inizializzazioni diverse su cui fare la media

# BEST GRID SEARCH
evaluate_configuration_with_restarts("Best Configuration Found", best_config, x_i, d, x_test, d_test, n_trials=N_TRIALS)

run2_config = best_config.copy()
original_l2 = run2_config.get('lambdal2', 0.0)

if original_l2 == 0.0:
    run2_config['lambdal2'] = 0.01
    run2_config['epsilon'] = 1e-7
    label_comp = "Confronto: With L2 (0.001) and epsilon 1e-7"
else:
    run2_config['lambdal2'] = 0.0
    label_comp = "Confronto: Without L2 (0.0)"

evaluate_configuration_with_restarts(label_comp, run2_config, x_i, d, x_test, d_test, n_trials=N_TRIALS)

print("\nAssessment completato.")