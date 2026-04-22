import numpy as np
from config import monk_config
from src.training.trainer.trainer_monk import TrainerMonk
from src.training.grid_search import GridSearch
from src.activationf import *
from src.utils import *
from src.training.validation.stratified_split import hold_out_validation_stratified

# =============================================================================
# 1. DATA LOADING
# =============================================================================
x_i, d = load_monks_data("../data/monk/train_data/monks-3.train")
x_i = x_i.to_numpy().astype(np.float64)
d = d.to_numpy().astype(np.float64)

tr_input, tr_target, vl_input, vl_target = hold_out_validation_stratified(x_i, d, monk_config.SPLIT)

# =============================================================================
# 2. GRID SEARCH
# =============================================================================
gs = GridSearch(
    units_list=[
        [3],  # best
        [4],
        #[3, 3],
        #[4,4]
    ],
    n_outputs=[monk_config.N_OUTPUTS],
    f_act_hidden=[sigmaf],
    f_act_output=[sigmaf],

    learning_rate=[0.25],

    use_decay=[False],
    decay_factor=[0.9],
    decay_step=[10],

    mini_batch_size=[len(tr_input)],

    epochs=[200],

    early_stopping=[True],
    patience=[15, 20],
    epsilon=[1e-25],

    momentum=[True],
    alpha_mom=[0.9],
    max_gradient_norm=[100],

    split=[monk_config.SPLIT],
    verbose=[False],
    validation=[True],
    lambdal2=[0.0]
)

print("\n🚀 Avvio Grid Search...")
best_config, best_acc_gs = gs.run_for_monk_holdout(tr_input, tr_target, vl_input, vl_target)

print("\n" + "═" * 60)
print(f"🏆 MIGLIOR CONFIGURAZIONE TROVATA (Val Acc: {best_acc_gs:.2%})")
print("═" * 60)
for k, v in best_config.items():
    print(f" • {k:<20}: {v}")
print("═" * 60)

# =============================================================================
# 3. FINAL RETRAINING & STATISTICAL ASSESSMENT
# =============================================================================

print("\n🔁 Caricamento Test Set...")
try:
    x_test, d_test = load_monks_data("../data/monk/test_data/monks-3.test")
    x_test = x_test.to_numpy().astype(np.float64)
    d_test = d_test.to_numpy().astype(np.float64)
    has_test_set = True
    print(f"✅ Test Set caricato: {x_test.shape[0]} patterns")
except:
    print("⚠️ Nessun Test Set trovato. I risultati saranno limitati al training.")
    has_test_set = False
    x_test, d_test = None, None


def evaluate_configuration_with_restarts(label, config, x_tr, y_tr, x_ts, y_ts, n_trials=10):
    print("\n" + "█" * 60)
    print(f"▶ ASSESSMENT: {label} (su {n_trials} inizializzazioni)")
    print(
        f"  Config: L2={config.get('lambdal2', 0)}, Eta={config.get('learning_rate')}, Batch={config.get('mini_batch_size')}")
    print("█" * 60)

    mses, accs_tr, accs_ts = [], [], []

    for i in range(n_trials):
        print(f"   ↳ Run {i + 1}/{n_trials}...", end="\r")

        # 1. Reinizializza
        trainer = TrainerMonk(input_size=x_tr.shape[1], **config)
        trainer.verbose = True

        # 2. Train
        final_mse, _ = trainer.fit(x_tr, y_tr, ts_x=x_ts, ts_d=y_ts)
        print(final_mse)
        # 3. CALCOLO ESPLICITO ACCURACY (FIX)
        # Calcoliamo l'accuracy di TRAIN chiamando direttamente la funzione interna
        final_acc_tr = trainer._compute_accuracy_internal(x_tr, y_tr)

        mses.append(final_mse)
        accs_tr.append(final_acc_tr)

        if x_ts is not None:
            acc_ts = trainer._compute_accuracy_internal(x_ts, y_ts)
            accs_ts.append(acc_ts)

    print(f"   ✅ Completato {n_trials} run.                 ")

    # Calcolo statistiche
    mean_mse, std_mse = np.mean(mses), np.std(mses)
    mean_tr, std_tr = np.mean(accs_tr), np.std(accs_tr)

    print("-" * 60)
    print(f"   MSE (Train):       {mean_mse:.5f} ± {std_mse:.5f}")
    print(f"   Accuracy (Train):  {mean_tr:.2%} ± {std_tr:.2%}")

    mean_ts, std_ts = 0.0, 0.0
    if x_ts is not None:
        mean_ts, std_ts = np.mean(accs_ts), np.std(accs_ts)
        print(f"   Accuracy (TEST):   {mean_ts:.2%} ± {std_ts:.2%}")
    print("-" * 60)

    return mean_mse, mean_tr, mean_ts


# --- SETUP RUNS ---
N_TRIALS = 20  # Numero di inizializzazioni diverse su cui fare la media

# 1. Configurazione Ottimale (Best Grid Search)
evaluate_configuration_with_restarts("Best Configuration Found", best_config, x_i, d, x_test, d_test, n_trials=N_TRIALS)

# 2. Configurazione di Confronto (Toggle L2)
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

print("\n🏁 Assessment completato.")