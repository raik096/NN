import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import textwrap
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import textwrap
from datetime import datetime

def plot_grid_analysis_cup2(results, top_k=4, save_path="results/grid_kfold"):
    
    # Crea cartella
    timestamp = datetime.now().strftime("%H%M%S")
    out_dir = os.path.join(save_path, f"run_{timestamp}")
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    # Ordina i risultati (miglior MEE di validazione per primo)
    sorted_res = sorted(results, key=lambda x: np.mean([fold[-1] for fold in x['vl_metric']]))

    # 1. Summary veloce (tutti insieme)
    print("Salvataggio summary...")
    top_12 = sorted_res[:12]
    rows = min(4, (len(top_12) + 2) // 3)
    fig, axes = plt.subplots(rows, 3, figsize=(18, 5*rows))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(top_12):
            res = top_12[i]
            vl_mean = np.mean([f[-1] for f in res['vl_metric']])
            ax.set_title(f"Rank #{i+1} | Mean VL: {vl_mean:.4f}", fontweight='bold')
            
            # Plot veloce sovrapposto
            k = len(res['vl_metric'])
            colors = cm.viridis(np.linspace(0, 0.9, k))
            for j in range(k):
                ax.plot(res['tr_metric'][j], 'r', alpha=0.3)
                ax.plot(res['vl_metric'][j], color=colors[j], linestyle='--')
            
            ax.set_yscale('log')
        else:
            ax.axis('off')
            
    plt.tight_layout()
    plt.savefig(f"{out_dir}/summary_top12.png")
    plt.close()

    # 2. Plot Dettagliati (K-Fold esplosi)
    print(f"Salvataggio top {top_k} modelli...")
    for i in range(min(top_k, len(sorted_res))):
        plot_single_model(sorted_res[i], i + 1, out_dir)


def plot_single_model(res, rank, folder):
    tr_data = res['tr_metric']
    vl_data = res['vl_metric']
    k = len(vl_data)

    # Setup figura larga per i subplot affiancati
    fig, axs = plt.subplots(1, k, figsize=(5 * k, 6), sharey=True)
    if k == 1: axs = [axs]

    # Plot di ogni fold
    val_means = []
    for i in range(k):
        ax = axs[i]
        tr, vl = tr_data[i], vl_data[i]
        epochs = range(1, len(tr) + 1)

        ax.plot(epochs, tr, 'r-', label='Train', lw=2)
        ax.plot(epochs, vl, 'b--', label='Val', lw=2)
        
        # Pallino finale (senza testo numerico)
        ax.scatter(epochs[-1], vl[-1], c='blue', s=40)

        ax.set_title(f"FOLD {i+1}")
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, ls='--')
        
        val_means.append(vl[-1])
        if i == 0:
            ax.set_ylabel("MEE (log)")
            ax.legend()

    # Titolo generale
    avg_mee = np.mean(val_means)
    fig.suptitle(f"Rank #{rank} - Mean MEE: {avg_mee:.4f}", fontsize=16, fontweight='bold')

    # Preparazione testo configurazione
    cfg = res['params']
    # Filtro parametri inutili per pulire la stringa
    exclude = ['verbose', 'validation', 'n_outputs', 'early_stopping', 'epsilon', 'split']
    cfg_items = [f"{k}={v}" for k, v in cfg.items() if k not in exclude]
    cfg_str = ", ".join(cfg_items)
    
    # Formattazione testo (va a capo se troppo lungo)
    wrapped_txt = "\n".join(textwrap.wrap(cfg_str, width=110))

    # Aggiungo il testo in basso al centro
    fig.text(0.5, 0.02, wrapped_txt, ha='center', va='bottom', fontsize=10, family='monospace',
             bbox=dict(boxstyle="round", facecolor='#f0f0f0', alpha=0.5))

    # Lascio spazio sotto per il testo
    plt.subplots_adjust(bottom=0.22, top=0.88, wspace=0.1)

    plt.savefig(f"{folder}/rank_{rank}_kfold.png", bbox_inches='tight')
    plt.close()


    
def plot_errors_with_validation_error(trainer, training_time, save_path="results/cup/plots"):
    
    # Crea cartella se non esiste
    if not os.path.exists(save_path): os.makedirs(save_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Estrazione dati per comodità
    epochs = range(1, len(trainer.tr_mee_history) + 1)
    tr_mee = trainer.tr_mee_history
    vl_mee = trainer.vl_mee_history # Qui dentro c'è il Test Set ora
    tr_mse = trainer.tr_mse_history
    vl_mse = trainer.vl_mse_history

    # Trova il punto migliore
    best_vl = min(vl_mee)
    best_ep = np.argmin(vl_mee) + 1

    # Setup figura (2 grafici affiancati)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- GRAFICO 1: MEE (Metrica target) ---
    ax1.plot(epochs, tr_mee, 'r-', label=f'Train (final={tr_mee[-1]:.4f})', lw=2)
    # Cambio etichetta da Val a Test
    ax1.plot(epochs, vl_mee, 'b--', label=f'Test (best={best_vl:.4f})', lw=2)
    
    # Segna il punto migliore con un pallino semplice (niente stella)
    ax1.scatter(best_ep, best_vl, c='green', marker='o', s=50, label=f'Best @ ep {best_ep}', zorder=10)

    ax1.set_title("MEE Curve")
    ax1.set_ylabel("Error")
    ax1.set_xlabel("Epochs")
    ax1.grid(True, alpha=0.3, ls='--')
    ax1.legend()

    # --- GRAFICO 2: MSE (Loss function) ---
    ax2.plot(epochs, tr_mse, 'r-', label='Train Loss')
    # Cambio etichetta specifica richiesta
    ax2.plot(epochs, vl_mse, 'b--', label='Test Internal Loss')

    ax2.set_title("MSE Loss (Log Scale)")
    ax2.set_xlabel("Epochs")
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, ls='--')
    ax2.legend()

    # --- Info Box Orizzontale ---
    # Tutto su una riga separato da spazi ampi o barre
    info_txt = (f"Training Time: {training_time:.2f}s   |   "
                f"Final TR MEE: {tr_mee[-1]:.5f}   |   "
                f"Best Test MEE: {best_vl:.5f}")
    
    fig.text(0.5, 0.02, info_txt, ha='center', fontsize=11, 
             bbox=dict(boxstyle="round", facecolor='#f0f0f0', alpha=0.5))

    # Aggiusta i margini per far stare il testo orizzontale
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Salvataggio
    fname = f"{save_path}/cup_final_assessment_{timestamp}.png"
    plt.savefig(fname, dpi=200)
    plt.close()
    
    print(f"Grafico salvato: {fname}")


def plot_errors(trainer, training_time, save_path="results/cup/plots"):
    
    # Crea cartella e timestamp
    if not os.path.exists(save_path): os.makedirs(save_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Dati
    epochs = range(1, len(trainer.tr_mee_history) + 1)
    tr_mee = trainer.tr_mee_history
    tr_mse = trainer.tr_mse_history

    # Setup figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- GRAFICO 1: MEE ---
    ax1.plot(epochs, tr_mee, 'r-', label=f'Train (final={tr_mee[-1]:.4f})', lw=2)
    
    ax1.set_title("Training MEE")
    ax1.set_ylabel("MEE")
    ax1.set_xlabel("Epochs")
    ax1.grid(True, alpha=0.3, ls='--')
    ax1.legend()

    # --- GRAFICO 2: MSE ---
    ax2.plot(epochs, tr_mse, 'r-', label=f'Train Loss (final={tr_mse[-1]:.4f})', lw=2)
    
    ax2.set_title("Training MSE (Log Scale)")
    ax2.set_xlabel("Epochs")
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, ls='--')
    ax2.legend()

    # --- Info Box ---
    info_txt = f"Final MEE: {tr_mee[-1]:.5f}  |  Time: {training_time:.2f}s"
    fig.text(0.5, 0.02, info_txt, ha='center', fontsize=11, 
             bbox=dict(boxstyle="round", facecolor='#f0f0f0', alpha=0.5))

    # Layout e salvataggio
    plt.suptitle("CUP Training Profile (No Validation)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    fname = f"{save_path}/cup_train_only_{timestamp}.png"
    plt.savefig(fname, dpi=200)
    plt.close()

    print(f"Grafico salvato: {fname}")