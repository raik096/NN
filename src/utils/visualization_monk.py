import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import textwrap

def plot_grid_search_analysis_monk2(all_results, relative_path="results/monk/grid_search", top_k=6):
    
    # Creazione cartelle
    if not os.path.exists(relative_path):
        os.makedirs(relative_path)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    details_dir = os.path.join(relative_path, f"details_{timestamp}")
    if not os.path.exists(details_dir):
        os.makedirs(details_dir)
    
    # Ordiniamo i risultati (migliore accuracy prima)
    sorted_res = sorted(all_results, key=lambda x: max(x['vl_accuracy']), reverse=True)
    top_models = sorted_res[:top_k]
    
    # ---------------------------------------------------------
    # 1. GRIGLIA RIASSUNTIVA (Summary)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    plt.suptitle("Top 6 Modelli MONK", fontsize=16, fontweight='bold')
    
    for i in range(6):
        ax = axes[i]
        
        if i < len(top_models):
            res = top_models[i]
            mse = res['tr_mse']
            best_acc = max(res['vl_accuracy'])
            epochs = range(1, len(mse) + 1)
            
            # Plot semplice dell'MSE
            ax.plot(epochs, mse, color='red', label='Train MSE')
            
            # Titolo standard
            ax.set_title(f"Rank #{i+1} - Val Acc: {best_acc*100:.1f}%", fontsize=10, fontweight='bold')
            ax.set_yscale('log')
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # Testo parametri piccolo (formattato stretto per la griglia)
            params_str = format_params(res['params'], layout='grid')
            ax.text(0.5, -0.15, params_str, transform=ax.transAxes, 
                    ha='center', va='top', fontsize=8, 
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.15, hspace=0.4)
    
    summary_name = f"Summary_Top6_{timestamp}.png"
    plt.savefig(os.path.join(relative_path, summary_name))
    plt.close()
    print(f"✅ Grafico sommario salvato in: {relative_path}")

    # ---------------------------------------------------------
    # 2. DETTAGLI SINGOLI (Orizzontali)
    # ---------------------------------------------------------
    print(f"   ...Generazione dei {len(top_models)} grafici di dettaglio...")
    
    for i, res in enumerate(top_models):
        plot_single_detail(res, i+1, details_dir)


def plot_single_detail(res, rank, folder):
    
    tr_mse = res['tr_mse']
    tr_acc = res['tr_accuracy']
    vl_acc = res['vl_accuracy']
    epochs = range(1, len(tr_mse) + 1)
    
    best_val = max(vl_acc)
    best_ep = np.argmax(vl_acc) + 1
    
    # Figura larga
    # bottom=0.20 è sufficiente per il testo orizzontale
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    plt.subplots_adjust(bottom=0.20) 
    
    # Grafico 1: MSE
    ax1.plot(epochs, tr_mse, color='red', linewidth=2, label='Train MSE')
    ax1.set_title("Training Loss (MSE)", fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Grafico 2: Accuracy
    ax2.plot(epochs, tr_acc, color='red', label='Train Acc')
    ax2.plot(epochs, vl_acc, color='blue', linestyle='--', label='Val Acc')
    ax2.scatter(best_ep, best_val, color='green', s=100, marker='*', label='Best Val')
    
    ax2.set_title(f"Accuracy (Best: {best_val*100:.2f}%)", fontweight='bold')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right')
    
    # Testo Parametri (Layout ORIZZONTALE)
    params_txt = format_params(res['params'], layout='horizontal')
    
    info_box = (
        f"MODELLO #{rank}  |  Best Val Acc: {best_val*100:.2f}% (Epoca {best_ep})  |  MSE Finale: {tr_mse[-1]:.5f}\n"
        f"----------------------------------------------------------------------------------------------------\n"
        f"{params_txt}"
    )
    
    # Inseriamo il testo in basso al centro
    plt.figtext(0.5, 0.03, info_box, ha="center", va="bottom", fontsize=10, family='monospace',
                bbox=dict(boxstyle="round", facecolor="#f0f0f0", edgecolor="gray"))
    
    filename = f"Rank_{rank}_Detail.png"
    plt.savefig(os.path.join(folder, filename))
    plt.close()


def format_params(params, layout='horizontal'):
    # Parametri da ignorare
    ignore = ['verbose', 'validation', 'split', 'early_stopping', 'epsilon', 'n_outputs']
    
    parts = []
    for k, v in params.items():
        if k in ignore: continue
        
        # Abbreviazioni
        label = k
        if "mini_batch_size" in k: label = "Batch"
        elif "learning_rate" in k: label = "LR"
        elif "decay_factor" in k: label = "Decay"
        elif "decay_step" in k: label = "DecayStep"
        elif "momentum" in k: label = "Mom"
        elif "alpha" in k: label = "Alpha"
        elif "lambda" in k: label = "L2"
        elif "units" in k: label = "Units"
        elif "act" in k: label = "Act"
        
        val = str(v)
        if hasattr(v, '__name__'): val = v.__name__
        
        parts.append(f"{label}={val}")
    
    # Uniamo tutto in una stringa unica separata da virgole
    full_string = ", ".join(parts)
    
    if layout == 'grid':
        # Per la griglia piccola, andiamo a capo spesso (stretto)
        return "\n".join(textwrap.wrap(full_string, width=50))
    else:
        # Per il dettaglio orizzontale, usiamo una larghezza ampia (es. 110 caratteri)
        # così sembra un paragrafo e non una lista
        return "\n".join(textwrap.wrap(full_string, width=110))
    


def plot_errors_with_validation_error(trainer, training_time, save_path="results/cup/plots"):
    
    # Crea cartella se non esiste
    if not os.path.exists(save_path): os.makedirs(save_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Estrazione dati per comodità
    epochs = range(1, len(trainer.tr_mee_history) + 1)
    tr_mee = trainer.tr_mee_history
    vl_mee = trainer.vl_mee_history
    tr_mse = trainer.tr_mse_history
    vl_mse = trainer.vl_mse_history

    # Trova il punto migliore (minimo MEE di validazione)
    best_vl = min(vl_mee)
    best_ep = np.argmin(vl_mee) + 1

    # Setup figura (2 grafici affiancati)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- GRAFICO 1: MEE (Metrica target) ---
    ax1.plot(epochs, tr_mee, 'r-', label=f'Train (final={tr_mee[-1]:.4f})', lw=2)
    ax1.plot(epochs, vl_mee, 'b--', label=f'Val (best={best_vl:.4f})', lw=2)
    
    # Segna il punto migliore con una stella
    ax1.scatter(best_ep, best_vl, c='green', marker='*', s=120, label=f'Best @ ep {best_ep}', zorder=10)

    ax1.set_title("MEE Curve")
    ax1.set_ylabel("Error")
    ax1.set_xlabel("Epochs")
    ax1.grid(True, alpha=0.3, ls='--')
    ax1.legend()

    # --- GRAFICO 2: MSE (Loss function) ---
    ax2.plot(epochs, tr_mse, 'r-', label='Train Loss')
    ax2.plot(epochs, vl_mse, 'b--', label='Val Loss')

    ax2.set_title("MSE Loss (Log Scale)")
    ax2.set_xlabel("Epochs")
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, ls='--')
    ax2.legend()

    # --- Info Box in basso ---
    info_txt = (f"Training Time: {training_time:.2f}s  |  "
                f"Best VL MEE: {best_vl:.5f}  |  "
                f"Final TR MEE: {tr_mee[-1]:.5f}")
    
    fig.text(0.5, 0.02, info_txt, ha='center', fontsize=11, 
             bbox=dict(boxstyle="round", facecolor='#f0f0f0', alpha=0.5))

    # Aggiusta i margini per far stare il testo
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Salvataggio
    fname = f"{save_path}/cup_training_{timestamp}.png"
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

import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import textwrap

def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def plot_monk(
    tr_mse_history: list,
    vl_mse_history: list,
    tr_accuracy_history: list,
    vl_accuracy_history: list,
    ts_accuracy_history: list,
    training_time: float,
    config: dict = None,
    relative_path="results/monk/plots"
    ):

    dir_path = _ensure_dir(relative_path)
    epochs = np.arange(1, len(tr_mse_history) + 1)
    
    # Layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7)) # Altezza ridotta leggermente
    plt.subplots_adjust(bottom=0.20) # Meno spazio sotto perché il testo è orizzontale
    
    # --- GRAFICO 1: MSE ---
    final_tr_mse = tr_mse_history[-1]
    ax1.plot(epochs, tr_mse_history, color='#d62728', linewidth=2, label=f'MSE Train ({final_tr_mse:.2e})')
    if len(vl_mse_history) > 0:
        ax1.plot(epochs, vl_mse_history, color='#2ca02c', linestyle='--', label='MSE Val')

    ax1.set_title("Mean Squared Error (Loss)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE (log)")
    ax1.set_yscale("log")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend()
    
    # --- GRAFICO 2: ACCURACY ---
    final_tr_acc = tr_accuracy_history[-1]
    ax2.plot(epochs, tr_accuracy_history, color='#d62728', linewidth=2, label=f'Train Acc ({final_tr_acc:.1%})')
    
    if len(ts_accuracy_history) > 0:
        t_epochs = np.arange(1, len(ts_accuracy_history) + 1)
        final_ts_acc = ts_accuracy_history[-1]
        best_ts_acc = max(ts_accuracy_history)
        ax2.plot(t_epochs, ts_accuracy_history, color='#1f77b4', linestyle='-', linewidth=2, label=f'Test Acc ({final_ts_acc:.1%})')
        best_epoch = np.argmax(ts_accuracy_history) + 1
        ax2.scatter(best_epoch, best_ts_acc, color='#1f77b4', marker='*', s=150, zorder=5, edgecolors='black')

    if len(vl_accuracy_history) > 0:
        v_epochs = np.arange(1, len(vl_accuracy_history) + 1)
        ax2.plot(v_epochs, vl_accuracy_history, color='#2ca02c', linestyle=':', alpha=0.7, label='Val Acc')

    ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_title("Classification Accuracy", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Epoch")
    ax2.set_ylim([-0.05, 1.05]) 
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc='lower right')
    
    # --- INFO BOX ORIZZONTALE ---
    if config:
        # Crea stringa orizzontale
        formatted_params = _format_params_horizontal(config)
        
        info_text = (
            f"TRAINING REPORT | Time: {training_time:.2f}s | "
            f"Final Train Acc: {final_tr_acc:.2%} | Final Test Acc: {ts_accuracy_history[-1] if ts_accuracy_history else 0:.2%}\n"
            f"CONFIG: {formatted_params}"
        )
        
        plt.figtext(
            0.5, 0.03, # Posizione in basso
            info_text, 
            ha="center", 
            fontsize=10, 
            family='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="#aaaaaa")
        )

    # Titolo senza ORA
    plt.suptitle(f"MONK Training Analysis", fontsize=16, fontweight="bold")
    
    fname = _generate_filename("MONK_Assessment")
    full_path = os.path.join(dir_path, fname)
    plt.savefig(full_path, dpi=300)
    print(f"✅ Grafico MONK salvato in: {full_path}")
    plt.close()

def _format_params_horizontal(params):
    """Formatta il dizionario in una stringa lunga orizzontale, andando a capo solo se necessario"""
    ignore = ['verbose', 'validation', 'split', 'early_stopping', 'epsilon', 'n_outputs']
    
    items = []
    for k, v in params.items():
        if k in ignore: continue
        
        # Abbreviazioni chiave
        key = k.replace("mini_batch_size", "BS").replace("learning_rate", "LR") \
               .replace("decay_factor", "D_Fact").replace("decay_step", "D_Step") \
               .replace("momentum", "Mom").replace("alpha_mom", "A_Mom") \
               .replace("lambdal2", "L2").replace("max_gradient_norm", "Clip") \
               .replace("units_list", "Units")
        
        val = str(v)
        if hasattr(v, '__name__'): val = v.__name__ 
        if isinstance(v, list): val = str(v).replace(" ", "")
        
        items.append(f"{key}={val}")

    # Unisci tutto con virgole
    full_str = ", ".join(items)
    
    # Usa textwrap con una larghezza molto ampia (es. 130 caratteri) per forzare l'orizzontalità
    return "\n".join(textwrap.wrap(full_str, width=130))

def _format_params_short(params):
    """Versione compressa per i box piccoli della griglia"""
    return _format_params_horizontal(params) # Riutilizziamo la stessa logica ma il box è più stretto, quindi andrà a capo da solo"


def _generate_filename(prefix="plot"):
    """Genera un nome file unico basato sul timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.png"