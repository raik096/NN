import matplotlib.pyplot as plt
import numpy as np
import os
import time
from datetime import datetime

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

def plot_grid_analysis(all_results, top_k_individual=5, relative_path="results/grid_search"):
    """
    Analizza i risultati della Grid Search e produce:
    1. Un 'Summary Plot' 4x4 con i migliori 16 modelli (solo MSE).
    2. Una collezione di plot individuali per i migliori 'top_k_individual' modelli.
    
    Args:
        all_results (list): Lista di dizionari {params, tr_mse, vl_mse}.
        top_k_individual (int): Quanti grafici singoli salvare (es. i migliori 5).
        relative_path (str): Cartella base di salvataggio.
    """
    
    # 1. Setup Cartelle
    base_dir = _ensure_dir(relative_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sottocartella per i plot singoli per non intasare la root
    single_plots_dir = os.path.join(base_dir, f"details_{timestamp}")
    if top_k_individual > 0:
        os.makedirs(single_plots_dir, exist_ok=True)

    # 2. Ordinamento: dal MSE di validazione più basso (migliore) al più alto
    #    Assumiamo che tr_mse e vl_mse siano liste storiche
    sorted_results = sorted(all_results, key=lambda x: min(x['vl_mse']))
    
    # ==========================================
    # PARTE A: SUMMARY GRID (4x4)
    # ==========================================
    top_16 = sorted_results[:16]
    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(20, 16))
    axes = axes.flatten()
    
    plt.suptitle(f"Top 16 Configurations (Sorted by Val MSE) - {timestamp}", fontsize=18, fontweight='bold')
    
    for i in range(rows * cols):
        ax = axes[i]
        
        if i < len(top_16):
            res = top_16[i]
            _draw_single_mse_plot(ax, res, title_prefix=f"Rank #{i+1}")
            
            # Label assi solo sui bordi esterni per pulizia
            if i >= 12: ax.set_xlabel("Epoch")
            if i % 4 == 0: ax.set_ylabel("MSE (log)")
        else:
            ax.axis('off') # Nascondi riquadri vuoti

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    summary_fname = f"GRID_Summary_Top16_{timestamp}.png"
    summary_path = os.path.join(base_dir, summary_fname)
    plt.savefig(summary_path, dpi=200)
    print(f"✅ [Summary] Griglia 4x4 salvata in: {summary_path}")
    plt.close()

    # ==========================================
    # PARTE B: COLLEZIONE INDIVIDUALE (Top K)
    # ==========================================
    if top_k_individual > 0:
        print(f"   ...Generazione di {top_k_individual} grafici individuali...")
        
        for i in range(min(top_k_individual, len(sorted_results))):
            res = sorted_results[i]
            
            # Creiamo una figura nuova per ogni plot
            fig_single, ax_single = plt.subplots(figsize=(10, 6))
            
            # Disegna
            params_str = _format_params(res['params'])
            _draw_single_mse_plot(ax_single, res, title_prefix=f"Rank #{i+1}", font_scale=1.2)
            
            # Aggiungi Info Box dettagliata solo nel plot singolo
            final_tr = res['tr_mse'][-1]
            best_vl = min(res['vl_mse'])
            info_text = (f"Params: {params_str}\n"
                         f"Final TR MSE: {final_tr:.5e}\n"
                         f"Best VL MSE: {best_vl:.5e}")
            
            ax_single.text(0.5, -0.2, info_text, ha='center', va='top', transform=ax_single.transAxes,
                           bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3))
            
            ax_single.set_xlabel("Epoch")
            ax_single.set_ylabel("MSE (log)")
            ax_single.set_title(f"Configuration Rank #{i+1}", fontweight='bold')
            
            plt.tight_layout()
            
            # Nome file pulito
            safe_params = params_str.replace(", ", "_").replace("=", "").replace(".", "")[:50]
            fname_single = f"Rank_{i+1:02d}_{safe_params}.png"
            plt.savefig(os.path.join(single_plots_dir, fname_single), dpi=150, bbox_inches='tight')
            plt.close()
            
        print(f"✅ [Collection] {top_k_individual} grafici dettagliati salvati in: {single_plots_dir}")


"""def _draw_single_mse_plot(ax, result, title_prefix="", font_scale=1.0):
Funzione helper per disegnare lo stile 'Monk' su un asse dato.
    tr_hist = result['tr_mse']
    vl_hist = result['vl_mse']
    params = result['params']
    epochs = np.arange(1, len(tr_hist) + 1)
    
    best_vl = min(vl_hist)
    best_epoch = np.argmin(vl_hist) + 1
    
    # 1. Plot Training (Rosso)
    ax.plot(epochs, tr_hist, color='red', label='Train MSE', linewidth=1.5 * font_scale, alpha=0.9)
    
    # 2. Plot Validation (Blu Tratteggiato)
    ax.plot(epochs, vl_hist, color='blue', linestyle='--', label='Val MSE', linewidth=1.5 * font_scale, alpha=0.9)
    
    # 3. Stella Verde sul punto migliore
    ax.scatter(best_epoch, best_vl, color='green', marker='*', s=100 * font_scale, zorder=5, edgecolors='black')
    
    # Stile
    ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.4)
    
    # Titolo breve
    param_str = _format_params(params)
    if len(param_str) > 40 * font_scale: 
        param_str = param_str[:int(40*font_scale)] + "..."
    
    ax.set_title(f"{title_prefix}\n{param_str}\nBest VL: {best_vl:.5f}", fontsize=9 * font_scale, fontweight='bold')
    
    # Legenda (piccola per non coprire)
    ax.legend(fontsize=8 * font_scale, loc='upper right')"""

def _format_params(params):
    """Converte dizionario parametri in stringa leggibile: 'lr=0.01, mom=0.9'"""
    return ", ".join([f"{k}={v}" for k, v in params.items()])

def plot_errors_with_validation_error(
    trainer,
    training_time: float,
    relative_path="results/cup/plots"
):
    dir_path = _ensure_dir(relative_path)

    epochs = np.arange(1, len(trainer.tr_mee_history) + 1)

    # Valori finali
    final_tr_mee = trainer.tr_mee_history[-1]
    final_vl_mee = trainer.vl_mee_history[-1]
    best_vl_mee = min(trainer.vl_mee_history)
    best_epoch = np.argmin(trainer.vl_mee_history) + 1
    
    # Setup Figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # ==========================
    # GRAFICO 1: MEE (Metrica Principale)
    # ==========================
    ax1.plot(
        epochs,
        trainer.tr_mee_history,
        color='red',           # STILE MONK: Training rosso
        linewidth=2,
        label=f"MEE_tr (final={final_tr_mee:.4f})",
        alpha=0.9
    )

    ax1.plot(
        epochs,
        trainer.vl_mee_history,
        color='blue',          # STILE MONK: Validation blu
        linestyle="--",        # STILE MONK: Tratteggiato
        label=f"MEE_vl (best={best_vl_mee:.4f})",
        linewidth=2
    )

    # Marker per il punto migliore (Stella Verde come Monk)
    ax1.scatter(
        best_epoch,
        best_vl_mee,
        color="green",         # STILE MONK: Stella verde
        marker='*',
        s=150,
        edgecolors='black',
        zorder=5,
        label=f"Best vl @ ep {best_epoch}"
    )

    ax1.set_title("Mean Euclidean Error (MEE)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("MEE", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(fontsize=11)

    # ==========================
    # GRAFICO 2: MSE (Loss)
    # ==========================
    ax2.plot(
        epochs,
        trainer.tr_mse_history,
        color='red',           # STILE MONK
        label="MSE_tr",
        alpha=0.9
    )

    ax2.plot(
        epochs,
        trainer.vl_mse_history,
        color='blue',          # STILE MONK
        linestyle="--",
        label="MSE_vl",
        alpha=0.9
    )

    ax2.set_title("Mean Squared Error (Loss)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("MSE (log scale)", fontsize=12)
    ax2.set_yscale("log")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(fontsize=11)

    # ==========================
    # INFO BOX (Stile Monk)
    # ==========================
    info_text = (
        f"Final MEE_tr: {final_tr_mee:.5f}\n"
        f"Final MEE_vl: {final_vl_mee:.5f}\n"
        f"Best MEE_vl: {best_vl_mee:.5f} (ep {best_epoch})\n"
        f"Training time: {training_time:.2f}s"
    )

    fig.text(
        0.5, 0.02,
        info_text,
        ha="center",
        fontsize=12,
        # STILE MONK: Sfondo lightblue invece di wheat
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3) 
    )

    plt.suptitle(
        f"CUP Training Analysis ({datetime.now().strftime('%d/%m/%Y')})",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    fname = _generate_filename("CUP_tr_vs_vl")
    full_path = os.path.join(dir_path, fname)
    plt.savefig(full_path, dpi=300)

    print(f"✅ Grafico CUP salvato in: {full_path}")


def plot_errors(trainer, training_time: float, relative_path="results/cup/plots"):
    """
    Plot solo training (no validation) - Stile adattato al Monk (tutto Rosso)
    """
    dir_path = _ensure_dir(relative_path)
    epochs = np.arange(1, len(trainer.tr_mee_history) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # MEE
    ax1.plot(
        epochs,
        trainer.tr_mee_history,
        color='red',
        linewidth=2,
        label=f"MEE_tr (final={trainer.tr_mee_history[-1]:.4f})"
    )
    ax1.set_title("Training MEE", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MEE")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend()

    # MSE
    ax2.plot(
        epochs,
        trainer.tr_mse_history,
        color='red',
        linewidth=2,
        label=f"MSE_tr (final={trainer.tr_mse_history[-1]:.4f})"
    )
    ax2.set_title("Training MSE (Loss)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE")
    ax2.set_yscale("log")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend()

    # Info Box semplificato
    info_text = f"Final MEE: {trainer.tr_mee_history[-1]:.5f} | Time: {training_time:.2f}s"
    fig.text(
        0.5, 0.02,
        info_text,
        ha="center",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3)
    )

    plt.suptitle(
        f"CUP Training Profile (No Validation)", 
        fontsize=16, 
        fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    fname = _generate_filename("CUP_train_only")
    plt.savefig(os.path.join(dir_path, fname))

    print(f"✅ Grafico CUP (TrainOnly) salvato in: {os.path.join(dir_path, fname)}")


def plot_accuracy(trainer, training_time: float, relative_path="results/plots"):
    
    dir_path = _ensure_dir(relative_path)
    epochs = np.arange(1, len(trainer.tr_mee_history) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(
        epochs,
        trainer.tr_mee_history,
        label=f"MEE_tr (final={trainer.accuracy_history[-1]:.4f})"
    )
    ax1.set_title("Training accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("accuracy")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(
        epochs,
        trainer.tr_mse_history,
        label=f"MSE_tr (final={trainer.accuracy_history[-1]:.4f})"
    )
    ax2.set_title("Training MSE")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle(f"Training Profile – time: {training_time:.2f}s")
    plt.tight_layout()

    fname = _generate_filename("Monk_accuracy_training")
    plt.savefig(os.path.join(dir_path, fname))

    print(f"✅ Grafico salvato in: {os.path.join(dir_path, fname)}")

def _ensure_dir(relative_path="results/plots"):
    """
    Crea una directory RELATIVA ALLA CARTELLA DEL PROGETTO
    """

    PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))

    print("PROJECT_ROOT", PROJECT_ROOT)

    full_path = os.path.join(PROJECT_ROOT, relative_path)

    if not os.path.exists(full_path):
        os.makedirs(full_path, exist_ok=True)

    return full_path



def _generate_filename(prefix="plot"):
    """Genera un nome file unico basato sul timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.png"

import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import textwrap

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

    
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import textwrap  # Necessario per mandare a capo il testo

def plot_grid_analysis_cup(all_results, top_k_individual=5, relative_path="results/grid_search"):
    """
    Analizza i risultati della Grid Search e produce:
    1. Un 'Summary Plot' 4x3 (Top 12) con le configurazioni scritte SOTTO ogni grafico.
    2. Una collezione di plot individuali per i migliori 'top_k_individual' modelli.
    """
    
    # 1. Setup Cartelle
    base_dir = _ensure_dir(relative_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    single_plots_dir = os.path.join(base_dir, f"details_{timestamp}")
    if top_k_individual > 0:
        os.makedirs(single_plots_dir, exist_ok=True)

    # 2. Ordinamento: dal migliore al peggiore
    sorted_results = sorted(all_results, key=lambda x: min(x['vl_mse']))
    
    # ==========================================
    # PARTE A: SUMMARY GRID (Top 12 - 4x3)
    # ==========================================
    # Prendiamo solo i primi 12
    top_12 = sorted_results[:12]
    
    # Impostiamo una griglia 4 righe x 3 colonne
    rows, cols = 4, 3
    
    # Aumentiamo l'altezza della figura (24) per far stare comodamente il testo sotto
    fig, axes = plt.subplots(rows, cols, figsize=(20, 24))
    axes = axes.flatten()
    
    plt.suptitle(f"Top 12 Configurations (Sorted by Val MSE) - {timestamp}", fontsize=20, fontweight='bold', y=0.99)
    
    for i in range(rows * cols):
        ax = axes[i]
        
        if i < len(top_12):
            res = top_12[i]
            
            # Titolo Semplice (Solo Rank e Score)
            best_vl = min(res['vl_mse'])
            short_title = f"Rank #{i+1} | Best VL MSE: {best_vl:.5f}"
            
            # Disegna il grafico
            _draw_single_mse_plot(ax, res, title_prefix=short_title, show_params_in_title=False)
            
            # --- SCRITTURA CONFIGURAZIONE SOTTO IL PLOT ---
            # Formatta i parametri per andare a capo
            params_text = _format_params_wrapped(res['params'])
            
            # Inserisce il testo sotto l'asse X
            # (0.5 = centro orizzontale, -0.15 = sotto l'asse)
            ax.text(
                0.5, -0.18, 
                params_text, 
                transform=ax.transAxes, 
                ha='center', va='top', 
                fontsize=9, 
                family='monospace',
                bbox=dict(boxstyle="round,pad=0.5", fc="#f8f9fa", ec="#dee2e6", alpha=0.9)
            )
            
            # Label asse Y solo sulla colonna di sinistra
            if i % cols == 0: 
                ax.set_ylabel("MSE (log)", fontsize=10)
            
        else:
            ax.axis('off') # Nascondi riquadri vuoti se ce ne sono meno di 12

    # Aggiusta la spaziatura: hspace alto per il testo, wspace per separare colonne
    plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.65, wspace=0.2)
    
    summary_fname = f"GRID_Summary_Top12_{timestamp}.png"
    summary_path = os.path.join(base_dir, summary_fname)
    
    # bbox_inches='tight' è fondamentale per non tagliare il testo fuori dai bordi
    plt.savefig(summary_path, dpi=200, bbox_inches='tight')
    print(f"✅ [Summary] Griglia 4x3 salvata in: {summary_path}")
    plt.close()

    # ==========================================
    # PARTE B: COLLEZIONE INDIVIDUALE
    # ==========================================
    # (Codice invariato per i plot singoli)
    if top_k_individual > 0:
        print(f"   ...Generazione di {top_k_individual} grafici individuali...")
        for i in range(min(top_k_individual, len(sorted_results))):
            res = sorted_results[i]
            fig_single, ax_single = plt.subplots(figsize=(10, 6))
            
            _draw_single_mse_plot(ax_single, res, title_prefix=f"Rank #{i+1}", show_params_in_title=True)
            
            # Info box extra
            final_tr = res['tr_mse'][-1]
            best_vl = min(res['vl_mse'])
            info_text = (f"Final TR MSE: {final_tr:.5e}\nBest VL MSE: {best_vl:.5e}")
            ax_single.text(0.5, -0.2, info_text, ha='center', va='top', transform=ax_single.transAxes,
                           bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3))
            
            ax_single.set_xlabel("Epoch")
            ax_single.set_ylabel("MSE (log)")
            
            plt.tight_layout()
            safe_rank = f"Rank_{i+1:02d}"
            fname_single = f"{safe_rank}.png"
            plt.savefig(os.path.join(single_plots_dir, fname_single), dpi=150)
            plt.close()

# --- HELPER FUNCTIONS ---

def _format_params_wrapped(params):
    """
    Formatta i parametri pulendo le chiavi e andando a capo
    per adattarsi allo spazio sotto il grafico.
    """
    # Lista di chiavi da ignorare per risparmiare spazio
    ignored = ['verbose', 'validation', 'split', 'n_outputs', 'early_stopping', 'epsilon']
    
    items = []
    for k, v in params.items():
        if k in ignored: continue
        
        # Accorcia nomi lunghi delle chiavi
        key = k.replace("mini_batch_size", "batch").replace("learning_rate", "lr") \
               .replace("decay_factor", "d_fact").replace("decay_step", "d_step") \
               .replace("momentum", "mom").replace("alpha_mom", "a_mom") \
               .replace("lambdal2", "l2").replace("max_gradient_norm", "clip") \
               .replace("units_list", "units")

        # Formatta valore (es. nomi funzioni)
        val = str(v)
        if hasattr(v, '__name__'): val = v.__name__ # es: relu
        if isinstance(v, list): val = str(v).replace(" ", "") # Rimuove spazi nelle liste [10,10]
        
        items.append(f"{key}={val}")

    full_str = ", ".join(items)
    # Usa textwrap per spezzare le righe ogni 50 caratteri circa
    return "\n".join(textwrap.wrap(full_str, width=50))

def _draw_single_mse_plot(ax, result, title_prefix="", font_scale=1.0, show_params_in_title=False):
    """Disegna le curve. Se show_params_in_title=False, mette solo il titolo breve."""
    tr_hist = result['tr_mse']
    vl_hist = result['vl_mse']
    epochs = np.arange(1, len(tr_hist) + 1)
    
    best_vl = min(vl_hist)
    best_epoch = np.argmin(vl_hist) + 1
    
    ax.plot(epochs, tr_hist, color='red', label='Train', linewidth=1.5 * font_scale, alpha=0.8)
    ax.plot(epochs, vl_hist, color='blue', linestyle='--', label='Val', linewidth=1.5 * font_scale, alpha=0.9)
    
    # Stella sul punto migliore
    ax.scatter(best_epoch, best_vl, color='green', marker='*', s=120 * font_scale, zorder=5, edgecolors='black')
    
    ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.4)
    
    if show_params_in_title:
        # Vecchio stile per plot singoli
        title = f"{title_prefix}\n{result['params']}"
    else:
        # Nuovo stile pulito per la griglia
        title = title_prefix
        
    ax.set_title(title, fontsize=11 * font_scale, fontweight='bold')
    
    # Rimuoviamo la legenda interna se occupa troppo spazio, o la facciamo piccola
    ax.legend(fontsize=8 * font_scale, loc='upper right')

def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def plot_final_assessment(results, avg_target_range, k_folds_final, relative_path="results/cup/final_assessment"):
    """
    Plotta le curve medie di MSE e MEE fianco a fianco.
    """
    
    # Crea directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")) 
    full_path = os.path.join(base_dir, relative_path)
    os.makedirs(full_path, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Estrai MSE (presente di sicuro)
    tr_mse = results.get('mean_tr_history', [])
    vl_mse = results.get('mean_vl_history', [])
    
    # Estrai MEE (se l'hai implementato in run_k_fold_cup)
    tr_mee = results.get('mean_tr_mee_history', [])
    vl_mee = results.get('mean_vl_mee_history', [])
    
    epochs = np.arange(1, len(tr_mse) + 1)
    
    # Creiamo una figura con 2 colonne
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- PLOT 1: MSE ---
    ax_mse = axes[0]
    ax_mse.plot(epochs, tr_mse, label='Mean Train MSE', color='red')
    ax_mse.plot(epochs, vl_mse, label='Mean Val MSE', color='blue', linestyle='--')
    ax_mse.set_title(f"Mean MSE (K={k_folds_final})")
    ax_mse.set_xlabel('Epochs')
    ax_mse.set_ylabel('MSE (Normalized)')
    ax_mse.set_yscale('log')
    ax_mse.grid(True, linestyle='--', alpha=0.7)
    ax_mse.legend()

    # --- PLOT 2: MEE ---
    ax_mee = axes[1]
    if len(tr_mee) > 0:
        ax_mee.plot(epochs, tr_mee, label='Mean Train MEE', color='orange')
        ax_mee.plot(epochs, vl_mee, label='Mean Val MEE', color='green', linestyle='--')
        ax_mee.set_title(f"Mean MEE (K={k_folds_final})")
        ax_mee.set_xlabel('Epochs')
        ax_mee.set_ylabel('MEE (Normalized)')
        # MEE di solito è lineare, ma puoi mettere log se serve
        ax_mee.grid(True, linestyle='--', alpha=0.7)
        ax_mee.legend()
    else:
        ax_mee.text(0.5, 0.5, "MEE History not found in results", 
                    ha='center', va='center', fontsize=12, color='gray')
        print("⚠️ Attenzione: curve MEE non trovate in 'results'. Aggiorna run_k_fold_cup per raccoglierle.")

    plt.tight_layout()
    
    filename = f"Final_KFold_MSE_MEE_{timestamp}.png"
    save_path = os.path.join(full_path, filename)
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Grafico doppio (MSE + MEE) salvato in: {save_path}")
    plt.close()


import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import textwrap



def _format_params_monk_wrapped(params):
    """Helper formattazione parametri specifici Monk"""
    ignored = ['verbose', 'validation', 'split', 'n_outputs', 'early_stopping', 'epsilon']
    items = []
    for k, v in params.items():
        if k in ignored: continue
        # Abbreviazioni
        key = k.replace("mini_batch_size", "bs").replace("learning_rate", "lr") \
               .replace("decay_factor", "d_fact").replace("decay_step", "d_step") \
               .replace("momentum", "mom").replace("alpha_mom", "a_mom") \
               .replace("lambdal2", "l2").replace("max_gradient_norm", "clip") \
               .replace("units_list", "units").replace("f_act_hidden", "act") \
               .replace("f_act_output", "out")
        
        val = str(v)
        if hasattr(v, '__name__'): val = v.__name__ 
        if isinstance(v, list): val = str(v).replace(" ", "")
        items.append(f"{key}={val}")

    full_str = ", ".join(items)
    return "\n".join(textwrap.wrap(full_str, width=50))


"""import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import textwrap

def plot_grid_search_analysis_monk(all_results, relative_path="results/monk/grid_search", top_k_details=6):
    
    Produce:
    1. SUMMARY GRID: Una griglia 4x3 con i Top 12 modelli (Overview).
    2. DETAILS FOLDER: Immagini singole ad alta risoluzione per i 'top_k_details' modelli,
       con grafici affiancati (MSE e Accuracy) e lista completa dei parametri.
    
    
    # Setup Cartelle
    base_dir = _ensure_dir(relative_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Cartella per i plot singoli dettagliati
    details_dir = os.path.join(base_dir, f"details_{timestamp}")
    os.makedirs(details_dir, exist_ok=True)
    
    # 1. Ordinamento: Best Validation Accuracy (Decrescente)
    sorted_results = sorted(all_results, key=lambda x: max(x['vl_accuracy']), reverse=True)
    
    # =========================================================
    # FASE A: SUMMARY GRID (TOP 12)
    # =========================================================
    top_12 = sorted_results[:12]
    rows, cols = 4, 3
    fig, axes = plt.subplots(rows, cols, figsize=(20, 24))
    axes = axes.flatten()
    
    plt.suptitle(f"Top 12 MONK Models (Sorted by Best Val Acc) - {timestamp}", fontsize=20, fontweight='bold', y=0.99)
    
    for i in range(rows * cols):
        ax = axes[i]
        if i < len(top_12):
            res = top_12[i]
            _plot_summary_subplot(ax, res, i+1)
        else:
            ax.axis('off')

    plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.65, wspace=0.2)
    summary_path = os.path.join(base_dir, f"GRID_MONK_Summary_{timestamp}.png")
    plt.savefig(summary_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ [Summary] Griglia salvata in: {summary_path}")

    # =========================================================
    # FASE B: INDIVIDUAL PLOTS (TOP K DETAILS)
    # =========================================================
    print(f"   ...Generazione plot dettagliati per i migliori {top_k_details} modelli...")
    
    for i in range(min(top_k_details, len(sorted_results))):
        res = sorted_results[i]
        rank = i + 1
        _plot_single_model_detail(res, rank, details_dir)

    print(f"✅ [Details] {min(top_k_details, len(sorted_results))} plot singoli salvati in: {details_dir}")
"""

def _plot_summary_subplot(ax, res, rank):
    """Disegna il singolo riquadro della griglia riassuntiva."""
    tr_mse = res['tr_mse']
    vl_acc = res['vl_accuracy']
    tr_acc = res['tr_accuracy']
    
    best_vl_acc = max(vl_acc)
    best_idx = np.argmax(vl_acc)
    tr_acc_at_best = tr_acc[best_idx]
    epochs = np.arange(1, len(tr_mse) + 1)
    
    # Plot MSE (Rosso)
    ax.plot(epochs, tr_mse, color='red', label='Train MSE', linewidth=1.5, alpha=0.8)
    
    # Titolo (Accuracy info)
    title = f"Rank #{rank} | Best VL Acc: {best_vl_acc*100:.1f}% | TR: {tr_acc_at_best*100:.1f}%"
    color = 'green' if best_vl_acc >= 1.0 else 'black'
    ax.set_title(title, fontsize=10, fontweight='bold', color=color)
    
    ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.4)
    
    # Parametri ridotti
    params_text = _format_params_monk_wrapped(res['params'])
    ax.text(0.5, -0.2, params_text, transform=ax.transAxes, ha='center', va='top', 
            fontsize=8, family='monospace', bbox=dict(boxstyle="round", fc="#f8f9fa", alpha=0.9))


def _plot_single_model_detail(res, rank, folder):
    """
    Crea un'immagine dettagliata con 2 grafici affiancati:
    SX: MSE (Loss) con dettagli numerici.
    DX: Accuracy con confronto Train/Val.
    SOTTO: Box completo parametri.
    """
    fig, (ax_mse, ax_acc) = plt.subplots(1, 2, figsize=(18, 7))
    
    params = res['params']
    tr_mse = res['tr_mse']
    tr_acc = res['tr_accuracy']
    vl_acc = res['vl_accuracy']
    epochs = np.arange(1, len(tr_mse) + 1)
    
    # --- 1. GRAFICO MSE (Sinistra) ---
    final_mse = tr_mse[-1]
    min_mse = min(tr_mse)
    min_mse_epoch = np.argmin(tr_mse) + 1
    
    ax_mse.plot(epochs, tr_mse, color='#d62728', linewidth=2, label='Training MSE')
    
    # Annotazione Minimo MSE
    ax_mse.scatter(min_mse_epoch, min_mse, color='black', s=50, zorder=5)
    ax_mse.annotate(f"Min: {min_mse:.2e}", xy=(min_mse_epoch, min_mse), xytext=(10, 10), 
                    textcoords='offset points', fontsize=9)

    ax_mse.set_title("Mean Squared Error (Loss) Analysis", fontsize=14, fontweight='bold')
    ax_mse.set_xlabel("Epoch")
    ax_mse.set_ylabel("MSE (log)")
    ax_mse.set_yscale('log')
    ax_mse.grid(True, which="both", ls="-", alpha=0.2)
    ax_mse.legend()
    
    # --- 2. GRAFICO ACCURACY (Destra) ---
    best_vl = max(vl_acc)
    best_epoch = np.argmax(vl_acc) + 1
    
    ax_acc.plot(epochs, tr_acc, color='#ff7f0e', label='Training Acc', linewidth=2)
    ax_acc.plot(epochs, vl_acc, color='#1f77b4', label='Validation Acc', linewidth=2, linestyle='--')
    
    # Evidenzia Best Validation
    ax_acc.scatter(best_epoch, best_vl, s=150, c='green', marker='*', edgecolors='black', zorder=10, 
                   label=f"Best Val: {best_vl*100:.1f}%")
    
    # Linea 100%
    ax_acc.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    
    ax_acc.set_title("Accuracy Trajectory", fontsize=14, fontweight='bold')
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim([-0.05, 1.05])
    ax_acc.legend(loc='lower right')
    ax_acc.grid(True, alpha=0.3)

    # --- 3. BOX INFORMAZIONI ---
    # Formattazione parametri pulita
    param_str = ",  ".join([f"{k}={v}" for k, v in params.items() if k not in ['verbose', 'validation']])
    # Spezza la stringa se troppo lunga
    wrapped_params = "\n".join(textwrap.wrap(param_str, width=120))
    
    info_text = (
        f"RANK #{rank}\n"
        f"Best Validation Accuracy: {best_vl*100:.2f}% (Epoch {best_epoch})\n"
        f"Final Training MSE: {final_mse:.6f}\n"
        f"--------------------------------------------------\n"
        f"CONFIGURAZIONE:\n{wrapped_params}"
    )
    
    plt.figtext(0.5, 0.02, info_text, ha="center", fontsize=11, family='monospace',
                bbox={"boxstyle": "round,pad=1", "facecolor": "#f0f0f0", "edgecolor": "#aaaaaa"})

    plt.subplots_adjust(bottom=0.25) # Lascia spazio per il testo sotto
    
    # Salvataggio
    safe_name = f"Rank_{rank:02d}_Acc_{int(best_vl*100)}.png"
    plt.savefig(os.path.join(folder, safe_name), dpi=150)
    plt.close()


def _format_params_monk_wrapped(params):
    """Helper formattazione breve per la griglia"""
    ignored = ['verbose', 'validation', 'split', 'n_outputs', 'early_stopping', 'epsilon']
    items = []
    for k, v in params.items():
        if k in ignored: continue
        # Abbreviazioni per risparmiare spazio nel plot piccolo
        key = k.replace("mini_batch_size", "bs").replace("learning_rate", "lr") \
               .replace("decay_factor", "d_f").replace("decay_step", "d_s") \
               .replace("momentum", "mom").replace("alpha_mom", "a_m") \
               .replace("lambdal2", "l2").replace("max_gradient_norm", "clip") \
               .replace("units_list", "units").replace("f_act_hidden", "act")
        
        val = str(v)
        if hasattr(v, '__name__'): val = v.__name__ 
        if isinstance(v, list): val = str(v).replace(" ", "")
        items.append(f"{key}={val}")

    full_str = ", ".join(items)
    return "\n".join(textwrap.wrap(full_str, width=45))

def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import textwrap

# =============================================================================
# 1. PLOT MONK SINGOLO (Aggiornato: No ora, Info Orizzontali)
# =============================================================================
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


# =============================================================================
# 2. GRID SEARCH ANALYSIS (Solo Top 6 + Info Orizzontali)
# =============================================================================
def plot_grid_search_analysis_monk(all_results, relative_path="results/monk/grid_search", top_k_details=6):
    
    base_dir = _ensure_dir(relative_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    details_dir = os.path.join(base_dir, f"details_{timestamp}")
    os.makedirs(details_dir, exist_ok=True)
    
    # Ordinamento
    sorted_results = sorted(all_results, key=lambda x: max(x['vl_accuracy']), reverse=True)
    
    # --- FASE A: SUMMARY GRID (SOLO TOP 6) ---
    top_6 = sorted_results[:6]  # Prendiamo solo i primi 6
    rows, cols = 2, 3           # Griglia 2x3
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 10)) # Dimensioni adattate
    axes = axes.flatten()
    
    plt.suptitle(f"Top 6 MONK Models (Sorted by Best Val Acc)", fontsize=18, fontweight='bold', y=0.98)
    
    for i in range(rows * cols):
        ax = axes[i]
        if i < len(top_6):
            res = top_6[i]
            _plot_summary_subplot(ax, res, i+1)
        else:
            ax.axis('off')

    plt.subplots_adjust(top=0.90, bottom=0.05, hspace=0.4, wspace=0.2)
    summary_path = os.path.join(base_dir, f"GRID_MONK_Top6_{timestamp}.png")
    plt.savefig(summary_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ [Summary] Griglia Top 6 salvata in: {summary_path}")

    # --- FASE B: INDIVIDUAL PLOTS (Info Orizzontali) ---
    print(f"   ...Generazione plot dettagliati per i Top {len(top_6)}...")
    for i in range(len(top_6)):
        res = top_6[i]
        _plot_single_model_detail_horizontal(res, i + 1, details_dir)


def _plot_summary_subplot(ax, res, rank):
    """Sottografico per la griglia riassuntiva"""
    tr_mse = res['tr_mse']
    vl_acc = res['vl_accuracy']
    best_vl_acc = max(vl_acc)
    epochs = np.arange(1, len(tr_mse) + 1)
    
    ax.plot(epochs, tr_mse, color='red', label='Train MSE')
    
    title = f"#{rank} | Best VL Acc: {best_vl_acc*100:.1f}%"
    color = 'green' if best_vl_acc >= 1.0 else 'black'
    ax.set_title(title, fontsize=10, fontweight='bold', color=color)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.4)
    
    # Parametri molto stringati per la preview piccola
    params_short = _format_params_short(res['params'])
    ax.text(0.5, -0.25, params_short, transform=ax.transAxes, ha='center', va='top', 
            fontsize=7, family='monospace', bbox=dict(boxstyle="round", fc="#f8f9fa", alpha=0.9))


def _plot_single_model_detail_horizontal(res, rank, folder):
    """Plot dettagliato con info sviluppate in orizzontale"""
    fig, (ax_mse, ax_acc) = plt.subplots(1, 2, figsize=(16, 7)) # Largo
    plt.subplots_adjust(bottom=0.20) # Spazio sotto
    
    params = res['params']
    tr_mse = res['tr_mse']
    tr_acc = res['tr_accuracy']
    vl_acc = res['vl_accuracy']
    epochs = np.arange(1, len(tr_mse) + 1)
    
    # 1. MSE
    ax_mse.plot(epochs, tr_mse, color='#d62728', linewidth=2, label='Training MSE')
    ax_mse.set_title("Loss (MSE)", fontsize=12, fontweight='bold')
    ax_mse.set_yscale('log')
    ax_mse.grid(True, alpha=0.3)
    ax_mse.legend()
    
    # 2. Accuracy
    best_vl = max(vl_acc)
    best_epoch = np.argmax(vl_acc) + 1
    ax_acc.plot(epochs, tr_acc, color='#d62728', label='Train Acc', linewidth=2)
    ax_acc.plot(epochs, vl_acc, color='#1f77b4', label='Val Acc', linewidth=2, linestyle='--')
    ax_acc.scatter(best_epoch, best_vl, s=150, c='green', marker='*', zorder=10)
    ax_acc.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax_acc.set_title(f"Accuracy (Best Val: {best_vl*100:.2f}%)", fontsize=12, fontweight='bold')
    ax_acc.set_ylim([-0.05, 1.05])
    ax_acc.legend(loc='lower right')
    ax_acc.grid(True, alpha=0.3)

    # 3. Info Orizzontali
    formatted_params = _format_params_horizontal(params)
    info_text = (
        f"RANK #{rank} | Best Val Acc: {best_vl*100:.2f}% (Ep {best_epoch}) | Final Train MSE: {tr_mse[-1]:.2e}\n"
        f"CONFIG: {formatted_params}"
    )
    
    plt.figtext(0.5, 0.03, info_text, ha="center", fontsize=10, family='monospace',
                bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f0f0f0", "edgecolor": "#aaaaaa"})
    
    safe_name = f"Rank_{rank:02d}_Horizontal.png"
    plt.savefig(os.path.join(folder, safe_name), dpi=150)
    plt.close()


# =============================================================================
# HELPER FORMATTAZIONE ORIZZONTALE
# =============================================================================
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