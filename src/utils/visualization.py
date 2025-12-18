import matplotlib.pyplot as plt
import math
import numpy as np

def plot_errors(trainer_instance, tempo_di_training: float):
    """
    Riceve l'istanza del trainer e il tempo. 
    Calcola internamente l'asse delle epoche in base ai dati raccolti.
    """
    
    # --- 1. Recupero Dati ---
    mee_data = trainer_instance.mee_error_history
    mse_data = trainer_instance.mse_error_history
    
    # Creiamo l'asse X in base a quanti dati abbiamo effettivamente raccolto
    # (gestisce automaticamente l'Early Stopping)
    epoche_reali = np.arange(1, len(mee_data) + 1)

    # --- 2. Formattazione del Tempo ---
    minuti = math.floor(tempo_di_training / 60)
    secondi = tempo_di_training % 60
    tempo_str = f"{minuti}m {secondi:.2f}s"
    
    # --- 3. Creazione Figura (Subplots) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Analisi Training (Tempo Totale: {tempo_str})', fontsize=16, fontweight='bold')

    # --- Grafico MEE (Mean Euclidean Error) ---
    ax1.plot(epoche_reali, mee_data, color='#1f77b4', linewidth=2, label='MEE')
    ax1.set_title('MEE (Accuratezza per la CUP)', fontsize=13)
    ax1.set_xlabel('Epoca')
    ax1.set_ylabel('Errore Euclideo Medio')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # --- Grafico MSE (Mean Squared Error) ---
    ax2.plot(epoche_reali, mse_data, color='#d62728', linewidth=2, label='MSE', linestyle='--')
    ax2.set_title('MSE (Funzione di Costo)', fontsize=13)
    ax2.set_xlabel('Epoca')
    ax2.set_ylabel('Errore Quadratico Medio')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Lascia spazio per il suptitle
    plt.show()