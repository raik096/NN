import itertools
from src.training.trainer.trainer_cup import TrainerCup
from src.training.trainer.trainer_monk import TrainerMonk
from src.utils import *
from src.training.validation.k_fold import run_k_fold_cup
from src.utils import *
import pandas as pd



class GridSearch:
    """
    Creata per gestire il caso esaustivo e caso intervalli
    """
    def __init__(self, **kwargs):
        # **kwargs contiene tutti gli argomenti passati come lista di liste

        self.params_to_explore = kwargs
        # Risultato da restituire con un getter
        self.best_config = None
        self.best_mse = float('inf')
        self.best_epoch = None
        self.history_mean_vl_history = []
        self.history_mean_tr_history = []
        self.combinations = self._generate_combinations()



    def _generate_combinations(self):
        """
        Metodo che scompone la lista di liste in chiavi e valori e fa
        il prodotto cartesiano esaustivo
        """
        # Ad esempio keys = ['units', 'lr']
        keys = self.params_to_explore.keys()
        # Ad esempio values = [ [[32], [64]], [0.1, 0.01] ]
        values = self.params_to_explore.values()

        all_configs = []
        for combination in itertools.product(*values):
            # Crea una combinazione di valori MANTENENDO l'ordine
            # quindi zippo
            comb = dict(zip(keys, combination))
            all_configs.append(comb)
        
        return all_configs
    

    def run_for_monk_holdout(self, x_train, d_train,vl_input, vl_targets):
        """
        Da completare...

        Metodo che si chiama dall'esterno, questo è il cuore 
        del grid search, 
        """

        self.best_accuracy = -1.0
        all_results_data = []
        
        for i, config_dict in enumerate(self.combinations):
            # Qui istanziamo il Trainer usando lo spacchettamento del dizionario **
            trainer = TrainerMonk(
                input_size=x_train.shape[1],
                **config_dict 
            )

            final_mse, val_accuracy = trainer.fit(
                tr_x = x_train,
                tr_d = d_train,
                vl_x = vl_input,
                vl_d = vl_targets
            )
            
            print(f"Config {i+1}/{len(self.combinations)} | Val Accuracy: {val_accuracy:.4f} | Tr MSE: {final_mse:.4f}")

            # Logica di massimizzazione dell'accuracy
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                self.best_config = config_dict
                print("   ⭐️ New Best Found!")

            all_results_data.append({
            'params': config_dict,
            'tr_mse': trainer.tr_mse_history,
            # 'vl_mse': trainer.vl_mse_history, # Se la tracci
            'tr_accuracy': trainer.tr_accuracy_history,
            'vl_accuracy': trainer.vl_accuracy_history,
            'ts_accuracy': trainer.ts_accuracy_history
            })
        

        plot_grid_search_analysis_monk2(all_results_data)

        return self.best_config, self.best_accuracy
    

    
    def run_for_cup_with_kfold(self,
                            x_full, d_full, k_folds,
                            d_min, d_max,
                            x_test_internal = None):
            """
            Model Selection robusta: K-Fold su ogni configurazione.
            """
            print(f"🚀 Inizio Grid Search con K-Fold (K={k_folds})...")
            all_results = [] # Lista che passeremo al plotter
            results_data = []

            for i, config_dict in enumerate(self.combinations):

                full_config = config_dict.copy()
                full_config['d_max'] = d_max
                full_config['d_min'] = d_min
                
                # Chiamiamo la funzione aggiornata sopra
                # quindi ogni configurazione * k (k_folds)
                stats = run_k_fold_cup(
                    x_full=x_full,
                    d_full=d_full,
                    k_folds=k_folds,
                    model_config=config_dict,
                    verbose=False,
                    x_test_internal = x_test_internal,
                )
                
                mean_mse_val = stats['vl_mean_mse']

                row = config_dict.copy()
                row['Mean_VL_MEE'] = stats['vl_mean_mee']
                row['Std_VL_MEE'] = stats['vl_std_mee'] # Scala anche la deviazione standard
                row['Mean_TR_MEE'] = stats['tr_mean_mee']
                results_data.append(row)

                print(f"Config {i+1}/{len(self.combinations)} | Mean MSE: {mean_mse_val:.5f}")

                # Aggiorna il best model
                if mean_mse_val < self.best_mse:
                    self.best_mse = mean_mse_val
                    self.best_config = config_dict
                    self.history_mean_vl_history = stats["all_vl_history_mee"]
                    self.history_mean_tr_history = stats["all_tr_history_mee"]
                    self.best_epoch = stats["epoch_reached"]
                    print(f"   ⭐️ New Best Found!")
                
                # --- COSTRUZIONE DATI PER IL PLOT ---
                result_entry = {
                    'params': config_dict,
                    'tr_metric': stats['all_tr_history_mee'], 
                    'vl_metric': stats['all_vl_history_mee']  
                }
                all_results.append(result_entry)

            
            # Crea la tabella
            df_results = pd.DataFrame(results_data)
            df_results = df_results.sort_values(by='Mean_VL_MEE')
            df_results.to_csv('results/grid_search_results.csv', index=False)
            print("\n Tabella risultati salvata in 'results/grid_search_results.csv'")
            print(df_results.head(5)) # Stampa i top 5 a video

            # Chiamata al plotter FUORI dal ciclo for
            print("Generazione grafici con valori denormalizzati...")

            plot_grid_analysis_cup2(all_results, top_k=5, save_path="results/cup/grid_kfold")

            return self.best_config, self.best_mse, self.best_epoch, self.history_mean_tr_history, self.history_mean_vl_history



"""    def run_for_cup(self, x_train, d_train, vl_input, vl_targets):
        
        #Metodo che si chiama dall'esterno, questo è il cuore
        #del grid search
        

        all_results = []

        for i, config_dict in enumerate(self.combinations):
            # Qui istanziamo il Trainer usando lo spacchettamento del dizionario **
            trainer = TrainerCup(
                input_size=x_train.shape[1],
                **config_dict
            )

            #  Per ogni combinazione chiama il fit
            current_mee_tr, current_mse_tr, current_mee_vl, current_mse_vl = trainer.fit(x_train, d_train,
                                                                                        vl_input, vl_targets)

            print(f"Config {i + 1}/{len(self.combinations)} | MSE in training: {current_mse_tr:.4f}")
            print(f"Config {i + 1}/{len(self.combinations)} | MSE in Validation: {current_mse_vl:.4f}")

            if current_mse_vl < self.best_mse:
                self.best_mse = current_mse_vl
                self.best_config = config_dict

            result_entry = {
                'params': config_dict,
                'tr_mse': trainer.tr_mse_history, # Lista MSE train per ogni epoca
                'vl_mse': trainer.vl_mse_history, # Lista MSE val per ogni epoca
                # Se volessi plottare anche il MEE in futuro, puoi salvarlo qui:
                # 'tr_mee': trainer.tr_mee_history,
                # 'vl_mee': trainer.vl_mee_history
            }
            all_results.append(result_entry)

        plot_grid_analysis(
                    all_results, 
                    top_k_individual=5,              # Salva i dettagli delle migliori 5 configurazioni
                    relative_path="results/cup/grid_search"
                )

        return self.best_config, self.best_mse"""