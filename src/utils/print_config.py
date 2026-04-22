def print_config(config, score=None, metric_name="MSE"):
    """Stampa helper per visualizzare bene i risultati"""
    print("\n" + "═"*60)
    print(f" 🏆  BEST CONFIGURATION FOUND")
    print("═"*60)
    if score is not None:
        print(f" 📊  BEST {metric_name:<20}: {score:.6f}")
        print("─"*60)
    for key in sorted(config.keys()):
        val = config[key]
        val_str = val.__name__ if hasattr(val, '__name__') else str(val)
        print(f" • {key:<25} :  {val_str}")
    print("═"*60 + "\n")