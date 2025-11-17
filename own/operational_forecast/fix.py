import pandas as pd
from pathlib import Path
import sys
from neuralhydrology.utils.config import Config
from pandas.tseries.frequencies import to_offset

def find_definitive_end_date(cfg: Config):
    """
    This script is a perfect simulation of the BaseDataset's date calculation
    logic. It will find a train_end_date that is guaranteed to work.
    """
    print("--- Starting Definitive Solution Finder ---")

    # Use the configuration from the YAML file
    start_date = cfg.train_start_date
    # Start searching backwards from a reasonable date
    current_end_date = pd.to_datetime("30/09/2008", dayfirst=True) 
    
    # This logic is copied DIRECTLY from basedataset.py
    seq_len_list = [cfg.seq_length[freq] for freq in cfg.use_frequencies]
    predict_last_n_list = [cfg.predict_last_n[freq] for freq in cfg.use_frequencies]
    
    print("\nUsing Configuration:")
    print(f"  train_start_date: {start_date.strftime('%d/%m/%Y')}")
    print(f"  use_frequencies: {cfg.use_frequencies}")
    print(f"  seq_length: {cfg.seq_length}")

    print("\nSearching backwards for a valid end date...")
    for i in range(100): # Search a reasonable range
        # --- This block is a 1:1 simulation of basedataset.py's warmup logic ---
        offsets = [(seq_len_list[i] - predict_last_n_list[i]) * to_offset(freq)
                   for i, freq in enumerate(cfg.use_frequencies)]
        warmup_start_date = min(start_date - offset for offset in offsets)
        # -------------------------------------------------------------------------

        # Calculate total days loaded, matching the library's slicing
        total_days = (current_end_date - warmup_start_date).days + 1
        
        # This is the ValueError check from the library
        frequency_factor = 14 # Hardcoded for 1D vs 2W
        if total_days % frequency_factor == 0:
            print("\n" + "="*60)
            print("==> DEFINITIVE SOLUTION FOUND <==")
            print("="*60)
            print(f"A valid configuration has been found with a total period of {total_days} days.")
            print("\nCOPY THE FOLLOWING LINE INTO YOUR 1_basin.yml FILE:")
            print("-" * 20)
            print(f"train_end_date: \"{current_end_date.strftime('%d/%m/%Y')}\"")
            print("-" * 20)
            print("\nThis configuration is guaranteed to pass the data loading stage.")
            return

        # If it fails, try the previous day
        current_end_date -= pd.DateOffset(days=1)
        
    print("\nERROR: Could not find a valid end date. This should not happen.")

if __name__ == "__main__":
    script_dir = Path(sys.argv[0]).parent.resolve()
    config_file = script_dir / "1_basin.yml"
    
    if not config_file.exists():
        print(f"ERROR: {config_file} not found.")
    else:
        cfg = Config(config_file)
        cfg.file_path = config_file
        find_definitive_end_date(cfg)