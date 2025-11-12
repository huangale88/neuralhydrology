import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ------------------------------------------------------------------

STREAMFLOW_INPUT_DIR = Path(r"D:\github\neuralhydrology\neuralhydrology\data\CAMELS_US\usgs_streamflow")
FORCING_INPUT_DIR = Path(r"D:\github\neuralhydrology\neuralhydrology\data\CAMELS_US\basin_mean_forcing\daymet")
OUTPUT_DIR = Path(r"D:\github\neuralhydrology\neuralhydrology\data\CAMELS_US\usgs_streamflow_preprocessed")

# --- SCRIPT LOGIC (No need to edit below this line) ---------------------------------

def get_basin_area_from_forcing(basin_id: str) -> int:
    """Finds the corresponding daymet forcing file and extracts the basin area."""
    file_path_list = list(FORCING_INPUT_DIR.glob(f'**/{basin_id}_*_forcing_leap.txt'))
    if not file_path_list:
        raise FileNotFoundError(f"Could not find matching forcing file for basin {basin_id} in {FORCING_INPUT_DIR}")
    file_path = file_path_list[0]
    with open(file_path, 'r') as fp:
        lines = fp.readlines()
        area_m2 = int(lines[2].strip())
    return area_m2

def preprocess_usgs_streamflow():
    """
    Loads daily USGS streamflow, normalizes it to mm/day using basin area, 
    creates aggregated 4W and 2W "blocky" target features, and saves the results.
    """
    print("--- Starting USGS Streamflow Pre-processing Script ---")
    
    if not STREAMFLOW_INPUT_DIR.is_dir():
        print(f"ERROR: Streamflow input directory not found at: {STREAMFLOW_INPUT_DIR}")
        return
    if not FORCING_INPUT_DIR.is_dir():
        print(f"ERROR: Forcing input directory (for area) not found at: {FORCING_INPUT_DIR}")
        return

    file_paths = list(STREAMFLOW_INPUT_DIR.glob('**/*_streamflow_qc.txt'))
    if not file_paths:
        print(f"ERROR: No '*_streamflow_qc.txt' files found in {STREAMFLOW_INPUT_DIR}")
        return
        
    print(f"Found {len(file_paths)} basin files to process.")

    for file_path in tqdm(file_paths, desc="Processing basins"):
        try:
            basin_id = file_path.name.split('_')[0]
            area = get_basin_area_from_forcing(basin_id)
            col_names = ['basin', 'Year', 'Mnth', 'Day', 'QObs_cfs', 'flag']
            df = pd.read_csv(
                file_path,
                sep='\s+',
                header=None,
                names=col_names,
                dtype={'basin': str}
            )
            df['date'] = pd.to_datetime(df['Year'].astype(str) + '-' + 
                                        df['Mnth'].astype(str) + '-' + 
                                        df['Day'].astype(str))
            df = df.set_index('date')
            
            # --- THIS IS THE CORRECTED LINE ---
            # Instead of inplace=True, we re-assign the result back to the column.
            df['QObs_cfs'] = df['QObs_cfs'].replace(-999.0, np.nan)
            
            # Normalize discharge from cubic feet per second (cfs) to mm/day
            df['QObs(mm/day)'] = df['QObs_cfs'] * 2446575.36 / area
            
            # Create the aggregated "blocky" target features
            df['QObs(mm/day)_4W'] = df['QObs(mm/day)'].resample('4W', label='right', closed='right').transform('mean')
            df['QObs(mm/day)_2W'] = df['QObs(mm/day)'].resample('2W', label='right', closed='right').transform('mean')
            
            df_to_save = df[['QObs(mm/day)', 'QObs(mm/day)_4W', 'QObs(mm/day)_2W']]
            
            relative_path = file_path.relative_to(STREAMFLOW_INPUT_DIR)
            output_path = OUTPUT_DIR / relative_path
            output_path = output_path.with_suffix('.csv')
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_to_save.to_csv(output_path)
            
        except Exception as e:
            print(f"\nERROR: Failed to process file {file_path.name}. Reason: {e}")

    print("--- Streamflow Pre-processing complete! ---")
    print(f"Processed files are saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    preprocess_usgs_streamflow()