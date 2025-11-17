import pandas as pd
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ------------------------------------------------------------------

# 1. DEFINE BASE DIRECTORIES
# Use raw strings (r"...") for Windows paths to avoid issues with backslashes.
INPUT_DIR = Path(r"D:\github\neuralhydrology\neuralhydrology\data\CAMELS_US\basin_mean_forcing\daymet")
OUTPUT_DIR = Path(r"D:\github\neuralhydrology\neuralhydrology\data\CAMELS_US\basin_mean_forcing\daymet_preprocessed")

# 2. DEFINE WHICH VARIABLES TO AGGREGATE AND HOW
# This dictionary controls the processing.
# Key: The exact column name from the input file.
# Value: The aggregation method to use ('sum' for cumulative, 'mean' for averages).
VARS_TO_AGGREGATE = {
    'prcp(mm/day)': 'sum',
    'srad(W/m2)': 'mean',
    'swe(mm)': 'mean',
    'tmax(C)': 'mean',
    'tmin(C)': 'mean',
    'vp(Pa)': 'mean'
}

# --- SCRIPT LOGIC (No need to edit below this line) ---------------------------------

def preprocess_camels_daymet_data():
    """
    Loads daily CAMELS Daymet data, creates aggregated 4W (four-weekly) and 
    2W (bi-weekly) "blocky" features, and saves the result to a new directory 
    while maintaining the original folder structure.
    """
    print("--- Starting Pre-processing Script ---")
    
    # Check if the input directory exists
    if not INPUT_DIR.is_dir():
        print(f"ERROR: Input directory not found at: {INPUT_DIR}")
        return

    # Find all the forcing files recursively in the input directory
    # The glob pattern '**/*' searches through all subdirectories
    file_paths = list(INPUT_DIR.glob('**/*_forcing_leap.txt'))

    if not file_paths:
        print(f"ERROR: No '*_forcing_leap.txt' files found in {INPUT_DIR}")
        return
        
    print(f"Found {len(file_paths)} basin files to process.")

    # Loop through each file with a progress bar
    for file_path in tqdm(file_paths, desc="Processing basins"):
        try:
            # Load the data using pandas
            # sep='\s+' handles one or more whitespace characters as a separator
            # skiprows=3 skips the first 3 lines of metadata (lat, elev, area)
            # header=0 tells pandas that the next line (the 4th line) is the header
            df = pd.read_csv(
                file_path,
                sep='\s+',
                skiprows=3,
                header=0
            )

            # Create a proper datetime index, which is essential for resampling
            df['date'] = pd.to_datetime(df['Year'].astype(str) + '-' + 
                                        df['Mnth'].astype(str) + '-' + 
                                        df['Day'].astype(str))
            df = df.set_index('date')
            
            # Clean up by dropping the original date-part columns
            df = df.drop(columns=['Year', 'Mnth', 'Day', 'Hr'])

            # --- This is the core aggregation logic ---
            for var, method in VARS_TO_AGGREGATE.items():
                if var in df.columns:
                    # Create the '4W' (four-weekly) blocky feature
                    # ADDED: label='right' and closed='right' for explicit, consistent alignment
                    df[f'{var}_4W'] = df[var].resample('28D').transform(method) 
                    
                    # Create the '2W' (bi-weekly) blocky feature
                    df[f'{var}_2W'] = df[var].resample('14D').transform(method)
                else:
                    print(f"\nWarning: Column '{var}' not found in {file_path.name}")

            # --- Determine the output path ---
            # Get the relative path of the file from the input directory
            # e.g., '01\01054200_lump_cida_forcing_leap.txt'
            relative_path = file_path.relative_to(INPUT_DIR)

            # Create the full output path by joining it with the output directory
            output_path = OUTPUT_DIR / relative_path

            # It's good practice to save processed files in a more standard format like CSV
            # .with_suffix('.csv') changes the file extension from .txt to .csv
            output_path = output_path.with_suffix('.csv')
            
            # Create the parent directory (e.g., 'daymet_preprocessed\01') if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the processed DataFrame to a CSV file
            df.to_csv(output_path)
            
        except Exception as e:
            print(f"\nERROR: Failed to process file {file_path.name}. Reason: {e}")

    print("--- Pre-processing complete! ---")
    print(f"Processed files are saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    preprocess_camels_daymet_data()