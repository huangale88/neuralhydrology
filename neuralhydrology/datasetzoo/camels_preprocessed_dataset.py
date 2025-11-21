import pandas as pd
from pathlib import Path
import numpy as np

# Important: We need to import the BaseDataset we are inheriting from
from neuralhydrology.datasetzoo.basedataset import BaseDataset

# We will also reuse the original CAMELS attribute loader to avoid rewriting code
from neuralhydrology.datasetzoo import camelsus 
from neuralhydrology.utils.config import Config


class CamelsDaymetPreprocessed(BaseDataset):
    """Dataset class for pre-processed daily CAMELS US data.

    This class is designed to load the daily climate and streamflow data that has been
    pre-processed to include aggregated 'blocky' features for bi-weekly ('2W') and 
    four-weekly ('4W') timescales.
    
    It expects the `data_dir` in the config file to point to the root of the CAMELS_US
    dataset. It will then look for the pre-processed data in specific sub-folders:
    - `data_dir/basin_mean_forcing/daymet_preprocessed/`
    - `data_dir/usgs_streamflow_preprocessed/`
    """

    def __init__(self, *args, **kwargs):
        # This __init__ is simple, we just pass all arguments to the parent class
        super(CamelsDaymetPreprocessed, self).__init__(*args, **kwargs)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """
        Loads data for a single basin, switching between historical and operational sources.
        """
        # Use getattr to safely check for the 'run_mode' flag.
        # If the flag is not set, it defaults to 'historical'.
        run_mode = getattr(self.cfg, 'run_mode', 'historical')

        if run_mode == 'operational':
            # --- OPERATIONAL FORECAST MODE ---
            # This block is now triggered ONLY when you explicitly set the flag.
            
            # Use the correct path to your forecast files.
            forecast_dir = Path(r"D:\github\neuralhydrology\neuralhydrology\data\forecast\forecasts_processed")
            forecast_file = forecast_dir / f"{basin}_operational_forecast.csv"
            
            print(f"\n[CONFIRMATION] EXPLICIT OPERATIONAL MODE. Loading forecast data from: {forecast_file}\n")
            
            if not forecast_file.is_file():
                raise FileNotFoundError(
                    f"Operational forecast file not found: {forecast_file}\n"
                    f"Please ensure your formatted forecast CSV is in this location."
                )
            
            df_merged = pd.read_csv(forecast_file, index_col='date', parse_dates=True)
            
            # Add empty placeholder columns for the target variables.
            for target in self.cfg.target_variables:
                if target not in df_merged.columns:
                    df_merged[target] = np.nan 

        else:
            # --- HISTORICAL MODE (for train, validation, AND test) ---
            # This block is now the default for all standard runs.
            print(f"\n[CONFIRMATION] HISTORICAL MODE. Loading data for period '{self.period}' for basin {basin}...\n")
            
            climate_dir = self.cfg.data_dir / "basin_mean_forcing" / "daymet_preprocessed"
            streamflow_dir = self.cfg.data_dir / "usgs_streamflow_preprocessed"

            climate_files = list(climate_dir.glob(f"**/{basin}_*.csv"))
            df_climate = pd.read_csv(climate_files[0], index_col='date', parse_dates=True)

            streamflow_files = list(streamflow_dir.glob(f"**/{basin}_*.csv"))
            df_streamflow = pd.read_csv(streamflow_files[0], index_col='date', parse_dates=True)
            
            df_merged = df_climate.join(df_streamflow)
        
        return df_merged

    def _load_attributes(self) -> pd.DataFrame:
        """
        Loads the static catchment attributes from the original CAMELS dataset.
        
        We can reuse the existing function from the `camelsus` module for this.
        """
        # This function needs the root CAMELS US directory to find the attributes folder.
        return camelsus.load_camels_us_attributes(self.cfg.data_dir, basins=self.basins)