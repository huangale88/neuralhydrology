import pandas as pd
from pathlib import Path
import numpy as np
from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.datasetzoo import camelsus 
from neuralhydrology.utils.config import Config

class CamelsDaymetPreprocessed(BaseDataset):
    """
    Dataset class with a flexible 'run_mode' to switch between data sources.
    
    - 'historical' mode: Loads standard training/validation/testing data.
    - 'operational' mode: Loads a specific forecast file. This mode is now
      date-aware and can handle single forecasts or rolling evaluations.
    """

    def __init__(self, *args, **kwargs):
        super(CamelsDaymetPreprocessed, self).__init__(*args, **kwargs)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """
        Loads data for a single basin, switching between historical and operational sources.
        """
        run_mode = getattr(self.cfg, 'run_mode', 'historical')

        if run_mode == 'operational':
            # --- OPERATIONAL FORECAST MODE (NOW DATE-AWARE) ---
            
            # Use getattr to safely get the path to the root forecast directory.
            # This makes the loader more flexible.
            forecast_root_dir = getattr(self.cfg, 'forecast_dir', None)
            if forecast_root_dir is None:
                raise ValueError("For 'operational' run_mode, you must specify 'forecast_dir' in the config.")
            
            # ##############################################################
            # ################      THIS IS THE KEY CHANGE      ################
            # ##############################################################
            # The BaseDataset makes the start date of the current period available.
            # For an operational run, this corresponds to the start of the forecast.
            start_date_str = self.start_and_end_dates[basin]['start_dates'][0].strftime('%Y-%m-%d')
            
            # Construct the path to the specific, date-stamped forecast folder.
            forecast_dir = Path(forecast_root_dir) / start_date_str
            # ##############################################################

            forecast_file = forecast_dir / f"{basin}_operational_forecast.csv"
            
            print(f"\n[CONFIRMATION] OPERATIONAL MODE: Loading forecast data from: {forecast_file}\n")
            
            if not forecast_file.is_file():
                raise FileNotFoundError(f"Operational forecast file not found: {forecast_file}")
            
            df_merged = pd.read_csv(forecast_file, index_col='date', parse_dates=True)
            
            # Add placeholder columns for the target variables.
            for target in self.cfg.target_variables:
                if target not in df_merged.columns:
                    df_merged[target] = np.nan

        else:
            # --- HISTORICAL MODE (Unchanged) ---
            print(f"\n[CONFIRMATION] HISTORICAL MODE: Loading data for period '{self.period}' for basin {basin}...\n")
            
            climate_dir = self.cfg.data_dir / "basin_mean_forcing" / "daymet_preprocessed"
            streamflow_dir = self.cfg.data_dir / "usgs_streamflow_preprocessed"

            climate_files = list(climate_dir.glob(f"**/{basin}_*.csv"))
            df_climate = pd.read_csv(climate_files[0], index_col='date', parse_dates=True)

            streamflow_files = list(streamflow_dir.glob(f"**/{basin}_*.csv"))
            df_streamflow = pd.read_csv(streamflow_files[0], index_col='date', parse_dates=True)
            
            df_merged = df_climate.join(df_streamflow)
        
        return df_merged

    def _load_attributes(self) -> pd.DataFrame:
        return camelsus.load_camels_us_attributes(self.cfg.data_dir, basins=self.basins)