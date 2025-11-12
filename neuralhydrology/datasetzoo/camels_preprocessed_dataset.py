import pandas as pd
from pathlib import Path

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
        Loads the pre-processed climate and streamflow data for a single basin and 
        merges them into a single DataFrame.
        """
        # Define the paths to our specific pre-processed data folders
        # self.cfg.data_dir is the root CAMELS US path from the config file
        climate_dir = self.cfg.data_dir / "basin_mean_forcing" / "daymet_preprocessed"
        streamflow_dir = self.cfg.data_dir / "usgs_streamflow_preprocessed"

        # --- Load Climate Data ---
        # Find the correct climate file for the given basin ID
        climate_files = list(climate_dir.glob(f"**/{basin}_*.csv"))
        if not climate_files:
            raise FileNotFoundError(f"No pre-processed climate file found for basin {basin} in {climate_dir}")
        
        # Load the data, making sure the 'date' column is the index
        df_climate = pd.read_csv(climate_files[0], index_col='date', parse_dates=True)

        # --- Load Streamflow Data ---
        # Find the correct streamflow file for the given basin ID
        streamflow_files = list(streamflow_dir.glob(f"**/{basin}_*.csv"))
        if not streamflow_files:
            raise FileNotFoundError(f"No pre-processed streamflow file found for basin {basin} in {streamflow_dir}")
        
        # Load the data
        df_streamflow = pd.read_csv(streamflow_files[0], index_col='date', parse_dates=True)

        # --- Merge DataFrames ---
        # Join the two dataframes on their shared date index.
        # This creates one wide DataFrame with all columns (climate and flow).
        df_merged = df_climate.join(df_streamflow)
        
        # This merged, daily-indexed DataFrame is the "source of truth" that the
        # parent BaseDataset class expects. It will handle resampling this DataFrame
        # to '2W' and '4W' internally.
        return df_merged

    def _load_attributes(self) -> pd.DataFrame:
        """
        Loads the static catchment attributes from the original CAMELS dataset.
        
        We can reuse the existing function from the `camelsus` module for this.
        """
        # This function needs the root CAMELS US directory to find the attributes folder.
        return camelsus.load_camels_us_attributes(self.cfg.data_dir, basins=self.basins)