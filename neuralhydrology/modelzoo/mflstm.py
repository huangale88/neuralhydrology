import logging
from typing import Dict, Union

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)

class MFLSTM(BaseModel):
    """Multi-Frequency LSTM network, adapted for NeuralHydrology.

    This model processes inputs from different frequencies by optionally embedding them
    and then concatenating them along the sequence dimension before feeding them into a single LSTM.
    It then predicts the output based on the last 'predict_last_n' steps of the LSTM output.

    Parameters
    ----------
    cfg : Config
        The run configuration, containing all necessary hyperparameters for the model.
        The configuration should include:
        - `hidden_size` (int): The number of hidden units in the LSTM.
        - `predict_last_n` (int): The number of last time steps to predict from the LSTM output.
        - `use_frequencies` (list of str): List of frequencies to be processed (e.g., ['daily', 'hourly']).
        - `output_dropout` (float): Dropout rate for the LSTM output.
        - `output_size` (int): The size of the model output (e.g., 1 for single-variate prediction).
        - `head` (str): The name of the output head to use (e.g., 'Regression', 'GMM').

    References
    ----------
    (No specific paper reference provided for MFLSTM, referencing the idea of multi-frequency processing.)
    """
    # Specify submodules of the model that can later be used for finetuning. Names must match class attributes.
    module_parts = ['lstm', 'embedding_net', 'head']

    def __init__(self, cfg: Config):
        super().__init__(cfg=cfg)

        self.lstm = None
        self.embedding_net = None
        self.dropout = None
        self.head = None

        self.hidden_size = cfg.hidden_size
        self.predict_last_n = cfg.predict_last_n
        self._frequencies = cfg.use_frequencies # This corresponds to original custom_freq_processing.keys()

        if not self._frequencies:
            raise ValueError("MFLSTM requires at least one frequency specified in `use_frequencies`.")

        # --- HARDCODED PARAMETERS ---
        # These parameters are hardcoded here because they are not recognized
        # as direct configuration arguments by the neuralhydrology Config object
        # in the current setup.
        # If you need to change these, modify this file directly.

        self.num_layers = 1 # Hardcoded: Number of LSTM layers.
        
        self.dynamic_embeddings = True # Hardcoded: If True, use separate linear layers to embed dynamic inputs per frequency.
        
        # Hardcoded: Input size for dynamic embedding layers per frequency.
        # These numbers must match the actual number of dynamic input features
        # specified for each frequency in your config.yml.
        self.dynamic_input_size_embedding = {
            '1D': 5,  # Number of '1D' dynamic_inputs from config.yml (prcp, srad, tmax, tmin, vp)
            '1h': 16  # Number of '1h' dynamic_inputs from config.yml (11 NLDAS + 5 Daymet)
        }
        
        self.n_channels_dynamic_embedding = 64 # Hardcoded: Output size of dynamic embedding layers.

        # --- END HARDCODED PARAMETERS ---

        # Determine input size for the LSTM
        # If dynamic_embeddings are used (now hardcoded to True), the effective input size per frequency
        # will be self.n_channels_dynamic_embedding.
        if self.dynamic_embeddings: # Use the hardcoded value
            input_features_per_freq = self.n_channels_dynamic_embedding
        else:
            # This 'else' block will not be hit if self.dynamic_embeddings is hardcoded to True
            # but is kept for completeness if you were to change dynamic_embeddings to False.
            if isinstance(cfg.dynamic_inputs, list):
                input_features_per_freq = len(cfg.dynamic_inputs)
            elif isinstance(cfg.dynamic_inputs, dict) and self._frequencies:
                # This assumes consistent feature counts across frequencies if no embeddings are used.
                input_features_per_freq = len(cfg.dynamic_inputs[self._frequencies[0]])
            else:
                input_features_per_freq = 0 # Fallback, should be avoided if inputs are expected


        # Calculate the total input size to the LSTM
        # Static attributes and basin ID encoding are concatenated *after* frequency-specific processing
        # and repeat across the concatenated sequence length.
        total_static_features = len(cfg.static_attributes + cfg.hydroatlas_attributes + cfg.evolving_attributes)
        if cfg.use_basin_id_encoding:
            total_static_features += cfg.number_of_basins
        if cfg.head.lower() == "umal": # UMAL head sometimes adds a feature to the input
            total_static_features += 1 # This is specific to how UMAL prepares its input in NH

        # The input_size_lstm will be the sum of features from one embedded/raw dynamic input plus static features.
        # This is because the MFLSTM concatenates frequencies along the sequence dimension, so the features for each
        # time step remain constant (input_features_per_freq + total_static_features).
        self.input_size_lstm = input_features_per_freq + total_static_features


        # Initialize embedding network if specified (now based on hardcoded self.dynamic_embeddings)
        self.embedding_net = self._get_embedding_net() # No need to pass cfg anymore for embeddings, use self.*

        # LSTM cell
        self.lstm = nn.LSTM(
            input_size=self.input_size_lstm,
            hidden_size=self.hidden_size,
            batch_first=True, # NeuralHydrology typically uses batch_first=True
            num_layers=self.num_layers # Uses the hardcoded num_layers
        )

        # Dropout layer for LSTM output
        self.dropout = torch.nn.Dropout(cfg.output_dropout)

        # Output head using NeuralHydrology's get_head function
        self.head = get_head(cfg, n_in=self.hidden_size, n_out=self.output_size)
        
        self._reset_parameters()

    # Modified _get_embedding_net to use hardcoded parameters
    def _get_embedding_net(self) -> Union[None, nn.ModuleDict]:
        """Initializes the embedding network for dynamic inputs per frequency."""
        embedding_net = None
        if self.dynamic_embeddings: # Use the hardcoded value
            embedding_net = nn.ModuleDict()
            for freq in self._frequencies: # Use self._frequencies
                # Determine input size for the current frequency's embedding layer
                # Using the hardcoded dynamic_input_size_embedding dictionary
                input_size = self.dynamic_input_size_embedding.get(freq, 0)
                
                if input_size == 0:
                    LOGGER.warning(f"Dynamic input size for frequency {freq} is 0 or not specified. "
                                   f"Embedding layer might be trivial or cause issues. Check hardcoded "
                                   f"self.dynamic_input_size_embedding in MFLSTM __init__.")

                # Linear layer to map from an input size to a predefined number of channels
                embedding_net[freq] = nn.Linear(
                    in_features=input_size, out_features=self.n_channels_dynamic_embedding # Use hardcoded value
                )
        return embedding_net

    def _reset_parameters(self):
        """Reset the parameters of the LSTM based on the initial_forget_bias config."""
        if self.cfg.initial_forget_bias is not None:
            hidden_size = self.hidden_size
            # Apply to all layers if num_layers > 1
            for i in range(self.num_layers): # Use the hardcoded num_layers
                # Check if bias_hh_l{i} exists, as it might not for all LSTM types/versions
                if hasattr(self.lstm, f'bias_hh_l{i}'):
                    getattr(self.lstm, f'bias_hh_l{i}').data[hidden_size:2 * hidden_size] = self.cfg.initial_forget_bias
                else:
                    LOGGER.warning(f"LSTM layer {i} does not have 'bias_hh_l{i}' attribute. Cannot apply initial_forget_bias.")

    def _prepare_inputs(self, data: Dict[str, torch.Tensor], freq: str) -> torch.Tensor:
        """Prepares and embeds (if configured) the dynamic inputs for a given frequency."""
        # --- HARDCODED CHANGE ---
        # Changed how dynamic inputs are accessed. NeuralHydrology typically nests
        # frequency-specific dynamic inputs under 'x_d' as a sub-dictionary.
        # So, data['x_d'] is a dict, and data['x_d'][freq] gets the tensor for that frequency.
        x = data['x_d'][freq] # Changed from data[f"x_d_{freq}"] to data['x_d'][freq]
        # --- END HARDCODED CHANGE ---
        
        if self.embedding_net: # In case we use embedding for the different frequencies (now based on hardcoded self.dynamic_embeddings)
            x = self.embedding_net[freq](x)
            
        return x

    def forward(self, data: Dict[str, torch.Tensor | Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the MFLSTM model.

        Parameters
        ----------
        data : Dict[str, torch.Tensor | Dict[str, torch.Tensor]]
            Input data for the forward pass. This dictionary is prepared by the DataLoader
            and contains tensors for dynamic features (e.g., 'x_d_daily'), static features ('x_s'),
            and optionally one-hot encoded basin IDs ('x_one_hot').

        Returns
        -------
        Dict[str, torch.Tensor]
            Model predictions, typically under the 'y_hat' key.
        """
        process_tensor = []
        # Process the dynamic inputs for each frequency
        for freq in self._frequencies:
            processed_freq_input = self._prepare_inputs(data, freq)
            process_tensor.append(processed_freq_input)

        # Concatenate dynamic inputs from all frequencies along the sequence length dimension (dim=1)
        x_lstm = torch.cat(process_tensor, dim=1)

        # Concatenate static attributes and basin ID encoding (if present) across the sequence length.
        # These are broadcast to match the concatenated sequence length.
        static_features_to_concat = []
        if 'x_s' in data: # Static attributes (including hydroatlas and evolving attributes)
            static_features_to_concat.append(data['x_s'])
        if 'x_one_hot' in data: # Basin ID one-hot encoding
            static_features_to_concat.append(data['x_one_hot'])
        
        if static_features_to_concat:
            # Combine all static-like features
            combined_static = torch.cat(static_features_to_concat, dim=-1)
            # Repeat static features across the concatenated sequence length
            # unsqueeze(1) adds a sequence dimension for broadcasting
            # repeat(1, x_lstm.shape[1], 1) repeats it across the sequence dimension
            x_lstm = torch.cat((x_lstm, combined_static.unsqueeze(1).repeat(1, x_lstm.shape[1], 1)), dim=2)
        
        # Initialize hidden and cell states for the LSTM
        # Initial states should be zeros and have the correct shape: (num_layers, batch_size, hidden_size)
        # and be on the same device as the input tensor.
        h0 = torch.zeros(
            self.num_layers, # Uses the hardcoded num_layers
            x_lstm.shape[0], # batch_size
            self.hidden_size,
            requires_grad=True,
            dtype=torch.float32,
            device=x_lstm.device,
        )
        c0 = torch.zeros(
            self.num_layers, # Uses the hardcoded num_layers
            x_lstm.shape[0], # batch_size
            self.hidden_size,
            requires_grad=True,
            dtype=torch.float32,
            device=x_lstm.device,
        )

        # Pass the concatenated input through the LSTM
        out, (hn_1, cn_1) = self.lstm(x_lstm, (h0, c0))
        
        # Take the output from the last 'predict_last_n' time steps
        out = out[:, -self.predict_last_n :, :]
        
        # Apply dropout
        out = self.dropout(out)
        
        # Pass through the NeuralHydrology head
        # The head expects input shape [batch_size, sequence_length, features]
        predictions = self.head(out)

        # Return predictions in the expected NeuralHydrology format
        # Since this model produces a single output for the combined frequencies,
        # it will just be 'y_hat'.
        return predictions