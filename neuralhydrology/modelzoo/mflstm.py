import logging
<<<<<<< HEAD
from typing import Dict, Union
=======
from typing import Dict, Union, List
>>>>>>> 4003273a910247b100337ec576cd4d8da85ed64f

import torch
import torch.nn as nn

<<<<<<< HEAD
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)

class MFLSTM(BaseModel):
    """Multi-Frequency LSTM network, adapted for NeuralHydrology.

    This model processes inputs from different frequencies by optionally embedding them
    and then concatenating them along the sequence dimension before feeding them into a single LSTM.
    It then predicts the output based on the last 'predict_last_n' steps of the LSTM output.
=======
# Import necessary neuralhydrology components
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config
from neuralhydrology.datautils.utils import get_frequency_factor, sort_frequencies
from neuralhydrology.modelzoo.emb_net import EmbeddingNetwork


LOGGER = logging.getLogger(__name__)


class MFLSTM(BaseModel):
    """Multi-frequency LSTM network adapted for neuralhydrology.
>>>>>>> 4003273a910247b100337ec576cd4d8da85ed64f

    Parameters
    ----------
    cfg : Config
<<<<<<< HEAD
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
=======
        The run configuration.
    """
    
>>>>>>> 4003273a910247b100337ec576cd4d8da85ed64f
    module_parts = ['lstm', 'embedding_net', 'head']

    def __init__(self, cfg: Config):
        super().__init__(cfg=cfg)

<<<<<<< HEAD
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
=======
        self.hidden_size = cfg.hidden_size
        self.num_layers = 1 # Hardcoded default for num_layers
        LOGGER.info(f"MFLSTM: 'num_layers' hardcoded to {self.num_layers}.")

        self.dropout = torch.nn.Dropout(cfg.output_dropout)

        self.lstm = None        
        self.embedding_net = None 
        self.head = None        

        self.predict_last_n = 1
        LOGGER.info(f"MFLSTM: 'predict_last_n' hardcoded to {self.predict_last_n}.")

        if len(cfg.use_frequencies) < 1:
            raise ValueError("MFLSTM expects at least one input frequency in cfg.use_frequencies.")
        self._frequencies = sort_frequencies(cfg.use_frequencies)
        LOGGER.info(f"MFLSTM: Using frequencies: {self._frequencies}")

        self.use_dynamic_embeddings = True # Hardcoded to True.
        LOGGER.info(f"MFLSTM: 'use_dynamic_embeddings' hardcoded to {self.use_dynamic_embeddings}.")

        self.n_dynamic_channels_lstm = self.hidden_size 
        LOGGER.info(f"MFLSTM: 'n_dynamic_channels_lstm' hardcoded to {self.n_dynamic_channels_lstm} (equal to hidden_size).")
        
        self._init_modules() 
        self._reset_parameters()

    def _init_modules(self):
        """Initializes the LSTM, embedding networks, and prediction head."""
        
        input_size_lstm = 0

        if self.cfg.static_attributes:
            input_size_lstm += self.cfg.n_static_features
            LOGGER.info(f"MFLSTM: Adding {self.cfg.n_static_features} static features to LSTM input.")
        
        if self.cfg.hydroatlas_attributes:
            input_size_lstm += self.cfg.n_hydroatlas_attributes
            LOGGER.info(f"MFLSTM: Adding {self.cfg.n_hydroatlas_attributes} HydroATLAS features to LSTM input.")

        if self.cfg.use_basin_id_encoding:
            input_size_lstm += self.cfg.n_basin_id_encoding
            LOGGER.info(f"MFLSTM: Adding {self.cfg.n_basin_id_encoding} basin ID encoding features to LSTM input.")
        
        if self.cfg.evolving_attributes:
            input_size_lstm += self.cfg.n_evolving_attributes
            LOGGER.info(f"MFLSTM: Adding {self.cfg.n_evolving_attributes} evolving features to LSTM input.")

        if self.use_dynamic_embeddings:
            self.embedding_net = nn.ModuleDict()
            for freq in self._frequencies:
                if isinstance(self.cfg.dynamic_inputs, dict):
                    input_dim_freq = len(self.cfg.dynamic_inputs[freq])
                else: 
                    input_dim_freq = len(self.cfg.dynamic_inputs)
                
                LOGGER.info(f"MFLSTM: Creating embedding for frequency '{freq}' from {input_dim_freq} features to {self.n_dynamic_channels_lstm} dimensions.")
                self.embedding_net[freq] = EmbeddingNetwork(cfg=self.cfg, 
                                                            input_size=input_dim_freq, 
                                                            output_size=self.n_dynamic_channels_lstm, 
                                                            name='dynamics_embedding') 
            
            input_size_lstm += self.n_dynamic_channels_lstm * len(self._frequencies)
            LOGGER.info(f"MFLSTM: Total dynamic input to LSTM (after embeddings): {self.n_dynamic_channels_lstm * len(self._frequencies)}.")

        else: 
            input_size_lstm += self.cfg.n_dynamic_features
            LOGGER.info(f"MFLSTM: Using raw dynamic features. Total dynamic input to LSTM: {self.cfg.n_dynamic_features}.")

        LOGGER.info(f"MFLSTM: Final LSTM input size calculated as {input_size_lstm}.")

        self.lstm = nn.LSTM(input_size=input_size_lstm,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)

        self.head = nn.Linear(in_features=self.hidden_size, 
                              out_features=self.cfg.output_size)
        LOGGER.info(f"MFLSTM: Prediction head initialized with input {self.hidden_size} and output {self.cfg.output_size}.")


    def _reset_parameters(self):
        """Initializes/resets the model parameters."""
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                hidden_size = self.hidden_size
                if 'lstm.bias_ih_l0' in name:
                    self.lstm.__getattr__(name).data[hidden_size : 2 * hidden_size].fill_(1.0)
                elif 'lstm.bias_hh_l0' in name:
                    self.lstm.__getattr__(name).data[hidden_size : 2 * hidden_size].fill_(1.0)
                    
    def forward(self, data: Dict[str, torch.Tensor], h_n: torch.Tensor = None, c_n: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Performs a forward pass through the MFLSTM model.
        
        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary containing the input data. Expected keys depend on the configuration.
            - 'x_d_<freq>': Dynamic inputs for each frequency, shape (N, S, F_freq)
            - 'x_s': Combined static inputs, shape (N, F_s_total)
        h_n : torch.Tensor, optional
            Initial hidden state of the LSTM, shape (num_layers, N, hidden_size).
            If None, LSTM defaults to zero-initialization.
        c_n : torch.Tensor, optional
            Initial cell state of the LSTM, shape (num_layers, N, hidden_size).
            If None, LSTM defaults to zero-initialization.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing:
            - 'y_hat': Model predictions, shape (N, predict_last_n, output_size) or (N, output_size) if predict_last_n=1
            - 'h_n': Final hidden state of the LSTM
            - 'c_n': Final cell state of the LSTM
        """

        # Step 1: Process Dynamic Inputs for each frequency
        x_d_freq_list = [] # List to store processed dynamic inputs for concatenation
        
        # Determine the sequence length from the first dynamic input tensor
        # This assumes all frequencies have the same sequence length.
        seq_length = data[f'x_d_{self._frequencies[0]}'].shape[1] 

        for freq in self._frequencies:
            freq_data = data[f'x_d_{freq}'] # Shape: (N, S, F_freq)
            
            if self.use_dynamic_embeddings:
                # Pass through the embedding network specific to this frequency
                embedded_freq_data = self.embedding_net[freq](freq_data) # Output shape: (N, S, self.n_dynamic_channels_lstm)
                x_d_freq_list.append(embedded_freq_data)
            else:
                # If no embeddings, use raw dynamic data
                x_d_freq_list.append(freq_data)
        
        # Concatenate all processed dynamic features along the last dimension (feature dimension)
        # Resulting shape: (N, S, sum_of_processed_dynamic_features)
        # If using embeddings: (N, S, self.n_dynamic_channels_lstm * num_frequencies)
        x_d_concat = torch.cat(x_d_freq_list, dim=-1)

        # Step 2: Prepare Static Inputs
        # static_features from `data['x_s']` (shape: (N, F_s_total)) needs to be expanded
        # to match the sequence length of the dynamic features for concatenation.
        static_features = data['x_s'] 
        static_features_expanded = static_features.unsqueeze(1).repeat(1, seq_length, 1) # Shape: (N, S, F_s_total)

        # Step 3: Combine Static and Dynamic features for LSTM input
        # Concatenate along the feature dimension (dim=-1)
        x_lstm_input = torch.cat([static_features_expanded, x_d_concat], dim=-1) # Shape: (N, S, F_s_total + F_d_processed_total)

        # Step 4: Pass through the LSTM layer
        # If initial hidden states h_n, c_n are provided, use them.
        # Otherwise, the LSTM will implicitly initialize them with zeros.
        if h_n is not None and c_n is not None:
            lstm_output, (h_n, c_n) = self.lstm(x_lstm_input, (h_n, c_n))
        else:
            lstm_output, (h_n, c_n) = self.lstm(x_lstm_input)
        
        # Step 5: Apply dropout to the LSTM output
        lstm_output = self.dropout(lstm_output) # Output shape: (N, S, hidden_size)

        # Step 6: Select the last 'predict_last_n' time steps for prediction
        # Since predict_last_n is hardcoded to 1, this selects the last timestep's output.
        output_for_head = lstm_output[:, -self.predict_last_n:, :] # Shape: (N, self.predict_last_n, hidden_size)

        # Step 7: Pass through the prediction head (linear layer)
        predictions = self.head(output_for_head) # Output shape: (N, self.predict_last_n, cfg.output_size)

        # Step 8: Squeeze the 'predict_last_n' dimension if it's 1 for consistent output shape
        if self.predict_last_n == 1:
            predictions = predictions.squeeze(1) # Final shape: (N, output_size)

        # Step 9: Return predictions and final hidden/cell states
        return {"y_hat": predictions, "h_n": h_n, "c_n": c_n}
>>>>>>> 4003273a910247b100337ec576cd4d8da85ed64f
