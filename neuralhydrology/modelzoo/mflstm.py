import logging
from typing import Dict, Union, List

import torch
import torch.nn as nn

# Import necessary neuralhydrology components
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config
from neuralhydrology.datautils.utils import get_frequency_factor, sort_frequencies
from neuralhydrology.modelzoo.emb_net import EmbeddingNetwork


LOGGER = logging.getLogger(__name__)


class MFLSTM(BaseModel):
    """Multi-frequency LSTM network adapted for neuralhydrology.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """
    
    module_parts = ['lstm', 'embedding_net', 'head']

    def __init__(self, cfg: Config):
        super().__init__(cfg=cfg)

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