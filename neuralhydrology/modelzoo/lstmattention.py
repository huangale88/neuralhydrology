import inspect
from typing import Dict

import torch
from torch import nn

from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config
from neuralhydrology.modelzoo.inputlayer import InputLayer

class LSTMAttention(BaseModel):

    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = ['embedding_net', 'lstm', 'attention_W', 'attention_v', 'head']

    def __init__(self, cfg: Config):

        super().__init__(cfg=cfg)

        # retrieve the input layer
        self.embedding_net = InputLayer(cfg)

        # create the actual LSTM
        self.lstm = nn.LSTM(input_size=self.embedding_net.output_size, hidden_size=cfg.hidden_size)

        # linear layer to transform the hidden states from the LSTM output
        self.attention_W = nn.Linear(cfg.hidden_size, cfg.hidden_size)

        # linear layer to produce a scalar score for each hidden state
        self.attention_v = nn.Linear(cfg.hidden_size, 1)

        # add dropout between LSTM and head
        self.dropout = nn.Dropout(p=cfg.output_dropout)

        # retrieve the model head
        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)

        # initialize weights for the forget gate
        self._reset_parameters()

    def _reset_parameters(self):
        '''sets the forget gate bias of the LSTM'''
        if self.cfg.initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[self.cfg.hidden_size:2 * self.cfg.hidden_size] = self.cfg.initial_forget_bias
            self.lstm.bias_ih_l0.data[self.cfg.hidden_size:2 * self.cfg.hidden_size] = self.cfg.initial_forget_bias

    def forward(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:

        # possibly pass dynamic and static inputs through embedding layers, then concatenate them
        x_d = self.embedding_net(data, concatenate_output=True)    

        # run the actual LSTM
        lstm_output, (h_n, c_n) = self.lstm(input=x_d)

        # Apply the first linear layer
        attention_energies = self.attention_W(lstm_output)

        # Apply non-linearity
        attention_energies = torch.tanh(attention_energies)

        # Apply second linear layer to get scores
        attention_scores_raw = self.attention_v(attention_energies)

        # Apply softmax across the sequence dimension (dim=0)
        attention_weights = torch.softmax(attention_scores_raw, dim=0)

        # transpose lstm_output to [batch_size, sequence_length, hidden_size]
        lstm_output_transposed = lstm_output.transpose(0, 1)

        # squeeze the last dimension of attention_weights, then transpose again
        attention_weights = attention_weights.squeeze(2).transpose(0, 1)
        # add new dimension at position 1
        attention_weights = attention_weights.unsqueeze(1)

        # matrix multiplication
        context_vector = torch.bmm(attention_weights, lstm_output_transposed)

        # reshape to [batch_size, 1, n_hiddens]
        h_n = h_n.transpose(0, 1)
        c_n = c_n.transpose(0, 1)

        # prepare the prediction dictionary 
        pred = {'h_n': h_n, 'c_n': c_n}
        
        # add the final output as it's returned by the head to the prediction dict
        # (this will contain the 'y_hat')
        pred.update(self.head(self.dropout(context_vector)))

        return pred

