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
    module_parts = ['embedding_net', 'lstm', 'head']

    def __init__(self, cfg: Config):

        super().__init__(cfg=cfg)

        # retrieve the input layer
        self.embedding_net = InputLayer(cfg)

        # create the actual GRU
        self.lstm = nn.LSTM(input_size=self.embedding_net.output_size, hidden_size=cfg.hidden_size)

        # add dropout between GRU and head
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

        # reshape to [batch_size, 1, n_hiddens]
        h_n = h_n.transpose(0, 1)
        c_n = c_n.transpose(0, 1)

        # prepare the prediction dictionary 
        pred = {'h_n': h_n, 'c_n': c_n}
        
        # add the final output as it's returned by the head to the prediction dict
        # (this will contain the 'y_hat')
        pred.update(self.head(self.dropout(lstm_output.transpose(0, 1))))

        return pred

