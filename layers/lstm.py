import torch
from torch import nn 
import numpy as np 
from torch.nn.utils.rnn import pad_packed_sequence


__all__ = ['MaskedLSTM', 'MaskedLSTMV2']


class MaskedLSTM(nn.LSTM):
    def __init__(self, *args, **kwargs):
        super(MaskedLSTM, self).__init__(*args, **kwargs)
        if self.num_layers > 1:
            raise ValueError("Do not use multiple lstm layers as one lstm cell, stack them instead.")
        self.full_hidden_size = self.hidden_size 
        
    def forward(self, x, hx=None, p=None):
        # in_dim = self.input_size
        in_dim = x.size(2) 
        self.input_size = x.size(2)
        if not p:
            self.hidden_size = self.full_hidden_size
        else:
            self.hidden_size = int(np.ceil(self.full_hidden_size * p))
        
        ind_out = range(self.hidden_size)
        ind_out_ext = np.concatenate([np.array(ind_out) + i * self.full_hidden_size for i in range(4)], axis=0)
        
        # print(x.shape, self.input_size, in_dim, self.hidden_size)
        
        self._flat_weights = []
        
        # subsampled weights and biases 
        for i, p_name in enumerate(self._flat_weights_names):
            p_parts = p_name.split('_')
            p_type, p_connection = p_parts[:2]
            
            if p_type == 'weight':
                weight = getattr(self, p_name)
                if p_connection == 'ih':
                    weight_red = weight[ind_out_ext][:, :in_dim]
                elif p_connection == 'hh':
                    weight_red = weight[ind_out_ext][:, ind_out]
                else: # hr 
                    weight_red = weight[:, ind_out]
                self._flat_weights.append(weight_red.contiguous())
            elif p_type == 'bias':
                bias = getattr(self, p_name)
                bias_red = bias[ind_out_ext] if bias is not None else None
                self._flat_weights.append(bias_red.contiguous() if bias_red is not None else None)
            else:
                raise ValueError(f"Unknown LSTM layer type: {p_name}")
        
        self.flatten_parameters()
        if hx is not None:
            hx = (hx[0].contiguous(), hx[1].contiguous())
            
        return super(MaskedLSTM, self).forward(x, hx)
    
