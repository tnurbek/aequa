import numpy as np 
import torch.nn as nn 

__all__ = []


def masked_conv_forward(layer, x, p=None): 
    in_dim = x.size(1)  # second dimension is input dimension 
    if not p:  # i.e., don't apply masking 
        out_dim = layer.out_channels 
    else:
        out_dim = int(np.ceil(layer.out_channels * p)) 

    # subsampled weights and bias
    weights_red = layer.weight[:out_dim, :in_dim]
    bias_red = layer.bias[:out_dim] if layer.bias is not None else None
    return layer._conv_forward(x, weights_red, bias_red)


class MaskedConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super(MaskedConv1d, self).__init__(*args, **kwargs)
    
    def forward(self, x, p=None):
        return masked_conv_forward(self, x, p) 
    

class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
    
    def forward(self, x, p=None):
        return masked_conv_forward(self, x, p) 
    

class MaskedConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super(MaskedConv3d, self).__init__(*args, **kwargs)
    
    def forward(self, x, p=None):
        return masked_conv_forward(self, x, p) 