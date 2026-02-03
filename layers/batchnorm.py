import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# class MaskedBatchNorm(nn.BatchNorm2d):
#     def __init__(self, num_features, *args, **kwargs):
#         super(MaskedBatchNorm, self).__init__(num_features, *args, **kwargs)

#     def forward(self, x, p=None):
#         # Get the input dimension (number of channels/features)
#         in_dim = x.size(1)  # assuming (N, C, H, W) format for 2D data
        
#         # If p is None or zero, don't apply masking, use all features
#         if not p:
#             out_dim = self.num_features
#         else:
#             out_dim = int(np.ceil(self.num_features * p))
        
#         # Subsample weight and bias according to p
#         weight_red = self.weight[:out_dim] if self.weight is not None else None
#         bias_red = self.bias[:out_dim] if self.bias is not None else None
        
#         # Subsample running_mean and running_var
#         running_mean_red = self.running_mean[:out_dim]
#         running_var_red = self.running_var[:out_dim]
        
#         # Apply the batch normalization to the reduced number of features
#         return F.batch_norm(
#             x, # [:, :out_dim, :, :],   # Use only the masked channels 
#             running_mean_red,
#             running_var_red,
#             weight_red,
#             bias_red,
#             self.training or not self.track_running_stats,
#             self.momentum,
#             self.eps
#         )


# class MaskedBatchNorm(nn.BatchNorm2d):
#     def __init__(self, num_features, *args, **kwargs):
#         super(MaskedBatchNorm, self).__init__(num_features, affine=True, track_running_stats=False, *args, **kwargs)
        
#         self.num_features_max = num_features
        
#         self.bn = nn.ModuleList([
#             nn.BatchNorm2d(i) for i in [
#                 int(np.ceil(self.num_features_max * width)) for width in [round(x, 2) for x in np.arange(0.25, 1.01, 0.05)]
#             ]
#         ])
#         self.width_mult = None

#     def forward(self, x, p=None): 
#         # If p is None or zero, don't apply masking, use all features
#         if not p:
#             out_dim = self.num_features
#         else:
#             out_dim = int(np.ceil(self.num_features * p))
        

#         if self.training:
#             # if self.width_mult in [round(x, 2) for x in np.arange(0.25, 1.01, 0.05)]: # p >= 0.25 and p <= 1.0: 
#             # print("]]]]]]]]]]]]]] 1. >>>>", p, self.width_mult, out_dim)
#             idx = [round(x, 2) for x in np.arange(0.25, 1.01, 0.05)].index(p) 
            
#             # Debugging size mismatch
#             # print(f"Using Masked BN idx: {idx}, out_dim: {out_dim}, running_mean size: {self.bn[idx].running_mean.size(0)}")
#             # print("Before forward pass:")
#             # print(f"Running mean: {self.bn[idx].running_mean.mean()}")
#             # print(f"Running var: {self.bn[idx].running_var.mean()}")
            
#             y = F.batch_norm(
#                 x, 
#                 self.bn[idx].running_mean, #[:out_dim], 
#                 self.bn[idx].running_var, # [:out_dim], 
#                 self.weight[:out_dim], 
#                 self.bias[:out_dim], 
#                 self.training, 
#                 self.momentum, 
#                 self.eps
#             )
            
#             # y = self.bn[idx](x)
            
#             # print("After forward pass:")
#             # print(f"Running mean: {self.bn[idx].running_mean.mean()}")
#             # print(f"Running var: {self.bn[idx].running_var.mean()}")

#             # else:
#             #     # print(">>>>", p, self.width_mult, out_dim)
#             #     idx = [round(x, 2) for x in np.arange(0.25, 1.01, 0.05)].index(p) 
#             #     # print(self.running_mean, self.running_var)
#             #     # print(self.bn[idx].running_mean[:out_dim], self.bn[idx].running_var[:out_dim])
#             #     y = F.batch_norm(
#             #         x, 
#             #         self.running_mean, # [:out_dim], 
#             #         self.running_var, # [:out_dim], 
#             #         self.weight[:out_dim], 
#             #         self.bias[:out_dim], 
#             #         self.training or not self.track_running_stats, 
#             #         self.momentum, 
#             #         self.eps
#             #     )
#         else: # val
            
#             # Ensure running_mean and running_var are not None
#             # if self.running_mean is None or self.running_var is None:
#             #     raise ValueError("running_mean and running_var must be defined in evaluation mode.")
#             idx = [round(x, 2) for x in np.arange(0.25, 1.01, 0.05)].index(p) 
#             y = F.batch_norm(
#                 x, 
#                 self.bn[idx].running_mean, # [:out_dim], 
#                 self.bn[idx].running_var, # [:out_dim], 
#                 self.weight[:out_dim], 
#                 self.bias[:out_dim], 
#                 False, 
#                 self.momentum, 
#                 self.eps
#             )
            
#             # print(self.bn[idx].running_mean[:out_dim].mean(), self.bn[idx].running_var[:out_dim].mean())

#         return y
    

class MaskedBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, *args, **kwargs):
        super(MaskedBatchNorm, self).__init__(num_features, *args, **kwargs)
                
        self.bn = nn.ModuleList([
            nn.BatchNorm2d(i) for i in [
                int(np.ceil(num_features * width)) for width in [round(x, 2) for x in np.arange(0.25, 1.01, 0.05)]
            ]
        ])
        # self.width_mult = None

    def forward(self, x, p=1.0): 
        idx = [round(x, 2) for x in np.arange(0.25, 1.01, 0.05)].index(p)
        y = self.bn[idx](x)
        
        return y