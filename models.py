import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.linear import MaskedLinear 
from layers.cnn import MaskedConv2d
from layers.batchnorm import MaskedBatchNorm
from layers.lstm import MaskedLSTM

import torchvision.models as models
import numpy as np
import math

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Model(nn.Module):
    """
    Small FC network for Synthetic dataset
    """
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = MaskedLinear(5, 32)
        self.fc2 = MaskedLinear(32, 64)
        self.fc3 = MaskedLinear(64, 128)
        self.fc4 = MaskedLinear(128, 32)
        self.fc5 = MaskedLinear(32, 4)
        # 0.0142M parameters
    
    def forward(self, x, p=None):
        x = F.relu(self.fc1(x, p=p))
        x = F.relu(self.fc2(x, p=p))
        x = F.relu(self.fc3(x, p=p))
        x = F.relu(self.fc4(x, p=p))
        x = self.fc5(x, p=1.)
        return x


class SmallModel(nn.Module):
    def __init__(self):
        super(SmallModel, self).__init__()
        self.fc1 = MaskedLinear(5, 16)
        self.fc2 = MaskedLinear(16, 32)
        self.fc3 = MaskedLinear(32, 64)
        self.fc4 = MaskedLinear(64, 16)
        self.fc5 = MaskedLinear(16, 4)
        # 0.0142M parameters
    
    def forward(self, x, p=None):
        x = F.relu(self.fc1(x, p=p))
        x = F.relu(self.fc2(x, p=p))
        x = F.relu(self.fc3(x, p=p))
        x = F.relu(self.fc4(x, p=p))
        x = self.fc5(x, p=1.)
        return x


class ConvModel(nn.Module):
    """Model for MNIST Dataset"""
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv1 = MaskedConv2d(1, 10, kernel_size=5)  # 1 32
        self.conv2 = MaskedConv2d(10, 20, kernel_size=5)  # 32 64
        self.fc1 = MaskedLinear(500, 64)  # 4096 512
        self.fc2 = MaskedLinear(64, 10)  # 512 10
        # 0.0362M parameters - 10, 20, 500, 64          <<<< 57 times less parameters 
        # 2.0551M parameters - 32, 64, 4096, 512 
    
    def forward(self, x, p=None):
        b = x.shape[0]
        x = F.relu(F.max_pool2d(self.conv1(x, p=p), 2))
        x = F.relu(F.max_pool2d(self.conv2(x, p=p), 2))
        x = x.view(b, -1)
        x = F.relu(self.fc1(x, p=p))
        x = self.fc2(x, p=1.)
        return x
        

class SmallConvModel(nn.Module):
    """Model for MNIST Dataset"""
    def __init__(self):
        super(SmallConvModel, self).__init__()
        self.conv1 = MaskedConv2d(1, 5, kernel_size=5)  # 1 32
        self.fc1 = MaskedLinear(980, 64)  # 4096 512
        self.fc2 = MaskedLinear(64, 10)  # 512 10
    
    def forward(self, x, p=None):
        b = x.shape[0]
        x = F.relu(F.max_pool2d(self.conv1(x, p=p), 2))
        x = x.view(b, -1)
        x = F.relu(self.fc1(x, p=p))
        x = self.fc2(x, p=1.)
        return x

class ConvCIFAR10Model(nn.Module):
    """Model for MNIST Dataset"""
    def __init__(self, n_classes=10):
        super(ConvCIFAR10Model, self).__init__()
        self.conv1 = MaskedConv2d(3, 32, kernel_size=5)  # 3 32
        self.conv2 = MaskedConv2d(32, 64, kernel_size=5)  # 32 64
        self.fc1 = MaskedLinear(4096, 512)  # 4096 512
        self.fc2 = MaskedLinear(512, n_classes)  # 512 10
    
    def forward(self, x, p=None):
        b = x.shape[0]
        x = F.relu(F.max_pool2d(self.conv1(x, p=p), 2))
        x = F.relu(F.max_pool2d(self.conv2(x, p=p), 2))
        x = x.view(b, -1)
        x = F.relu(self.fc1(x, p=p))
        x = self.fc2(x, p=1.)
        return x


# Old way 
class SyntheticModel(nn.Module):
    def __init__(self):
        super(SyntheticModel, self).__init__()
        self.fc1 = nn.Linear(5, 32) 
        self.fc2 = nn.Linear(32, 64) 
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 32) 
        self.fc5 = nn.Linear(32, 4) 
        self.relu = nn.ReLU() 
        
        # Initialize score masks as trainable parameters
        self.mask_fc2 = torch.ones(32)
        self.mask_fc3 = torch.ones(64)
        self.mask_fc4 = torch.ones(128)
        self.mask_fc5 = torch.ones(32)
        self.mask_fc5 = torch.ones(32)

    def forward(self, x): 
        x = self.relu(self.fc1(x)) 
        
        # Apply mask to the weights 
        masked_weights_fc2 = self.fc2.weight * self.mask_fc2.view(1, -1) 
        x = F.relu(F.linear(x, masked_weights_fc2, self.fc2.bias)) 
        
        masked_weights_fc3 = self.fc3.weight * self.mask_fc3.view(1, -1) 
        x = F.relu(F.linear(x, masked_weights_fc3, self.fc3.bias)) 
        
        masked_weights_fc4 = self.fc4.weight * self.mask_fc4.view(1, -1)
        x = F.relu(F.linear(x, masked_weights_fc4, self.fc4.bias)) 
        
        masked_weights_fc5 = self.fc5.weight * self.mask_fc5.view(1, -1)
        x = F.linear(x, masked_weights_fc5, self.fc5.bias) 

        return x 
    
    # initial call 
    def set_masks(self, mask_fc2, mask_fc3, mask_fc4, mask_fc5): 
        # Sample from Bernoulli distribution to generate binary masks 
        self.mask_fc2 = mask_fc2 
        self.mask_fc3 = mask_fc3 
        self.mask_fc4 = mask_fc4 
        self.mask_fc5 = mask_fc5 

    def get_masks(self):
        return self.mask_fc2, self.mask_fc3, self.mask_fc4, self.mask_fc5 
    
    
    
# ResNet implementation with GroupNorm
# Fetched from "IAFL"


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = MaskedConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = MaskedBatchNorm(planes)
        self.conv2 = MaskedConv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1)
        self.bn2 = MaskedBatchNorm(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                MaskedConv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride),
                # nn.BatchNorm2d(self.expansion*planes)
                MaskedBatchNorm(self.expansion*planes)
            )

    def forward(self, x, p=1.0):
        out = F.relu(self.bn1(self.conv1(x, p=p), p=p))
        # out = F.relu(self.bn1(self.conv1(x, p=p)))
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out, p=p), p=p)
        # out = self.bn2(self.conv2(out, p=p))
        # out = self.bn2(self.conv2(out))
        
        # Handle shortcut manually if it exists 
        shortcut_out = x
        if len(self.shortcut) > 0:
            # shortcut_out = self.shortcut(shortcut_out)
            for layer in self.shortcut:
                shortcut_out = layer(shortcut_out, p=p)  # Pass `p` to each shortcut layer || , p=p
        out += shortcut_out  # Add the shortcut 
        
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = MaskedConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = MaskedBatchNorm(planes)
        self.conv2 = MaskedConv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = MaskedBatchNorm(planes)
        self.conv3 = MaskedConv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = MaskedBatchNorm(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                MaskedConv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                MaskedBatchNorm(self.expansion*planes)
            )

    def forward(self, x, p=1.0):
        out = F.relu(self.bn1(self.conv1(x, p=p), p=p))
        out = F.relu(self.bn2(self.conv2(out, p=p), p=p))
        out = self.bn3(self.conv3(out, p=p), p=p)
        
        shortcut_out = x 
        if len(self.shortcut) > 0:
            for layer in self.shortcut:
                shortcut_out = layer(shortcut_out, p=p)
        out += shortcut_out 

        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        self.in_planes = 64

        self.conv1 = MaskedConv2d(self.in_channels, 64, kernel_size=3,
                               stride=1, padding=1)
        self.bn1 = MaskedBatchNorm(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = MaskedLinear(512*block.expansion, num_classes)
        
        for m in self.modules():
        # #     if isinstance(m, nn.Conv2d):
        # #         nn.init.kaiming_normal_(
        # #             m.weight, mode="fan_out", nonlinearity="relu"
        # #         )
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride, p=1.0):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, p=1.0):
        out = F.relu(self.bn1(self.conv1(x, p=p), p=p))
        
        for layer in self.layer1:
            out = layer(out, p=p)
        for layer in self.layer2:
            out = layer(out, p=p)
        for layer in self.layer3:
            out = layer(out, p=p)
        for layer in self.layer4:
            out = layer(out, p=p)

        out = F.avg_pool2d(out, 4)
        # out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def ResNet18(in_channels=3, num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes)


def ResNet34(in_channels=3, num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes)


def ResNet50(in_channels=3, num_classes=10):
    return ResNet(Bottleneck, [3, 6, 4, 3], in_channels=in_channels, num_classes=num_classes)


# LSTM 
class LSTMModel(nn.Module):
    def __init__(self, args=None, hidden_dims=[120, 84], device=None):
        super(LSTMModel, self).__init__()
        
        self.args = args
        self.device = device
        
        V = args['embed_num']
        D = args['embed_dim']
        C = 5 
        
        self.dimension = hidden_dims[0]
        
        self.embed = nn.Embedding(V, D)
        self.lstm = MaskedLSTM(input_size=300, hidden_size=self.dimension, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=0.5)
        
        self.fc1 = MaskedLinear(2 * self.dimension, hidden_dims[0])
        self.fc2 = MaskedLinear(hidden_dims[0], hidden_dims[1])
        
        self.classifier = MaskedLinear(hidden_dims[1], C)
    
    def forward(self, x, p=1.0):
        text, text_len = x 
        text_emb = self.embed(text)
        
        packed_input = pack_padded_sequence(text_emb, text_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input, p=p)
        
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        out_forward = output[range(len(output)), text_len-1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.dropout(out_reduced)
        
        x = F.relu(self.fc1(text_fea, p=p))
        x = F.relu(self.fc2(x, p=p))
        x = self.classifier(x, p=1.)
        
        return x 


class LSTM_Small_Model(nn.Module):
    def __init__(self):
        super(LSTM_Small_Model, self).__init__()
        
        self.embed = nn.Embedding(80, 8)
        self.lstm1 = MaskedLSTM(8, 512, 1, batch_first=True)
        self.lstm2 = MaskedLSTM(512, 512, 1, batch_first=True)
        
        self.drop = nn.Dropout()
        self.fc = MaskedLinear(512, 80)


    def forward(self, x, h1=None, h2=None, p=1.0):
        x = self.embed(x)
        out_dim = int(math.ceil(p * h1[0].size(2)))
        
        if p < 1.0:
            h1_a = (h1[0][:, :, :out_dim], h1[1][:, :, :out_dim])
            x, h1_a = self.lstm1(x, h1_a, p=p)
            h1[0][:, :, :out_dim], h1[1][:, :, :out_dim] = h1_a[0], h1_a[1]
        else:
            x, h1 = self.lstm1(x, h1, p=p)
        
        if p < 1.0:
            h2_a = (h2[0][:, :, :out_dim], h2[1][:, :, :out_dim])
            x, h2_a = self.lstm2(x, h2_a, p=p)
            h2[0][:, :, :out_dim], h2[1][:, :, :out_dim] = h2_a[0], h2_a[1]
        else:
            x, h2 = self.lstm2(x, h2, p=p)
        
        x = self.drop(x[:, -1, :])
        x = self.fc(x, p=1.)
        # h1, h2 = (h1[0].detach(), h1[1].detach()), (h2[0].detach(), h2[1].detach())
        return x, (h1, h2) 


    def init_hidden(self, batch_size, device):
        # hidden1 = (torch.zeros(1, batch_size, 512).to(device).detach(), torch.zeros(1, batch_size, 512).to(device).detach())
        # hidden2 = (torch.zeros(1, batch_size, 512).to(device).detach(), torch.zeros(1, batch_size, 512).to(device).detach())
        hidden1 = (torch.zeros(1, batch_size, 512).to(device), torch.zeros(1, batch_size, 512).to(device))
        hidden2 = (torch.zeros(1, batch_size, 512).to(device), torch.zeros(1, batch_size, 512).to(device))
        return hidden1, hidden2


class LSTM_Small_ModelF(nn.Module):
    def __init__(self):
        super(LSTM_Small_ModelF, self).__init__()
        
        self.embed = nn.Embedding(80, 8)
        self.lstm1 = nn.LSTM(8, 256, 2, batch_first=True)
        
        self.drop = nn.Dropout()
        self.fc = nn.Linear(256, 80)


    def forward(self, x, h=None, p=1.0):
        x = self.embed(x)
        x, h = self.lstm1(x, h)
        x = self.drop(x[:, -1, :])
        x = self.fc(x)
        h = (h[0].detach(), h[1].detach())
        return x, h 


    def init_hidden(self, batch_size, device):
        hidden1 = (torch.zeros(2, batch_size, 256).to(device).detach(), torch.zeros(2, batch_size, 256).to(device).detach())
        return hidden1
