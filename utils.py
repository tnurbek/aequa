import os 
import math
import torch
import random
import logging
import numpy as np 
from torch import nn

from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score

from torchtext.data import Batch


def init_logging(logFilename):
    """Init for logging"""
    logger = logging.getLogger("")

    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s-%(levelname)s] %(message)s",
            datefmt="%y-%m-%d %H:%M:%S",
            filename=logFilename,
            filemode="w",
        )
        console = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s-%(levelname)s] %(message)s")
        console.setFormatter(formatter)
        logger.addHandler(console)
        
        
def generate_mask(shape, percentile=0.5): 
    assert len(shape) == 1, "This function currently supports 1D weight matrices only."
    
    mask = torch.zeros(shape)
    until_num = math.ceil(percentile * shape[0])
    mask[:until_num] = 1
    return mask.cuda() 


def contribution_mapping(contribution):
    thresholds = [round(x, 2) for x in np.arange(0.25, 1.01, 0.05)]
    for threshold in thresholds:
        if contribution <= threshold:
            return threshold
    
    # Return the maximum threshold as fallback
    return thresholds[-1]


def mkdir(path):
    try:
        os.makedirs(path)
    except Exception as _:
        pass


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_data_stats(targets, indices):
    participant_class_counts = {}
    
    for participant_id, data_id in indices.items():
        unq, unq_count = np.unique(targets[data_id], return_counts=True)
        participant_class_counts[participant_id] = {unq[i]: unq_count[i] for i in range(len(unq))}

    return participant_class_counts


CLIP_RNN_GRAD = 5
# TRAINING UTILS 

def train_single_epoch(model, train_loader, criterion, optimizer, constant_width=False, is_rnn=False):
    model.train()
    total_loss = 0
    p = 1.0
    
    # if is_rnn:
    #     hidden = model.init_hidden(256, 'cuda')
        
    for idx, batch in enumerate(train_loader):
        if isinstance(batch, Batch):
            data, label = batch.text, batch.label 
            if isinstance(data, tuple):
                inputs, labels = (data[0].cuda(), data[1].cuda()), label.cuda()
            else:
                inputs, labels = data.cuda(), label.cuda() 
        else:
            inputs, labels = batch[0], batch[1] 
            inputs, labels = inputs.cuda(), labels.cuda()
        
        if is_rnn and idx==0: #  and (constant_width or idx==0):
            batch_size = inputs.size(0)
            h1, h2 = model.init_hidden(batch_size, 'cuda')
        
        optimizer.zero_grad()
        # uniform sampling
        if constant_width is False: 
            p = np.random.choice([round(x, 2) for x in np.arange(0.25, 1.01, 0.05)], size=1)
        
        # forward pass with masks
        if is_rnn:
            outputs, (h1, h2) = model(inputs, h1, h2, p=p)
            labels = labels.view(-1)
        else:
            outputs = model(inputs, p=p)
            
        loss = criterion(outputs, labels)
        
        loss.backward(retain_graph=True)
        
        # shakespeare 
        if is_rnn:
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_RNN_GRAD)
        
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader) 


def train_single_epoch_appr2(model, train_loader, criterion, optimizer, constant_width=False, max_width=1.0):
    model.train()
    total_loss = 0
    p = max_width
    for inputs, labels in train_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        
        if constant_width is False:
            p = np.random.choice([round(x, 2) for x in np.arange(0.25, max_width+0.01, 0.05)], size=1)
        
        # forward pass with masks
        outputs = model(inputs, p=p)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader) 


def validate(model, val_loader, p, is_rnn=False):
    model.eval()
    predicted_labels, true_labels = [], []
    
    if is_rnn: 
        h1, h2 = model.init_hidden(256, 'cuda')
    
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, Batch):
                data, label = batch.text, batch.label 
                if isinstance(data, tuple):
                    inputs, labels = (data[0].cuda(), data[1].cuda()), label.cuda()
                else:
                    inputs, labels = data.cuda(), label.cuda() 
            else:
                inputs, labels = batch[0], batch[1] 
                inputs, labels = inputs.cuda(), labels.cuda()
            
            if is_rnn:
                outputs, (h1, h2) = model(inputs, h1, h2, p=p)
            else:
                outputs = model(inputs, p=p)

            _, predicted = torch.max(outputs.data, 1)
            
            # collect all true labels and predictions
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
    
    # calculate balanced accuracy
    balanced_acc = balanced_accuracy_score(true_labels, predicted_labels)
    return balanced_acc * 100 


def broadcast_models(global_model, subnets, excluded_params=[]):
    global_state_dict = global_model.state_dict()
    filtered_state_dict = {k: v for k, v in global_state_dict.items() if k not in excluded_params}

    for subnet in subnets: 
        subnet_state_dict = subnet.state_dict()
        
        # Update only the parameters that exist in the subnet and are not excluded 
        updated_state_dict = {k: filtered_state_dict[k] if k in filtered_state_dict else v for k, v in subnet_state_dict.items()}
        subnet.load_state_dict(updated_state_dict) 
        

def bn_calibration_init(m):
    """calculating post-statistics of bn"""
    
    if getattr(m, 'track_running_stats', False):
        # reset all values for post-statistics 
        # print(m)
        # print("before", m.running_mean.mean())
        m.reset_running_stats()
        # print("after ", m.running_mean.mean())
        # set bn in training mode to update post-statistics 
        m.training = True 
        m.momentum = 0.0 
        

# Language models utils 

ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)

trls = ['-', 'a', '?', 'l', 'L', 't', 'h', 'S', 'v', 'Z', 'R', 'y', 'T', ')', 'd', 'i', 's', 'o', 'k', 'M', 'G', 'H', '"', ' ', "'", '&', 'P', 'E', 'A', 'n', 'Q', 'X', ':', ',', 'K', 'U', 'j', 'q', 'u', 'g', 'b', '4', '>', 'z', 'C', 'D', 'Y', 'f', 'V', '[', 'I', 'm', 'e',
 'F', '.', 'x', 'B', 'N', 'W', '(', 'r', 'J', 'c', ';', ']', 'w', 'p', '!', 'O']
tels = ['-', 'a', '?', 'l', 'L', 't', 'h', 'S', 'v', 'Z', 'R', 'y', 'T', ')', 'd', '8', 's', 'i', 'o', 'k', 'M', 'H', 'G', ' ', "'", 'P', '1', 'E', 'A', 'n', 'Q', ':', ',', 'K', 'U', 'j', 'q', 'u', 'b', 'g', '>', 'z', 'C', 'D', '6', 'Y', 'f', 'V', '[', 'I', 'm', 'e', 'F',
 '.', 'x', 'B', 'N', 'W', '(', '0', 'r', 'J', 'c', ';', ']', 'w', 'p', '!', 'O']

def _one_hot(index, size):
    '''returns one-hot vector with given size and value 1 at given index
    '''
    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec


def letter_to_vec(letter):
    '''returns one-hot representation of given letter
    '''
    index = ALL_LETTERS.find(letter)
    return index


def word_to_indices(word):
    '''returns a list of character indices
    Args:
        word: string
    Return:
        indices: int list with length len(word)
    '''
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices



def train_single_epoch_iafl(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
        
    # if is_rnn:
    #     hidden = model.init_hidden(256, 'cuda')
        
    for batch in train_loader:
        if isinstance(batch, Batch):
            data, label = batch.text, batch.label 
            if isinstance(data, tuple):
                inputs, labels = (data[0].cuda(), data[1].cuda()), label.cuda()
            else:
                inputs, labels = data.cuda(), label.cuda() 
        else:
            inputs, labels = batch[0], batch[1] 
            inputs, labels = inputs.cuda(), labels.cuda()
        
        optimizer.zero_grad()
        
        # forward pass with masks
        outputs = model(inputs, p=1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return model, total_loss / len(train_loader) 


def compute_grad_update(old_model, new_model, device=None):
    if device:
        old_model, new_model = old_model.to(device), new_model.to(device)
    
    global_dict = old_model.state_dict()
    
    for k in global_dict.keys():
        global_dict[k] = new_model.state_dict()[k].float() - old_model.state_dict()[k].float()
    return global_dict


def add_gradients_to_model_batch(models, gradients, weights, device=None):
    '''
        Slightly faster than looping add_gradients_to_model
        
        models: list of N models
        gradients: list of N gradients
        weights: list of N x N, where weights[i] stores the dim N weights for each gradient
    '''
    
    state_dicts = [model.state_dict() for model in models]
        
    for i, gradient in enumerate(gradients):
        for j, state in enumerate(state_dicts):
            for k in state_dicts[0].keys():
                state[k] = state[k].float()
                state[k] += gradient[k] * weights[j][i]

    for i, model in enumerate(models):
        model.load_state_dict(state_dicts[i])



def add_gradients_to_model(model, gradients, weights, device=None):
    
    global_dict = model.state_dict()
    
    for k in global_dict.keys():
        global_dict[k] = model.state_dict()[k].float()
        for i in range(len(gradients)):
            global_dict[k] += gradients[i][k] * weights[i]
  
    model.load_state_dict(global_dict)

