"""
- they share all the weights and get broadcasted all the weights; 
- only, by the end of the collaboration, they get to see/receive what they really contributed:
    - subnetwork / slim network [either small/medium/large(whole) network];
- that they will use in the inference stage;
"""

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict 

import os 
import copy 
import wandb
import logging
import argparse 

from tqdm import tqdm

from utils import init_logging, set_seed
from data_loader import CommonDataLoader
from models import Model, SmallConvModel, ConvCIFAR10Model, ResNet18, ResNet34, ResNet50, LSTMModel, LSTM_Small_Model, LSTM_Small_ModelF

from utils import train_single_epoch, validate, broadcast_models, bn_calibration_init


parser = argparse.ArgumentParser(description='Proper Fairness algorithm. Approach I')
parser.add_argument('-D', '--dataset', type=str, help='Dataset name', default='cifar10')
parser.add_argument('-N', '--n_participants', type=int, help='Number of participants', default=10)
parser.add_argument('-RS', '--random_seed', type=int, help='Random seed for reproducibility', default=1)
parser.add_argument('-S', '--split', type=str, help='Data splitting method', default='homogeneous')
parser.add_argument('-T', '--rounds', type=int, help='Number of comm. rounds', default=100)
parser.add_argument('-lr', '--lr', type=float, help='Learning rate', default=0.01)
parser.add_argument('-alpha', '--alpha', type=float, help='Dirichlet parameter', default=0.5)
parser.add_argument('-model_arch', '--model_arch', type=str, help='Model architecture', default='cnn', choices=['fc', 'cnn', 'cnn_small', 'resnet', 'resnet34', 'lstm', 'resnet50'])
parser.add_argument('-gpu', '--gpu', type=int, help='GPU id', default=0)
parser.add_argument('--batch_size', type=int, help='Batch size', default=128)

# logging details
parser.add_argument('--group', type=str, help='Group name', default='exp1')
parser.add_argument('--jobtype', type=str, help='Method name', default='sa')

config = parser.parse_args()
print(config)

# GPU Setting
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID' 
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu) 

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

args = {
    'data_dir': 'data', 
    'num_workers': 4, 
    'num_classes': 10, 
    # LSTM settings 
    'embed_num': 20000, 
    'embed_dim': 300, 
    'static': False,
}

args.update(vars(config))
print(args)

print('device:', device) 

LOCAL_EPOCHS = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 8}

def main(args):
    # set seeds 
    set_seed(args['random_seed'])
    
    # to create necessary directories [if not already present]
    os.makedirs("logs", exist_ok=True)
    os.makedirs("ckpt", exist_ok=True)

    # logging init 
    if args['split'] != 'dirichlet':
        file_name = f"{args['group']}_{args['jobtype']}_{args['dataset']}_{args['n_participants']}_{args['split']}_{args['model_arch']}_bs{args['batch_size']}_lr{args['lr']}_seed{args['random_seed']}"
        if args['rounds'] == 50:
            file_name = f"{args['group']}_{args['jobtype']}_{args['dataset']}_{args['n_participants']}_{args['split']}_{args['model_arch']}_T{args['rounds']}_bs{args['batch_size']}_lr{args['lr']}_seed{args['random_seed']}"
    elif args['split'] == 'dirichlet':
        file_name = f"{args['group']}_{args['jobtype']}_{args['dataset']}_{args['n_participants']}_{args['split']}_{args['alpha']}_{args['model_arch']}_bs{args['batch_size']}_lr{args['lr']}_seed{args['random_seed']}"
        if args['rounds'] == 50:
            file_name = f"{args['group']}_{args['jobtype']}_{args['dataset']}_{args['n_participants']}_{args['split']}_{args['alpha']}_{args['model_arch']}_T{args['rounds']}_bs{args['batch_size']}_lr{args['lr']}_seed{args['random_seed']}"
    else:
        raise NotImplementedError()

    wandb.init(project='aequa-runs', name=file_name, group=args['group'], job_type=args['jobtype'], config=args)
    init_logging(f'logs/{file_name}.log')
    logging.info(args)


    n_participants = args['n_participants']
    data_loader = CommonDataLoader(args['dataset'], batch_size=args['batch_size'], n_participants=n_participants, partition=args['split'], seed=args['random_seed'], device=device, alpha=args['alpha'], args_dict=args)
    train_loaders = data_loader.get_train_loaders()
    test_loader = data_loader.get_test_loader()

    participants = []
    if args['dataset'] == 'synthetic':
        for i in range(n_participants):
            model = Model().to(device)
            participants.append(model)
    elif args['dataset'] == 'mnist':
        for i in range(n_participants):
            model = SmallConvModel().to(device)
            participants.append(model)
    elif args['dataset'] == 'fmnist':
        for i in range(n_participants):
            model = SmallConvModel().to(device)
            participants.append(model)
    elif args['dataset'] == 'cifar10':
        if args['model_arch'] == 'cnn':
            for i in range(n_participants):
                model = ConvCIFAR10Model().to(device) 
                participants.append(model)
        elif args['model_arch'] == 'resnet':
            for i in range(n_participants):
                model = ResNet18().to(device)
                participants.append(model)
        else:
            raise NotImplementedError() 
    elif args['dataset'] == 'cifar100':
        if args['model_arch'] == 'resnet':
            for i in range(n_participants): 
                model = ResNet18(num_classes=100).to(device)
                participants.append(model)
        elif args['model_arch'] == 'cnn':
            for i in range(n_participants):
                model = ConvCIFAR10Model(n_classes=100).to(device)
                participants.append(model)
        else:
            raise NotImplementedError()
    elif args['dataset'] == 'svhn':
        if args['model_arch'] == 'resnet':
            for i in range(n_participants): 
                model = ResNet18(num_classes=10).to(device)
                participants.append(model)
        elif args['model_arch'] == 'cnn':
            for i in range(n_participants):
                model = ConvCIFAR10Model().to(device) 
                participants.append(model)
        else:
            raise NotImplementedError()
    elif args['dataset'] == 'fedisic':
        if args['model_arch'] == 'resnet':
            for i in range(n_participants): 
                model = ResNet18(num_classes=8).to(device)
                participants.append(model)
        elif args['model_arch'] == 'resnet34':
            for i in range(n_participants): 
                model = ResNet34(num_classes=8).to(device) 
                participants.append(model)
        elif args['model_arch'] == 'resnet50':
            for i in range(n_participants): 
                model = ResNet50(num_classes=8).to(device) 
                participants.append(model)
        else:
            raise NotImplementedError()
    elif args['dataset'] == 'sst':
        global_model = LSTMModel(args=args).to(device)
        for i in range(n_participants):
            model = LSTMModel(args=args).to(device)
            participants.append(model)
    elif args['dataset'] == 'shakespeare':
        global_model = LSTM_Small_Model().to(device)
        for i in range(n_participants):
            model = LSTM_Small_ModelF().to(device)
            participants.append(model)
    else:
        raise NotImplementedError()
        
    num_params = sum(p.numel() for p in participants[0].parameters() if p.requires_grad) 
    logging.info(f"trainable parameters: {num_params/2**20:.4f}M") 
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    criterion = nn.CrossEntropyLoss()
    optimizers, schedulers = [], []
    milestones = [int(0.5 * args['rounds']), int(0.75 * args['rounds'])]
    
    for participant_model in participants:
        if args['dataset'] == 'sst':
            optimizer = optim.Adam(participant_model.parameters(), lr=args['lr']) 
        else:
            optimizer = optim.SGD(participant_model.parameters(), lr=args['lr'], momentum=0.9, weight_decay=5e-4, nesterov=True)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
    
    # Communication Rounds 
    for round_ in range(args['rounds']):
        losses = []
        
        # local training 
        for i, participant_model in enumerate(tqdm(participants, desc=f"Round {round_+1} training.", leave=True)):
            
            # local run over the data 
            l_epochs = LOCAL_EPOCHS[i] if (args['dataset'] == 'fedisic' and args['rounds'] == 101) else 1 
            for _ in range(l_epochs):
                if args['dataset'] == 'shakespeare':
                    loss = train_single_epoch(participant_model, train_loaders[i], criterion, optimizers[i], constant_width=True, is_rnn=True)
                else:
                    loss = train_single_epoch(participant_model, train_loaders[i], criterion, optimizers[i], constant_width=True)
            losses.append(loss) 
        
        for scheduler in schedulers:
            scheduler.step() 

        # Validation Phase 
        accuracies = []
        for participant_model in participants:
            if args['dataset'] == 'shakespeare':
                accuracy = validate(participant_model, test_loader, p=1., is_rnn=True)
            else:
                accuracy = validate(participant_model, test_loader, p=1.)
            accuracies.append(accuracy)
        
        logging.info(f"Round {round_ + 1}: P[FIRST]: {accuracies[0]:.2f}%, P[LAST]: {accuracies[-1]:.2f}%")
        
        # logging 
        if True:
            for idx, acc_p in enumerate(accuracies):
                wandb.log({f"valid_acc_{idx+1}": acc_p, "round": round_+1})

if __name__ == '__main__':
    main(args)