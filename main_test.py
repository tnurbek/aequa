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

import os 
import wandb
import argparse 

from utils import set_seed
from data_loader import CommonDataLoader
from models import Model, SmallConvModel, ConvCIFAR10Model, ResNet18

from utils import train_single_epoch, validate, broadcast_models


parser = argparse.ArgumentParser(description='Proper Fairness algorithm. Approach I')
parser.add_argument('-D', '--dataset', type=str, help='Dataset name', default='cifar10')
parser.add_argument('-N', '--n_participants', type=int, help='Number of participants', default=10)
parser.add_argument('-RS', '--random_seed', type=int, help='Random seed for reproducibility', default=1)
parser.add_argument('-S', '--split', type=str, help='Data splitting method', default='imbalanced')
parser.add_argument('-T', '--rounds', type=int, help='Number of comm. rounds', default=100)
parser.add_argument('-lr', '--lr', type=float, help='Learning rate', default=0.1)
parser.add_argument('-alpha', '--alpha', type=float, help='Dirichlet parameter', default=0.5)
parser.add_argument('-model_arch', '--model_arch', type=str, help='Model architecture', default='resnet', choices=['fc', 'cnn', 'cnn_small', 'resnet'])
parser.add_argument('-gpu', '--gpu', type=int, help='GPU id', default=0)
parser.add_argument('--batch_size', type=int, help='Batch size', default=128)

# logging details
parser.add_argument('--group', type=str, help='Group name', default='exp1')
parser.add_argument('--jobtype', type=str, help='Method name', default='appr1')

parser.add_argument('-CA', '--contribution_algorithm', type=str, help='Contribution assessment algorithm', default='shapfed', choices=['cgsv', 'shapfed'])
parser.add_argument('-beta', '--beta', type=float, help='Softmax Softening parameter', default=15.0)

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
}

args.update(vars(config))
print(args)


def main(args):
    # set seeds 
    set_seed(args['random_seed'])
    # file_name = f"{args['group']}_{args['jobtype']}_{args['dataset']}_{args['n_participants']}_{args['split']}_seed{args['random_seed']}"
    if 'appr1' in args['jobtype']:
        if args['split'] != 'dirichlet':
            file_name = f"{args['group']}_{args['jobtype']}_{args['dataset']}_{args['n_participants']}_{args['split']}_{args['model_arch']}_bs{args['batch_size']}_lr{args['lr']}_seed{args['random_seed']}"
        else:  # elif args['split'] == 'dirichlet':
            file_name = f"{args['group']}_{args['jobtype']}_{args['dataset']}_{args['n_participants']}_{args['split']}_{args['alpha']}_{args['model_arch']}_bs{args['batch_size']}_lr{args['lr']}_seed{args['random_seed']}"
    elif 'appr2' in args['jobtype']:
        if args['split'] != 'dirichlet':
            file_name = f"{args['group']}_{args['jobtype']}_{args['dataset']}_{args['n_participants']}_{args['split']}_{args['model_arch']}_{args['contribution_algorithm']}_beta{args['beta']}_bs{args['batch_size']}_lr{args['lr']}_seed{args['random_seed']}"
        else:
            file_name = f"{args['group']}_{args['jobtype']}_{args['dataset']}_{args['n_participants']}_{args['split']}_{args['alpha']}_{args['model_arch']}_{args['contribution_algorithm']}_beta{args['beta']}_bs{args['batch_size']}_lr{args['lr']}_seed{args['random_seed']}"
    else:
        raise NotImplementedError()

    n_participants = args['n_participants']
    data_loader = CommonDataLoader(args['dataset'], batch_size=args['batch_size'], n_participants=n_participants, partition=args['split'], seed=args['random_seed'], device=device, alpha=args['alpha'])
    test_loader = data_loader.get_test_loader()

    participants = []
    if args['dataset'] == 'synthetic':
        global_model = Model().to(device)
        for i in range(n_participants):
            model = Model().to(device)
            participants.append(model)
            
        # participants = [Model().cuda() for _ in range(n_participants)]
    elif args['dataset'] == 'mnist':
        global_model = SmallConvModel().to(device)
        for i in range(n_participants):
            model = SmallConvModel().to(device)
            participants.append(model)
        # participants = [SmallConvModel().to(device) for _ in range(n_participants)]
    elif args['dataset'] == 'fmnist':
        global_model = SmallConvModel().to(device)
        for i in range(n_participants):
            model = SmallConvModel().to(device)
            participants.append(model)
    elif args['dataset'] == 'cifar10':
        
        if args['model_arch'] == 'cnn':
            global_model = ConvCIFAR10Model().to(device)
            for i in range(n_participants):
                model = ConvCIFAR10Model().to(device) 
                participants.append(model)
            # participants = [ConvCIFAR10Model().to(device) for _ in range(n_participants)]
        elif args['model_arch'] == 'resnet':
            global_model = ResNet18().to(device)
            for i in range(n_participants):
                model = ResNet18().to(device)
                participants.append(model)
            # participants = [ResNet18().to(device) for _ in range(n_participants)]
        else:
            raise NotImplementedError() 

    elif args['dataset'] == 'cifar100':
        
        if args['model_arch'] == 'resnet':
            global_model = ResNet18(num_classes=100).to(device)
            for i in range(n_participants): 
                model = ResNet18(num_classes=100).to(device)
                participants.append(model)
            # participants = [ResNet18(num_classes=100).to(device) for _ in range(n_participants)]
        elif args['model_arch'] == 'cnn':
            global_model = ConvCIFAR10Model(n_classes=100).to(device)
            for i in range(n_participants):
                model = ConvCIFAR10Model(n_classes=100).to(device)
                participants.append(model)
            # participants = [ConvCIFAR10Model(n_classes=100).to(device) for _ in range(n_participants)]
        else:
            raise NotImplementedError()

    else:
        raise NotImplementedError()

    global_model.load_state_dict(torch.load(f"ckpt/{file_name}.pt")['model_state_dict'])
    global_model.eval()
    
    print(torch.load(f"ckpt/{file_name}.pt")['best_val_acc'])
    
    # Validation Phase 
    accuracies_p = []
    ps = [0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]
    ps = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.75, 0.8, 0.9, 0.95, 1.0]
    for p in ps:
        accuracy = validate(global_model, test_loader, p=p)
        accuracies_p.append(accuracy) 
    
    print(accuracies_p)
    for idx, acc in enumerate(accuracies_p):
        print(f"{acc:.2f}")
        # print(f"{ps[idx]}: {acc:.2f}")


if __name__ == '__main__':
    main(args)