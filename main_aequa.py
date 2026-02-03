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
from models import Model, SmallConvModel, ConvCIFAR10Model, ResNet18, ResNet34, ResNet50, LSTMModel, LSTM_Small_Model
from torchvision.models import resnet18, resnet34, resnet50

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
parser.add_argument('--jobtype', type=str, help='Method name', default='aequa')

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

    elif args['dataset'] == 'svhn':
        if args['model_arch'] == 'resnet':
            global_model = ResNet18(num_classes=10).to(device)
            for i in range(n_participants): 
                model = ResNet18(num_classes=10).to(device)
                participants.append(model)

        elif args['model_arch'] == 'cnn':
            global_model = ConvCIFAR10Model().to(device)
            for i in range(n_participants):
                model = ConvCIFAR10Model().to(device) 
                participants.append(model)
        else:
            raise NotImplementedError()
    
    elif args['dataset'] == 'fedisic':
        if args['model_arch'] == 'resnet':
            pretrained_resnet_model = resnet18(pretrained=True)
            global_model = ResNet18(num_classes=8).to(device)
            
            pretrained_dict = pretrained_resnet_model.state_dict()
            custom_dict = global_model.state_dict()
            
            # filter out unnecessary keys and map prrtrained weights
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in custom_dict and v.size() == custom_dict[k].size()}
            
            custom_dict.update(pretrained_dict)
            global_model.load_state_dict(custom_dict, strict=False)
            
            for i in range(n_participants): 
                model = copy.deepcopy(global_model)  # ResNet18(num_classes=8).to(device)
                participants.append(model)
        elif args['model_arch'] == 'resnet34':
            # pretrained_resnet_model = resnet34(pretrained=True)
            global_model = ResNet34(num_classes=8).to(device)
            
            # pretrained_dict = pretrained_resnet_model.state_dict()
            # custom_dict = global_model.state_dict()
            
            # # filter out unnecessary keys and map prrtrained weights
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in custom_dict and v.size() == custom_dict[k].size()}
            
            # custom_dict.update(pretrained_dict)
            # global_model.load_state_dict(custom_dict, strict=False)
            
            for i in range(n_participants): 
                # model = copy.deepcopy(global_model)  # ResNet18(num_classes=8).to(device)
                model = ResNet34(num_classes=8).to(device)
                participants.append(model)
        elif args['model_arch'] == 'resnet50':
            pretrained_resnet_model = resnet50(pretrained=True)
            global_model = ResNet50(num_classes=8).to(device)
            
            pretrained_dict = pretrained_resnet_model.state_dict()
            custom_dict = global_model.state_dict()
            
            # filter out unnecessary keys and map prrtrained weights
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in custom_dict and v.size() == custom_dict[k].size()}
            
            custom_dict.update(pretrained_dict)
            global_model.load_state_dict(custom_dict, strict=False)
            
            for i in range(n_participants): 
                model = copy.deepcopy(global_model) 
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
            model = LSTM_Small_Model().to(device)
            participants.append(model)

    else:
        raise NotImplementedError()

    # initial broadcasting
    for participant_model in participants: 
        participant_model.load_state_dict(copy.deepcopy(global_model.state_dict())) 
        
    num_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad) 
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

    # optimizers = [optim.SGD(participant_model.parameters(), lr=args['lr'], momentum=0.9, weight_decay=5e-4, nesterov=True) for participant_model in participants]
    # schedulers = [optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1) for optimizer in optimizers]

    for round_ in range(args['rounds']):
        losses = []
        
        # local training 
        for i, participant in enumerate(tqdm(participants, desc=f"Round {round_+1} training.", leave=True)):
            
            # local run over the data 
            l_epochs = LOCAL_EPOCHS[i] if (args['dataset'] == 'fedisic' and args['rounds'] == 101) else 1 
            for _ in range(l_epochs):
                loss = train_single_epoch(participant, train_loaders[i], criterion, optimizers[i], constant_width=True)
                loss += train_single_epoch(participant, train_loaders[i], criterion, optimizers[i], constant_width=False)
            losses.append(loss/2)
            
            # if (round_+1) % 100 == 0:
                
            #     # recomputing batch norm statistics. 
            #     participant.eval()
            #     # participant.apply(bn_calibration_init)
            #     if any(isinstance(m, nn.BatchNorm2d) for m in participant.modules()):
            #         for m in participant.modules():
            #             bn_calibration_init(m)
                    
            #         # recompute batch norm statistics 
            #         # participant.train()
            #         for p_val in sorted(np.arange(1.0, 1.01, 0.05), reverse=True):
            #             p_val = round(p_val, 2)
            #             total_batches = 0 
            #             for k, (images, _) in enumerate(train_loaders[i]):
            #                 participant.apply(lambda m: setattr(m, 'width_mult', p_val))
            #                 images = images.to(device)
            #                 participant(images, p_val) 
                            
            #                 total_batches += images.shape[0] 
                        
            #         # participant.apply(lambda m: setattr(m, 'width_mult', None))
        
        for i, scheduler in enumerate(schedulers):
            scheduler.step() 
    
        # aggregation phase 
        # in approach 1, we don't need to estimate or utilize contribution scores in the FL training
        # so, we simply aggregate them using vanilla FedAvg 
        
        # for name, global_param in global_model.named_parameters(): 
        #     global_param.data = sum((participant_model.state_dict()[name] for participant_model in participants)) / len(participants) 
    

        averaged_weights = OrderedDict() 
        coefs = [1 / n_participants for _ in range(n_participants)] 
        for i in range(n_participants): 
            local_weights = participants[i].state_dict() 
            for key in participants[i].state_dict().keys(): 
                if i == 0: 
                    averaged_weights[key] = coefs[i] * local_weights[key]
                else:
                    averaged_weights[key] += coefs[i] * local_weights[key]
        
        global_model.load_state_dict(copy.deepcopy(averaged_weights))

        # def recalibrate_bn(model, data_loader, device):
        #     model.train()
        #     with torch.no_grad():
        #         for p in sorted([0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0], reverse=True):
        #             for images, _ in data_loader:
        #                 images = images.to(device)
        #                 model(images, p=p)
        #     model.eval()

        # recalibrate_bn(global_model, test_loader, device)
        
        # Validation Phase 
        # accuracies = [validate(participant, test_loader, p=contributions[i]) for i, participant in enumerate(participants)] 
        accuracies_p = []
        ps = [0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]
        for p in ps:
            accuracy = validate(global_model, test_loader, p=p)
            accuracies_p.append(accuracy)
        
        logging.info(f"Round {round_ + 1}: GL[0.25]: {accuracies_p[0]:.2f}%, GL[0.5]: {accuracies_p[2]:.2f}%, GL[1.0]: {accuracies_p[-1]:.2f}%")
        
        # logging 
        if True:
            for idx, acc_p in enumerate(accuracies_p):
                wandb.log({f"valid_acc_p{ps[idx]}": acc_p, "round": round_+1}) 
            
            for idx, loss in enumerate(losses):
                wandb.log({f"loss_{idx+1}": loss, "round": round_+1}) 
        
        # BROADCASTING PHASE 
        broadcast_models(global_model, participants) 
    
    # Collaboration ended 
    # Save the checkpoints 
    torch.save({ 
        'model_state_dict': global_model.state_dict(), 
        'best_val_acc': accuracies_p[-1] 
    }, f"ckpt/{file_name}.pt")


if __name__ == '__main__':
    main(args)