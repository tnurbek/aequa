import math
import torch
import random
import numpy as np 

from torchvision import datasets
from torchvision import transforms

import os
from collections import Counter

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from datasets import CustomDataset, CustomCombinedDataset

from datasets import FastMNIST, FastCIFAR10, FastCIFAR100, FastRGBMNIST, FastSVHN, FastFMNIST

from torchtext.datasets import SST
from torchtext.data import Field, LabelField, BucketIterator
from torchtext.data import Dataset as TextDataset

from datasets import ShakeSpeareDataset

from utils import mkdir, set_seed, get_data_stats


####### SYNTHETIC DATASET 

def generate_data(n_points_per_center, primary_centers, secondary_offsets, cov_matrices):
    """
    Generate random data for each secondary center associated with a primary class.
    
    Parameters:
    - n_points_per_center: number of data points per secondary center
    - primary_centers: list of primary center points
    - secondary_offsets: list of relative offsets for the secondary centers
    - cov_matrices: list of 2x2 covariance matrices for each secondary center
    
    Returns:
    - data: tensor containing the data points
    - labels: tensor containing the primary class labels
    """
    data_list = []
    label_list = []

    for label, primary_center in enumerate(primary_centers):
        for j, (offset, cov_matrix) in enumerate(zip(secondary_offsets, cov_matrices)):
            # Calculate secondary center
            secondary_center = primary_center + offset

            # Generate random data points and apply the covariance transformation
            raw_data = torch.randn(n_points_per_center, 2)
            if label == 0:
                transformed_data = raw_data @ torch.tensor([[0.05, 0.], [0., 0.45]]) + secondary_center
            elif label == 3:
                transformed_data = raw_data @ torch.tensor([[0.25, 0.], [0., 0.05]]) + secondary_center
            # elif label == 1:
            #     transformed_data = raw_data @ torch.tensor([[0.01, 0.05], [0.05, 0.01]]) + secondary_center
            else:
                transformed_data = raw_data @ cov_matrix.T + secondary_center
            
            data_list.append(transformed_data)

            # Assign primary label to each data point 
            if label in [0, 1]:
                if j % 2 == 0:
                    labels = torch.full((n_points_per_center,), 0, dtype=torch.long)
                else:
                    labels = torch.full((n_points_per_center,), 1, dtype=torch.long)
            else:
                if j % 2 == 0:
                    labels = torch.full((n_points_per_center,), 2, dtype=torch.long)
                else:
                    labels = torch.full((n_points_per_center,), 3, dtype=torch.long)
            label_list.append(labels)

    return torch.cat(data_list, 0), torch.cat(label_list, 0)


def get_custom_train_loader(data_dir, batch_size, random_seed, shuffle=True, num_workers=4, pin_memory=True, model_name="", model_num=5, intersection=0.0):
    # define transforms
    
    if batch_size == 64:
        n_points_per_center = 2000 
    elif batch_size == 32: 
        n_points_per_center = 2000 
    elif batch_size == 8:
        n_points_per_center = 200 
    else: 
        n_points_per_center = 2000 # 200 

    # Parameters 
    primary_centers = [torch.tensor([-4, 4]),  # top left 
                    torch.tensor([4, 4]),   # top right 
                    torch.tensor([-4, -4]), # bottom left 
                    torch.tensor([4, -4])]  # bottom right  

    secondary_offsets = [torch.tensor([0, 0]), 
                        torch.tensor([1, -1]), 
                        torch.tensor([2, -2]), 
                        torch.tensor([3, -3])] 

    cov_matrices = [torch.tensor([[0.25, 0.15], [0.15, 0.25]]), 
                    torch.tensor([[0.25, 0.15], [0.15, 0.25]]), 
                    torch.tensor([[0.25, 0.15], [0.15, 0.25]]), 
                    torch.tensor([[0.25, 0.15], [0.15, 0.25]]),] 

    # Generate data
    data, labels = generate_data(n_points_per_center, primary_centers, secondary_offsets, cov_matrices) 
    
    # 2 to 5D 
    x = data[:, 0].unsqueeze(1)  # unsqueeze adds a new dimension, making it a column vector 
    y = data[:, 1].unsqueeze(1) 
    x2 = x**2 
    y2 = y**2 
    xy = x * y 
    
    data = torch.Tensor(torch.cat([x, y, x2, y2, xy], dim=1))
    labels = torch.Tensor(labels)  # .long()
    
    mean_values = torch.mean(data, dim=0)
    std_values = torch.std(data, dim=0)

    data = (data - mean_values) / std_values 

    # Create an instance of the CustomDataset using the previously generated data and labels
    dataset = CustomDataset(data, labels)
    
    # Create a DataLoader 
    
    if shuffle:
        np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    is_iid, pnumber = False, model_num 
    if "_iid_" in model_name:
        is_iid = True
    
    if pnumber == 1:
        return [
                    torch.utils.data.DataLoader(
                        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory,
                    )
                ]

    lst = []
    class_size = len(dataset) // len(set(np.array(dataset.labels)))
    num_classes = len(set(np.array(dataset.labels)))

    # dictionary of labels map
    labels = np.array(dataset.labels)
    dct = {}
    for idx, label in enumerate(labels):
        if label not in dct:
            dct[label] = []
        dct[label].append(idx)

    for i in range(num_classes):
        temp = random.sample(dct[i], len(dct[i]))
        dct[i] = temp
    
    until_index = (1 - intersection) * num_classes 

    # probabilities 
    torch.set_printoptions(precision=3)
    probs = []
    for i in range(num_classes):
        if is_iid: 
            probs.append([1.0 / pnumber] * pnumber)
        else: 
            if pnumber == 2: 
                if i < until_index:
                    # if i % 2 == 0:
                    if i < 2:
                        probs.append([1.0, 0.0])
                    else:
                        probs.append([0.0, 1.0])
                else:
                    probs.append([1.0 / pnumber] * pnumber) 
            elif pnumber == 4: 
                if i < until_index:
                    if i == 0:
                        probs.append([1.0, 0.0, 0.0, 0.0])
                    elif i == 1:
                        probs.append([0.0, 1.0, 0.0, 0.0])
                    elif i == 2:
                        probs.append([0.0, 0.0, 1.0, 0.0])
                    elif i == 3:
                        probs.append([0.0, 0.0, 0.0, 1.0]) 
                else:
                    probs.append([1.0 / pnumber] * pnumber) 
    
    print(model_name)
    if 'homo_hetero' in model_name:
        print('here')
        probs = []
        for i in range(num_classes):
            if i < 2:
                probs.append([0.5, 0.0, 0.25, 0.25])
            else:
                probs.append([0.0, 0.5, 0.25, 0.25])

    print(probs, end="\n\n") 
    
    if 'imbalanced' in model_name:
        probs = []
        for i in range(num_classes):
            rho = 0.15 # config.rho 
            n = 6 # config.alloc_n 
            major_allocation = [rho] * n 
            remaining_allocation = [(1 - sum(major_allocation)) / (pnumber - n)] * (pnumber - n) 
            prob = major_allocation + remaining_allocation 
            probs.append(prob)  

    # division 
    if not is_iid:
        intersection = 0.0 
    lst = {i: [] for i in range(pnumber)} 
    for class_id, distribution in enumerate(probs):
        from_id = 0 
        for participant_id, prob in enumerate(distribution):
            to_id = int(from_id + prob * class_size)
            if participant_id == pnumber - 1:
                lst[participant_id] += dct[class_id][from_id:to_id]  # to_id 
            else:
                lst[participant_id] += dct[class_id][from_id:to_id]
            to_id = math.ceil((1 - intersection) * to_id) 
            from_id = to_id 
    
    print("[data_loader.py: ] Number of common data points:", len(list(set(lst[0]) & set(lst[1])))) 
    
    subsets = [torch.utils.data.Subset(dataset, lst[i]) for i in range(pnumber)]
    t_loaders = [torch.utils.data.DataLoader(subsets[i], batch_size=batch_size, shuffle=True) for i in range(pnumber)]
    
    for pi in range(pnumber): 
        counts = [0] * 4 
        for label in subsets[pi]:
            counts[label[1]] += 1
        print(f'{pi+1} set: ', counts, sum(counts), end="\n")


    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory,
    )
    
    # for batch in train_loader:
    #     print(batch['data'], batch['label'])
    #     break

    return t_loaders + [train_loader] 


def get_custom_test_loader(data_dir, batch_size, random_seed, shuffle=True, num_workers=4, pin_memory=True, model_name="", model_num=5, intersection=0.0): 

    # Parameters 
    n_points_per_center = 400 
    primary_centers = [torch.tensor([-4, 4]),  # top left 
                    torch.tensor([4, 4]),   # top right 
                    torch.tensor([-4, -4]), # bottom left 
                    torch.tensor([4, -4])]  # bottom right 

    secondary_offsets = [torch.tensor([0, 0]), 
                        torch.tensor([1, -1]), 
                        torch.tensor([2, -2]), 
                        torch.tensor([3, -3])] 

    cov_matrices = [torch.tensor([[0.25, 0.15], [0.15, 0.25]]), 
                    torch.tensor([[0.25, 0.15], [0.15, 0.25]]), 
                    torch.tensor([[0.25, 0.15], [0.15, 0.25]]), 
                    torch.tensor([[0.25, 0.15], [0.15, 0.25]]),] 

    # Generate data
    data, labels = generate_data(n_points_per_center, primary_centers, secondary_offsets, cov_matrices)
    
    # 2 to 5D 
    x = data[:, 0].unsqueeze(1)  # unsqueeze adds a new dimension, making it a column vector
    y = data[:, 1].unsqueeze(1) 
    x2 = x**2
    y2 = y**2
    xy = x * y
    
    data = torch.Tensor(torch.cat([x, y, x2, y2, xy], dim=1))
    labels = torch.Tensor(labels) # .long()
    
    mean_values = torch.mean(data, dim=0)
    std_values = torch.std(data, dim=0)

    data = (data - mean_values) / std_values 

    # Create an instance of the CustomDataset using the previously generated data and labels
    dataset = CustomDataset(data, labels)
    
    # Create a DataLoader
    # batch_size = 32 
    
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    pnumber = model_num

    lst = []
    class_size = len(dataset) // len(set(np.array(dataset.labels)))
    num_classes = len(set(np.array(dataset.labels))) 

    # dictionary of labels map
    labels = np.array(dataset.labels)
    dct = {}
    for idx, label in enumerate(labels):
        label = int(label)
        if label not in dct:
            dct[label] = []
        dct[label].append(idx)
    
    print(class_size)
    print(len(dct[3])) 
    print(num_classes)

    for i in range(num_classes):
        temp = random.sample(dct[i], len(dct[i]))
        dct[i] = temp

    # probabilities
    torch.set_printoptions(precision=3)
    probs = []
    for i in range(num_classes): 
        probs.append([1.0 / pnumber] * pnumber)
    print(probs, end="\n\n")

    # division
    lst = {i: [] for i in range(pnumber)} 
    for class_id, distribution in enumerate(probs): 
        from_id = 0
        for participant_id, prob in enumerate(distribution):
            class_size = len(dct[class_id])
            to_id = int(from_id + prob * class_size)
            if participant_id == pnumber - 1:
                lst[participant_id] += dct[class_id][from_id:]  # to_id
            else:
                lst[participant_id] += dct[class_id][from_id:to_id]
            from_id = to_id

    subsets = [torch.utils.data.Subset(dataset, lst[i]) for i in range(pnumber)]
    t_loaders = [torch.utils.data.DataLoader(subsets[i], batch_size=batch_size, shuffle=False) for i in range(pnumber)]
    
    for pi in range(pnumber):
        counts = [0] * 4 
        for label in subsets[pi]: 
            counts[label[1]] += 1  
        print(f'{pi+1} set: ', counts, sum(counts), end="\n")


    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return t_loaders + [data_loader]


####### MNIST DATASET 


def get_mnist_train_loader(data_dir, batch_size, random_seed, shuffle=True, num_workers=4, pin_memory=True, model_name="", model_num=5, intersection=0.0):
    # define transforms
    trans = transforms.Compose([
        transforms.Resize((32, 32)),  # Resizing to 32x32
        transforms.ToTensor(),
    ])

    # load dataset
    dataset = datasets.MNIST(root=data_dir, transform=trans, download=True, train=True) 
    
    if shuffle:
        np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    is_iid, pnumber = False, model_num
    if "_iid_" in model_name:
        is_iid = True 

    lst = []
    class_size = len(dataset) // len(set(np.array(dataset.targets)))
    # class_size = 100 
    num_classes = len(set(np.array(dataset.targets)))

    # dictionary of labels map
    labels = np.array(dataset.targets)
    dct = {}
    for idx, label in enumerate(labels):
        if label not in dct:
            dct[label] = []
        dct[label].append(idx)

    for i in range(num_classes):
        temp = random.sample(dct[i], len(dct[i]))
        dct[i] = temp

    # probabilities 
    torch.set_printoptions(precision=3)
    probs = []
    for i in range(num_classes):
        if is_iid: 
            probs.append([1.0 / pnumber] * pnumber)
        else:
            # imbalanced split 
            rho = 0.15 # config.rho 
            n = 6 # config.alloc_n 
            major_allocation = [rho] * n 
            remaining_allocation = [(1 - sum(major_allocation)) / (pnumber - n)] * (pnumber - n) 
            prob = major_allocation + remaining_allocation 
            probs.append(prob)  
    print(probs, end="\n\n") 

    lst = {i: [] for i in range(pnumber)} 
    for class_id, distribution in enumerate(probs):
        from_id = 0 
        for participant_id, prob in enumerate(distribution):
            to_id = int(from_id + prob * class_size)
            if participant_id == pnumber - 1:
                lst[participant_id] += dct[class_id][from_id:to_id]  # to_id 
            else:
                lst[participant_id] += dct[class_id][from_id:to_id]
            from_id = to_id 

    print("[data_loader.py: ] Number of common data points:", len(list(set(lst[0]) & set(lst[1])))) 
    
    subsets = [torch.utils.data.Subset(dataset, lst[i]) for i in range(pnumber)]
    t_loaders = [torch.utils.data.DataLoader(subsets[i], batch_size=batch_size, shuffle=True) for i in range(pnumber)]

    for pi in range(pnumber):
        counts = [0] * 10 
        for label in subsets[pi]:
            counts[label[1]] += 1
        print(f'{pi+1} set: ', counts, sum(counts), end="\n")

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory,
    )

    return t_loaders + [train_loader] 


def get_mnist_test_loader(data_dir, batch_size, random_seed, shuffle=False, num_workers=4, pin_memory=True, model_name="", model_num=5, intersection=0.0): 
    # define transforms
    trans = transforms.Compose([
        transforms.Resize((32, 32)), 
        transforms.ToTensor(), 
    ]) 

    # load dataset
    dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=trans
    )

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    pnumber = model_num 

    lst = []
    class_size = len(dataset) // len(set(np.array(dataset.targets)))
    # class_size = 100 
    num_classes = len(set(np.array(dataset.targets)))

    # dictionary of labels map
    labels = np.array(dataset.targets)
    dct = {}
    for idx, label in enumerate(labels):
        label = int(label)
        if label not in dct:
            dct[label] = []
        dct[label].append(idx)
    
    print(class_size)
    print(len(dct[1]))

    for i in range(num_classes):
        temp = random.sample(dct[i], len(dct[i]))
        dct[i] = temp

    # probabilities
    torch.set_printoptions(precision=3)
    probs = []
    for i in range(num_classes):
        probs.append([1.0 / pnumber] * pnumber) 
    print(probs, end="\n\n")

    # division
    lst = {i: [] for i in range(pnumber)}
    for class_id, distribution in enumerate(probs):
        from_id = 0
        for participant_id, prob in enumerate(distribution):
            # class_size = 200 
            to_id = int(from_id + prob * class_size)
            if participant_id == pnumber - 1:
                lst[participant_id] += dct[class_id][from_id:to_id]  # to_id
            else:
                lst[participant_id] += dct[class_id][from_id:to_id]
            from_id = to_id

    subsets = [torch.utils.data.Subset(dataset, lst[i]) for i in range(pnumber)]
    t_loaders = [torch.utils.data.DataLoader(subsets[i], batch_size=batch_size, shuffle=False) for i in range(pnumber)]
    
    for pi in range(pnumber):
        counts = [0] * 10
        for label in subsets[pi]:
            counts[label[1]] += 1
        print(f'{pi+1} set: ', counts, sum(counts), end="\n")

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return t_loaders + [data_loader]


class CommonDataLoader: 
    def __init__(self, dataset_name, batch_size, n_participants, partition, seed, device, alpha=1.0, args_dict=None):
        self.dataset_name = dataset_name 
        self.batch_size = batch_size
        self.n_participants = n_participants 
        self.partition = partition
        self.seed = seed
        self.device = device
        self.alpha = alpha
        
        self.args = None
        self.args_dict = args_dict
        self.datasets_list = ['synthetic', 'mnist', 'fmnist', 'cifar10', 'cifar100', 'pathmnist', 'fedisic', 'svhn', 'sst', 'shakespeare']
        
        if dataset_name in ['synthetic', 'mnist', 'fmnist', 'cifar10', 'cifar100', 'pathmnist', 'fedisic', 'svhn', 'shakespeare']:
            # get train and test datasets 
            self.train_dataset, self.test_dataset = self.get_datasets()
        elif dataset_name in ['sst']:
            # language dataset 
            self.args = {}
            self.train_dataset, self.test_dataset = self.get_datasets()
            self.test_loader = BucketIterator(self.test_dataset, batch_size=self.batch_size, sort_key=lambda x: len(x.text), device=self.device)
            
            keys_to_copy = ['embed_dim', 'kernel_num', 'kernel_sizes', 'static']
            for key in keys_to_copy:
                if key in self.args_dict:
                    self.args[key] = self.args_dict[key]
            
            print("Model embedding arguments:", self.args)
        else:
            raise NotImplementedError()
        
    
    def get_datasets(self):
        transforms_list = []
        if self.dataset_name == 'mnist': 
            train_dataset = FastMNIST('data', train=True, download=True)
            test_dataset = FastMNIST('data', train=False, download=True)
        elif self.dataset_name == 'fmnist': 
            train_dataset = FastFMNIST('data', train=True, download=True)
            test_dataset = FastFMNIST('data', train=False, download=True)
        elif self.dataset_name == 'cifar10':
            transforms_list = [
                transforms.RandomCrop(32, padding=4), 
                transforms.RandomHorizontalFlip(), 
                transforms.RandomRotation(degrees=15),
            ]
            transform = transforms.Compose(transforms_list)
            train_dataset = FastCIFAR10('data', train=True, download=True)
            test_dataset = FastCIFAR10('data', train=False, download=True)
        elif self.dataset_name == 'cifar100':
            transforms_list = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=15),
            ]
            transform = transforms.Compose(transforms_list)
            train_dataset = FastCIFAR100('data', train=True, download=True)
            test_dataset = FastCIFAR100('data', train=False, download=True)
        elif self.dataset_name == 'fedisic':
            train_dataset = ''
            test_dataset = '' 
        elif self.dataset_name == 'svhn':
            transforms_list = []
            transform = transforms.Compose(transforms_list)
            train_dataset = FastSVHN('data', split='train', download=True)
            test_dataset = FastSVHN('data', split='test', download=True)
        elif self.dataset_name == 'cifar-mnist':
            transforms_list = [[], []]
            if transforms_list:
                transform1 = transforms.Compose(transforms_list[0]) 
                transform2 = transforms.Compose(transforms_list[1]) 
            
            mnist_train_dataset = FastRGBMNIST('data', train=True, download=True)  # 3x32x32 
            mnist_test_dataset = FastRGBMNIST('data', train=False, download=True)  # 3x32x32 
            cifar10_train_dataset = FastCIFAR10('data', train=True, download=True)
            cifar10_test_dataset = FastCIFAR10('data', train=False, download=True)
        elif self.dataset_name == 'sst':
            if 'kernel_sizes' in self.args_dict.keys():
                min_len_padding = get_pad_to_min_len_fn(min_length=max(self.args_dict['kernel_sizes']))
                text_field = Field(lower=True, postprocessing=min_len_padding, include_lengths=True, batch_first=True)
            else:
                text_field = Field(lower=True, include_lengths=True, batch_first=True)
            label_field = LabelField(dtype=torch.long, sequential=False)
            train_data, _, test_data = SST.splits(text_field, label_field, root='.data', fine_grained=True)
            
            text_field.build_vocab(*([train_data, test_data]))
            label_field.build_vocab(*([train_data, test_data]))
            
            self.args['embed_num'] = len(text_field.vocab)
            self.args['class_num'] = len(label_field.vocab)
            self.args['fields'] = train_data.fields
            
            return train_data, test_data 
        elif self.dataset_name == 'shakespeare':
            train_dataset = ''
            test_dataset = ''
        elif self.dataset_name == 'synthetic':
            if self.batch_size in [64, 32]:
                n_points_per_center = 2000 
            elif self.batch_size in [8]:
                n_points_per_center = 200 
            else:
                raise NotImplementedError()
            
            # parameters 
            primary_centers = [torch.tensor([-4, 4]), torch.tensor([4, 4]), torch.tensor([-4, -4]), torch.tensor([4, -4])] 
            secondary_offsets = [torch.tensor([0, 0]), torch.tensor([1, -1]), torch.tensor([2, -2]), torch.tensor([3, -3])] 
            cov_matrices = [torch.tensor([[0.25, 0.15], [0.15, 0.25]]), torch.tensor([[0.25, 0.15], [0.15, 0.25]]), torch.tensor([[0.25, 0.15], [0.15, 0.25]]), torch.tensor([[0.25, 0.15], [0.15, 0.25]]),]
            
            # Generate data
            data, labels = generate_data(n_points_per_center, primary_centers, secondary_offsets, cov_matrices) 
            # 2 to 5D 
            x, y = data[:, 0].unsqueeze(1), data[:, 1].unsqueeze(1) 
            x2, y2, xy = x**2, y**2, x*y 
            
            train_data = torch.Tensor(torch.cat([x, y, x2, y2, xy], dim=1))
            train_labels = torch.Tensor(labels)  # .long()
            
            mean_values = torch.mean(train_data, dim=0)
            std_values = torch.std(train_data, dim=0)

            train_data = (train_data - mean_values) / std_values 
            
            # generation of test set 
            n_points_per_center = 400 
            
            # Generate data
            data, labels = generate_data(n_points_per_center, primary_centers, secondary_offsets, cov_matrices)
            # 2 to 5D 
            x, y = data[:, 0].unsqueeze(1), data[:, 1].unsqueeze(1) 
            x2, y2, xy = x**2, y**2, x*y 
            
            test_data = torch.Tensor(torch.cat([x, y, x2, y2, xy], dim=1))
            test_labels = torch.Tensor(labels) # .long()
            
            mean_values = torch.mean(test_data, dim=0)
            std_values = torch.std(test_data, dim=0)

            test_data = (test_data - mean_values) / std_values
        else:
            raise NotImplementedError()
        
        # if self.dataset_name in ['mnist', 'cifar10', 'cifar100', 'pathmnist']:
        if self.dataset_name in ['mnist', 'fmnist', 'cifar10', 'cifar100', 'svhn']:
            if transforms_list:
                # train_set = CustomDataset(train_dataset.data, train_dataset.targets, device=self.device, transform=transform)
                train_set = CustomDataset(train_dataset.data, train_dataset.targets, device="cpu", transform=transform)
            else:
                train_set = CustomDataset(train_dataset.data, train_dataset.targets, device="cpu")
            test_set = CustomDataset(test_dataset.data, test_dataset.targets, device="cpu")
        elif self.dataset_name == 'fedisic':
            train_set = ''
            test_set = '' 
        elif self.dataset_name in ['synthetic']:
            train_set = CustomDataset(train_data, train_labels, device=self.device)
            test_set = CustomDataset(test_data, test_labels, device=self.device)
        elif self.dataset_name in ['cifar-mnist']:
            if transforms_list:
                train_set = CustomCombinedDataset(mnist_train_dataset, cifar10_train_dataset, device=self.device, transform1=transform1, transform2=transform2)
            else:
                train_set = CustomCombinedDataset(mnist_train_dataset, cifar10_train_dataset, device=self.device)
            test_set = CustomCombinedDataset(mnist_test_dataset, cifar10_test_dataset, device=self.device)
        elif self.dataset_name == 'shakespeare':
            train_set = ShakeSpeareDataset(train=True)
            test_set = ShakeSpeareDataset(train=False)
        else:
            raise NotImplementedError('Check your dataset name.')
        
        return train_set, test_set

    
    def get_test_loader(self):
        # random seed 
        set_seed(self.seed)
        
        if self.dataset_name in ['mnist', 'fmnist', 'cifar10', 'cifar100', 'svhn']:
            dataset = self.test_dataset
            labels = np.array(dataset.targets.detach().cpu()).reshape(-1)
            n_samples = len(labels)
            n_classes = len(set(labels))
            
            # dictionary of labels map
            dct = {}
            for idx, label in enumerate(labels):
                dct[label] = [] if label not in dct else dct[label] + [idx]
            for i in range(n_classes):
                temp = np.random.permutation(dct[i])
                dct[i] = temp 
            
            # partitions 
            indices = []
            for class_id in range(n_classes):
                class_size = len(dct[class_id]) 
                indices += list(dct[class_id][:class_size]) 

            subset = torch.utils.data.Subset(dataset, indices)
            test_loader = DataLoader(subset, batch_size=self.batch_size, num_workers=8, pin_memory=True) 

            self.test_loader = test_loader 
            
            counts = [0] * n_classes
            for label in subset: 
                counts[label[1]] += 1
            print('test set:', counts, sum(counts), '\noverall sum: ', n_samples, end="\n") 
        
        elif self.dataset_name in ['sst']:
            return self.test_loader 
        
        elif self.dataset_name == 'shakespeare':
            print(f"{len(self.test_dataset)} test samples!")
            # test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8, pin_memory=True)
            # print(DataLoader(self.test_dataset, batch_size=256, num_workers=8, pin_memory=True))
            # print(len(DataLoader(self.test_dataset, batch_size=256, num_workers=8, pin_memory=True)))
            
            test_loader = DataLoader(self.test_dataset, batch_size=256, num_workers=8, pin_memory=True, drop_last=True)
            self.test_loader = test_loader 
        
        return self.test_loader 


    def get_train_loaders(self):
        # random seed
        set_seed(self.seed)
        
        # if self.dataset_name != 'fedisic':
        if self.dataset_name in ['mnist', 'fmnist', 'cifar10', 'cifar100', 'svhn']:
            dataset = self.train_dataset
            labels = np.array(dataset.targets.detach().cpu()).reshape(-1)
            n_samples = len(labels)
            n_classes = len(set(labels))
            
            # dictionary of labels map
            dct = {}
            for idx, label in enumerate(labels):
                dct[label] = [] if label not in dct else dct[label] + [idx]
            for i in range(n_classes):
                temp = np.random.permutation(dct[i])
                dct[i] = temp
        elif self.dataset_name in ['sst']:
            dataset = self.train_dataset
            label_to_num = {'very negative': 0, 'negative': 1, 'neutral': 2, 'positive': 3, 'very positive': 4}
            labels = [label_to_num[label] for label in dataset.label]
            labels = np.array(labels).reshape(-1)
            
            n_samples = len(labels)
            n_classes = len(set(labels))
            
            # dictionary of labels map
            dct = {}
            for idx, label in enumerate(labels):
                dct[label] = [] if label not in dct else dct[label] + [idx]
            for i in range(n_classes):
                temp = np.random.permutation(dct[i])
                dct[i] = temp
        elif self.dataset_name in ['shakespeare']:
            pass

        # probabilities 
        torch.set_printoptions(precision=3)
        probs = []
        
        if self.partition == 'homogeneous':
            for i in range(n_classes):
                probs.append([1.0 / self.n_participants] * self.n_participants) 
            print(f'Homogeneous partitioning of {n_samples} data points into {self.n_participants} participants :', probs) 
        elif 'noc' in self.partition:
            C = int(self.partition[3:])
            if (n_classes // C) == self.n_participants:
                probs = np.zeros((n_classes, self.n_participants))
                res = np.random.choice(list(set(labels)), (n_classes // C, C), replace=False)  # randomly assign C classes into (n_classes // C) participants
                for j in range(self.n_participants):
                    probs[res[j], j] = 1.0 
                probs = [list(p) for p in probs] 
            else:
                raise NotImplementedError("Non-overlapping Classes Partition [Mismatch with the number of classes and participants]")
            print(f'Non-overlapping class partitioning with #C={C} of {n_samples} data points into {self.n_participants} participants :', probs) 
        elif 'oc' in self.partition:
            C = int(self.partition[2:])
            lsts = [list(np.random.choice(range(n_classes), C, replace=False)) for _ in range(self.n_participants)]
            counts = dict(Counter(k for l in lsts for k in l))
            probs = [[0.0 for _ in range(self.n_participants)] for _ in range(n_classes)] # [[0.0] * n_classes for _ in range(self.n_participants)]
            
            for i in range(self.n_participants):
                for j in lsts[i]:
                    probs[j][i] = 1 / counts[j] 
        elif self.partition == 'class-based':
            if self.dataset_name == 'synthetic':
                if self.n_participants == 2:
                    probs.extend([[1.0, 0.0] if i < 2 else [0.0, 1.0] for i in range(n_classes)]) 
                elif self.n_participants == 4:
                    probs = [[1.0 if j == i else 0.0 for j in range(n_classes)] for i in range(n_classes)]
                else:
                    raise NotImplementedError() 
            else:
                raise NotImplementedError()
        elif self.partition == 'dirichlet':
            dirichlet_distribution = np.random.dirichlet([self.alpha] * self.n_participants, n_classes) 
            probs = dirichlet_distribution.tolist() 
        elif self.partition == 'mod-dirichlet': # modified-dirichlet
            if self.dataset_name == 'cifar-mnist':
                dirichlet_distribution1 = np.random.dirichlet([self.alpha] * (self.n_participants // 2), n_classes // 2) 
                dirichlet_distribution2 = np.random.dirichlet([self.alpha] * (self.n_participants // 2), n_classes // 2) 
                probs1 = dirichlet_distribution1.tolist() 
                probs2 = dirichlet_distribution2.tolist()
                probs = np.concatenate([
                    np.concatenate([probs1, np.zeros((n_classes//2, self.n_participants//2))], axis=1), 
                    np.concatenate([np.zeros((n_classes//2, self.n_participants//2)), probs2], axis=1)
                ])
            else:
                raise NotImplementedError()
        elif self.partition == 'imbalanced':
            if self.n_participants == 10:
                for i in range(n_classes): 
                    rho = 0.15 # config.rho 
                    n = 6 # config.alloc_n 
                    major_allocation = [rho] * n 
                    remaining_allocation = [(1 - sum(major_allocation)) / (self.n_participants-n)] * (self.n_participants-n) 
                    prob = major_allocation + remaining_allocation 
                    probs.append(prob)
            else:
                raise NotImplementedError()
        elif self.partition == 'imbalanced2':
            if self.n_participants == 10:
                for i in range(n_classes): 
                    rho = 0.4 # config.rho 
                    n = 2 # config.alloc_n 
                    major_allocation = [rho] * n 
                    remaining_allocation = [(1 - sum(major_allocation)) / (self.n_participants-n)] * (self.n_participants-n) 
                    prob = major_allocation + remaining_allocation 
                    probs.append(prob)
            else:
                raise NotImplementedError()
        elif self.partition in ['natural', 'natural-iid', 'natural-noniid']:
            print("#####"*20)
            print(f"No need to split for FedISIC/Shakespeare! {self.partition}")
            print("#####"*20)
        else: 
            raise NotImplementedError() 
        
        print(probs, end="\n\n") 
        
        # if self.dataset_name != 'fedisic':
        if self.dataset_name in ['mnist', 'fmnist', 'cifar10', 'cifar100', 'svhn']:
            print("*****"*20)
            print(f"{self.dataset_name} dataset")
            print("*****"*20)
            
            # partitions 
            indices = {i: [] for i in range(self.n_participants)} 
            for class_id, prob_list in enumerate(probs):
                from_id = 0
                class_size = len(dct[class_id]) 
                for participant_id, prob in enumerate(prob_list):
                    to_id = int(from_id + prob * class_size)
                    indices[participant_id] += list(dct[class_id][from_id:to_id])
                    from_id = to_id

            self.shard_sizes = [len(indices[i]) for i in range(self.n_participants)]
            t_loaders = [DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8, pin_memory=True, sampler=SubsetRandomSampler(indices[i])) for i in range(self.n_participants)]
        
        elif self.dataset_name in ['shakespeare']:
            print("*****"*20)
            print(f"{self.dataset_name} dataset!")
            dict_users = self.train_dataset.get_client_dic()
            
            print(len(dict_users), dict_users.keys()) 
            print(len(dict_users[0]))
            print(len(dict_users[1]))
            # print(len(dict_users[2]))
            
            n_participants = len(dict_users)
            print(f"{n_participants} participants.")
            
            # print(DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True, sampler=SubsetRandomSampler(list(dict_users[1]))))
            # print(len(DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True, sampler=SubsetRandomSampler(list(dict_users[1])))))
            
            t_loaders = [DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True, drop_last=True, sampler=SubsetRandomSampler(list(dict_users[i]))) for i in range(n_participants)]
        
        elif self.dataset_name in ['sst']:
            print("*****"*20)
            print(f"{self.dataset_name} dataset")
            print("*****"*20)
            
            # partitions 
            indices = {i: [] for i in range(self.n_participants)} 
            for class_id, prob_list in enumerate(probs):
                from_id = 0
                class_size = len(dct[class_id]) 
                for participant_id, prob in enumerate(prob_list):
                    to_id = int(from_id + prob * class_size)
                    indices[participant_id] += list(dct[class_id][from_id:to_id])
                    from_id = to_id

            self.shard_sizes = [len(indices[i]) for i in range(self.n_participants)]
            datasets = []
            for indices_i in indices:
                examples = []
                for i in indices[indices_i]:
                    examples.append(self.train_dataset[i])
                datasets.append(TextDataset(examples, self.args['fields']))
            
            t_loaders = [BucketIterator(train_dataset, batch_size=self.batch_size, device=self.device, sort_key=lambda x: len(x.text), train=True) for train_dataset in datasets]
        
        else:
            raise NotImplementedError()
            
        self.train_loaders = t_loaders
        
        # statistics 
        if self.dataset_name not in ['fedisic', 'shakespeare']:
            participant_class_counts = get_data_stats(labels, indices) 
            print(f"Data statistics:", participant_class_counts) 
            
            path = f'partitions/{self.dataset_name}'
            if not os.path.exists(path):
                mkdir(path)
            
            if self.partition == 'dirichlet':
                specific_path = f'{path}/{self.dataset_name}_{self.partition}_{self.alpha}_parties{self.n_participants}_seed{self.seed}'
            else: 
                specific_path = f'{path}/{self.dataset_name}_{self.partition}_parties{self.n_participants}_seed{self.seed}'
            if os.path.exists(f'{specific_path}.npz'):
                print(f'The partition is already there: {specific_path}!')
            else:
                print('Directory for the partition:', specific_path) 
                np.savez(specific_path, client_indices=indices, participant_class_counts=participant_class_counts)

        return t_loaders


def get_pad_to_min_len_fn(min_length):
    def pad_to_min_len(batch, vocab, min_length=min_length):
        pad_idx = vocab.stoi['<pad>']
        for idx, ex in enumerate(batch):
            if len(ex) < min_length:
                batch[idx] = ex + [pad_idx] * (min_length - len(ex))
        return batch
    return pad_to_min_len