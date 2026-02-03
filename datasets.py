import os
import torch
import numpy as np 
from PIL import Image

from torch.utils.data import Dataset
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, SVHN, FashionMNIST

from medmnist import PathMNIST


# language datasets import 
import json
from collections import defaultdict
from utils import word_to_indices, letter_to_vec


class CustomDataset(Dataset):
    """
    Custom Dataset for single datasets
    """
    def __init__(self, data, targets, device=None, transform=None):
        self.data = data.to(device)
        self.targets = targets.to(device)
        self.count = len(data)
        self.device = device
        self.transform = transform 
    
    def __len__(self):
        return self.count
    
    def __getitem__(self, index: int):
        if self.transform:
            return self.transform(self.data[index]), self.targets[index] 
        
        return self.data[index], self.targets[index] 
    

class CustomCombinedDataset(Dataset):
    def __init__(self, dataset1, dataset2, device=None, transform1=None, transform2=None):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        
        labels1 = dataset1.targets
        labels2 = dataset2.targets + len(torch.unique(labels1))  # shift CIFAR-10 labels 
        self.targets = torch.cat((labels1, labels2)).to(device)
        
        self.count = len(self.targets)
        self.device = device
        self.transform1 = transform1
        self.transform2 = transform2
    
    def __len__(self):
        # The total length is the sum of both datasets
        return self.count 
    
    def __getitem__(self, idx):
        if idx < len(self.dataset1): 
            img, _ = self.dataset1[idx]
            if self.transform1:
                return self.transform1(img), self.targets[idx] 
        else:
            img, _ = self.dataset2[idx - len(self.dataset1)]
            if self.transform2:
                return self.transform2(img), self.targets[idx]
        return img, self.targets[idx] 


class FastMNIST(MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.data = self.data.unsqueeze(1).float().div(255)         
        self.targets = self.targets.long() 
        
        # normalization 
        self.data = self.data.sub_(self.data.mean()).div_(self.data.std()) 
        self.data, self.targets = self.data, self.targets 
        
        print(f"MNIST: data shape {self.data.shape}, targets shape {self.targets.shape}") 
    
    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        return img, target


class FastFMNIST(FashionMNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.data = self.data.unsqueeze(1).float().div(255)         
        self.targets = self.targets.long() 
        
        # normalization 
        self.data = self.data.sub_(self.data.mean()).div_(self.data.std()) 
        self.data, self.targets = self.data, self.targets 
        
        print(f"FashionMNIST: data shape {self.data.shape}, targets shape {self.targets.shape}") 
    
    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        return img, target


class FastRGBMNIST(MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.data = self.data.unsqueeze(1).float().div(255) 
        
        # Uncomment if for 32*32 MNIST 
        from torch.nn import ZeroPad2d 
        pad = ZeroPad2d(2) 
        self.data = torch.stack([pad(sample.data) for sample in self.data]) 
        
        self.targets = self.targets.long() 
        
        # normalization 
        self.data = self.data.sub_(self.data.mean()).div_(self.data.std()) 
        self.data, self.targets = self.data, self.targets 
        
        print(f"MNIST: data shape {self.data.shape}, targets shape {self.targets.shape}") 
    
    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        img = img.expand(3, -1, -1)  # Expands to [3, 32, 32]

        return img, target


class FastCIFAR10(CIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        from torch import from_numpy
        self.data = from_numpy(self.data)       
        self.data = self.data.float().div(255)
        self.data = self.data.permute(0, 3, 1, 2)
        self.targets = torch.Tensor(self.targets).long()
        
        # normalization 
        for i, (mean, std) in enumerate(zip((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))):
            self.data[:,i].sub_(mean).div_(std)

        self.data, self.targets = self.data, self.targets 
        
        print(f"CIFAR10: data shape {self.data.shape}, targets shape {self.targets.shape}") 
    
    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        return img, target 


class FastCIFAR100(CIFAR100):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        from torch import from_numpy
        self.data = from_numpy(self.data)
        self.data = self.data.float().div(255)
        self.data = self.data.permute(0, 3, 1, 2)
        self.targets = torch.Tensor(self.targets).long()
        
        # normalization 
        for i, (mean, std) in enumerate(zip((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))):
            self.data[:,i].sub_(mean).div_(std)

        self.data, self.targets = self.data, self.targets 
        
        print(f"CIFAR100: data shape {self.data.shape}, targets shape {self.targets.shape}") 
    
    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        return img, target 
    
    
class FastPathMNIST(PathMNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        from torch import from_numpy
        self.data = from_numpy(self.imgs)
        self.data = self.data.float().div(255) 
        self.data = self.data.permute(0, 3, 1, 2)
        self.targets = torch.Tensor(self.labels).long()
        
        # normalization 
        self.data = self.data.sub_(self.data.mean()).div_(self.data.std()) 
        self.data, self.targets = self.data, self.targets 
        
        print(f"PathMNIST: data shape {self.data.shape}, targets shape {self.targets.shape}") 
        
    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        return img, target 


class FastSVHN(SVHN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        from torch import from_numpy
        self.data = from_numpy(self.data)
        self.data = self.data.float().div(255)
        # self.data = self.data.permute(0, 3, 1, 2)
        self.targets = torch.Tensor(self.labels).long()
        
        # normalization 
        for i, (mean, std) in enumerate(zip((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))):
            self.data[:,i].sub_(mean).div_(std)

        self.data, self.targets = self.data, self.targets 
        
        print(f"SVHN: data shape {self.data.shape}, targets shape {self.targets.shape}") 
    
    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        return img, target 
    

class ShakeSpeareDataset(Dataset):
    # 380353 training data points 
    # 41393 test data points 
    
    # Non-IID [160 users]
    # 731291 train data points 
    # 68689 test data points 
    def __init__(self, train=True):
        super(ShakeSpeareDataset, self).__init__()
        
        train_clients, train_groups, train_data_temp, test_data_temp = read_data("data/leaf-datasets/shakespeare/data/train", "data/leaf-datasets/shakespeare/data/test")
        print(len(train_data_temp), len(test_data_temp))
        self.train = train 
        
        if self.train:
            self.dic_users = {}
            train_data_x, train_data_y = [], []
            
            for i in range(len(train_clients)):
                self.dic_users[i] = set()
                l = len(train_data_x)
                cur_x = train_data_temp[train_clients[i]]['x']
                cur_y = train_data_temp[train_clients[i]]['y']
                
                for j in range(len(cur_x)):
                    self.dic_users[i].add(j+l)
                    train_data_x.append(cur_x[j])
                    train_data_y.append(cur_y[j])
            
            self.data = train_data_x
            self.label = train_data_y
            print(len(train_data_x), len(train_data_y))
            
            print("unique labels", len(list(set(train_data_y))))
            print(list(set(train_data_y)))
        
        else:
            test_data_x, test_data_y = [], []
            
            for i in range(len(train_clients)):
                cur_x = test_data_temp[train_clients[i]]['x']
                cur_y = test_data_temp[train_clients[i]]['y']
                
                for j in range(len(cur_x)):
                    test_data_x.append(cur_x[j])
                    test_data_y.append(cur_y[j])
            
            self.data = test_data_x
            self.label = test_data_y
            print(len(test_data_x), len(test_data_y))
            
            print("unique labels", len(list(set(test_data_y))))
            print(list(set(test_data_y)))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sentence, target = self.data[index], self.label[index]
        indices = word_to_indices(sentence)
        target = letter_to_vec(target)
        
        indices = torch.LongTensor(np.array(indices))
        return indices, target 
    
    def get_client_dic(self):
        if self.train:
            return self.dic_users
        else:
            exit("The test dataset does not have dic_users!")


def read_dir(data_dir):
    clients, groups = [], [] 
    data = defaultdict(lambda: None) 
    
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])
    
    clients = list(sorted(data.keys()))
    return clients, groups, data 


def read_data(train_data_dir, test_data_dir):
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)
    
    assert train_clients == test_clients
    assert train_groups == test_groups
    
    return train_clients, train_groups, train_data, test_data 