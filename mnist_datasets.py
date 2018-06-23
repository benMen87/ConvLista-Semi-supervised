from __future__ import division
from random import shuffle
import torch
from torch.autograd import Variable
from torchvision import datasets
from torch.utils.data import TensorDataset

import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import common

DATA_PATH = '/data/dataset/mnist'
RANDOM_SEED = 10


def get_train_valid_loader(data_dir=DATA_PATH,
                           batch_size=32,
                           augment=True,
                           random_seed=RANDOM_SEED,
                           valid_size=0.1,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=False):


    # define transforms
    valid_transform = transforms.Compose([
            transforms.ToTensor(),
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    # load the dataset
    train_dataset = datasets.MNIST(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = datasets.MNIST(
        root=data_dir, train=True,
        download=True, transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return (train_loader, valid_loader)


def get_test_loader(data_dir,
                    batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=False):

    # transforms.Normalize(
    #    mean=[0.485, 0.456, 0.406],
    #    std=[0.229, 0.224, 0.225],
    #)

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = datasets.MNIST(
        root=data_dir, train=False,
        download=True, transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader

def semisup_mnist(lbl_cnt=3000, path=DATA_PATH):
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNIST(path, train=True, download=True, transform=transform)

    label_per_class = lbl_cnt // 10
    mnist_dict = dict([(i, [])for i in range(10)])

    for data, label in mnist:
        mnist_dict[int(label)].append(data)
    for v in mnist_dict.itervalues():
        shuffle(v)
    
    labels = []
    l_data = []
    u_data = []

    for label, data in mnist_dict.items():
        l_data += data[:label_per_class]
        labels += ([label] * label_per_class)
        u_data += data[label_per_class:]
    
    perm = torch.randperm(label_per_class * 10)
    l_data = torch.stack(l_data)[perm]
    labels = torch.Tensor(labels)[perm]
    u_data = torch.stack(u_data)[torch.randperm(len(u_data))]

    labeled = TensorDataset(l_data, labels)
    unlabeled = TensorDataset(u_data)

    dll = torch.utils.data.DataLoader(
        labeled, batch_size=64, shuffle=False
    )

    dlu = torch.utils.data.DataLoader(
        unlabeled, batch_size=64, shuffle=False
    )

    return (dll, dlu)

    
    
