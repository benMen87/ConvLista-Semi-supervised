from __future__ import print_function
import pickle
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable


DATA_PATH = '/data/dataset/mnist'
RANDOM_SEED = 10


def mnist_cls_size(): return 10

def get_train_loaders(
    labeled_size,
    valid_size,
    batch_size=32,
    augment=True,
    random_seed=RANDOM_SEED,
    show_sample=False,
    num_workers=1,
    pin_memory=False,
    data_dir=DATA_PATH):

    print('batch size is {}'.format(batch_size))
    transform=transforms.Compose([transforms.ToTensor(),
                                #  transforms.Normalize((0.1307,), (0.3081,))
                                 ])
    trainset_original = datasets.MNIST(data_dir, train=True, download=True,
                                 transform=transform)
    train_label_index = []
    valid_label_index = []
    cls_labeled_size = labeled_size // mnist_cls_size()
    cls_valid_size = valid_size // mnist_cls_size()

    for i in range(mnist_cls_size()):
       train_label_list = trainset_original.train_labels.numpy()
       label_index = np.where(train_label_list == i)[0]
       np.random.shuffle(label_index)
       label_subindex = list(label_index[:cls_labeled_size])
       valid_subindex = list(label_index[cls_labeled_size: cls_valid_size + cls_labeled_size])
       train_label_index += label_subindex
       valid_label_index += valid_subindex

    mnist_size = len(trainset_original)
    occupied_index = train_label_index + valid_label_index
    train_unlabel_index = [i for i in range(mnist_size) if i not in occupied_index]
    if len(train_unlabel_index) + len(occupied_index) != mnist_size:
        raise ValueError("numbers don't add up :(")
    # define transforms
    valid_transform = transforms.Compose([
            transforms.ToTensor(),
    ])
    if augment:
        train_transform = transforms.Compose([
            #transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
           #transforms.RandomCrop(28, padding=4),
           transforms.ToTensor(),
        ])

   
    train_sampler = SubsetRandomSampler(train_label_index)
    train_unlabel_sampler = SubsetRandomSampler(train_unlabel_index)
    valid_sampler = SubsetRandomSampler(valid_label_index)

    trainset_new = datasets.MNIST(root=data_dir, download=True, transform=train_transform)
    validset = datasets.MNIST(root=data_dir, download=True, transform=train_transform)
    trainset_new_unl = datasets.MNIST(root=data_dir, download=True, transform=train_transform)

    train_loader = torch.utils.data.DataLoader(
        trainset_new, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory,
    )

    valid_loader = torch.utils.data.DataLoader(
        validset, batch_size=batch_size,
        sampler=valid_sampler, num_workers=num_workers, pin_memory=pin_memory,
    )

    train_unl_loader = torch.utils.data.DataLoader(
        trainset_new_unl, batch_size=batch_size,
        sampler=train_unlabel_sampler, num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, train_unl_loader, valid_loader