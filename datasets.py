from __future__ import division
from random import shuffle
import torch
from torch.autograd import Variable
from torchvision import datasets
from torch.utils.data import TensorDataset


def split_mnist(path, ratio=0.1):
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = MNIST(path, train=True, download=True, transform=transform)

    label_per_class = ratio * len(mnist)) / 10
    mnist_dict = dict([(i, [])for i in range(10)])

    for data, label in mnist:
        mnist_dict[label].append(data)
    for v in mnist_dict.itervalues():
        shuffle(v)
    
    labels = []
    l_data = []
    u_data = []

    for label, data in mnist_dict:
        l_data += data[:label_per_class]
        labels += ([label] * label_per_class)
        u_data += data[label_per_class:]
    
    perm = torch.randperm(len(label_per_class) * 10)
    l_data = torch.stack(l_data)[perm]
    labels = torch.stack(labels)[perm]
    u_data = torch.stack(u_data)[torch.randperm(len(u_data))]

    labeled = TensorDataset(l_data, labels)
    unlabeld = TensorDataset(u_data)

    
    
