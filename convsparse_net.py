from __future__ import print_function
from numpy.random import normal
from numpy.linalg import svd
from math import sqrt
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.init
from torch.nn import Parameter
import numpy as np
from common import conv as dp_conv
from common import flip


class LISTAConvDictMNISTSSL(nn.Module):
    """Run use sparse prior with lista like model for SSL mnist task """
    num_of_classes = 10

    def __init__(self, embedding_model, embedding_size, hidden_size, downsample=2):

        super(LISTAConvDictMNISTSSL, self).__init__()

        self.embedding_model = embedding_model
        self.downsample_by = downsample
        self.input_dowsampeled_embedding_size = embedding_size // (self.downsample_by ** 2)

        self.classifier_model = nn.Sequential(
           nn.Linear(self.input_dowsampeled_embedding_size, hidden_size[0]),
           nn.ReLU(),
           nn.Linear(hidden_size[0], hidden_size[1]),
           nn.ReLU(),
           nn.Linear(hidden_size[1], self.num_of_classes)
        )

    #TODO(hillel): for training we need  2 diffrent models for training and infrence...
    def forward(self, inputs):
        _, embedding = self.embedding_model(inputs)
        embedding_flatten = F.max_pool2d(embedding, 2).view(embedding.shape[0], -1)
        logits = self.classifier_model(embedding_flatten)
        return logits, embedding_flatten



    

class LISTAConvDictADMM(nn.Module):
    """
    LISTA ConvDict encoder based on paper:
    https://arxiv.org/pdf/1711.00328.pdf
    """
    def __init__(self, num_input_channels=3, num_output_channels=3,
                 kc=64, ks=7, ista_iters=3, iter_weight_share=True,
                 pad='reflection', norm_weights=True):

        super(LISTAConvDictADMM, self).__init__()
        if iter_weight_share == False:
            raise NotImplementedError('untied weights is not implemented yet...')
        self._ista_iters = ista_iters
        self.softthrsh = SoftshrinkTrainable(Parameter(0.1 * torch.ones(1, kc), requires_grad=True))

        self.encode_conv = dp_conv(
            num_input_channels,
            kc,
            ks,
            stride=1,
            bias=False,
            pad=pad
        )

        self.decode_conv0 = dp_conv(
            kc,
            num_input_channels,
            ks,
            stride=1,
            bias=False,
            pad=pad
        )


        self.decode_conv1 = dp_conv(
            kc,
            num_input_channels,
            ks,
            stride=1,
            bias=False,
            pad=pad
        )

        self.mu = Parameter(0.6 * torch.ones(1), requires_grad=True)

       # self._init_vars()

    def _init_vars(self): 
        ###################################
        # Better  Results without this inilization.
        ##################################
        wd = self.decode_conv[1].weight.data
        wd = F.normalize(F.normalize(wd, p=2, dim=2), p=2, dim=3)
        self.decode_conv[1].weight.data = wd
        self.encode_conv[1].weight.data = we

    def forward_enc(self, inputs):
        sc = self.softthrsh(self.encode_conv(inputs))

        for step in range(self._ista_iters):
           
            _inputs = self.mu * inputs + (1 - self.mu) * self.decode_conv0(sc)
            sc_residual = self.encode_conv(
               _inputs - self.decode_conv1(sc)
               )
            sc = self.softthrsh(sc + sc_residual)
        return sc

    def forward_dec(self, sc):
        return self.decode_conv0(sc)

    def forward(self, inputs):
        sc = self.forward_enc(inputs)
        outputs = self.forward_dec(sc)
        return outputs, sc

class SoftshrinkTrainable(nn.Module):
    """
    Learn threshold (lambda)
    """
    grads = {'thrsh': 0}

    def __init__(self, _lambd):
        super(SoftshrinkTrainable, self).__init__()
        self._lambd = _lambd
#        self._lambd.register_hook(print)

    def forward(self, inputs):
        _lambd = self._lambd.clamp(0)
        pos = inputs - _lambd.unsqueeze(2).unsqueeze(3).expand_as(inputs)
        neg = (-1) * inputs - _lambd.unsqueeze(2).unsqueeze(3).expand_as(inputs)
        return pos.clamp(min=0) - neg.clamp(min=0)
