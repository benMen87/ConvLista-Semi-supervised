from __future__ import division
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np


def to_np(_x): return _x.data.cpu().numpy()

def I(_x): return _x

def normalize(_x, _val=255): return _x / _val

def nhwc_to_nchw(_x, keep_dims=True): 
    if len(_x.shape) == 3 and (_x.shape[-1] == 1 or _x.shape[-1] == 3): #unsqueeze N dim
        _x = _x[None, ...]
    elif len(_x.shape) == 3: #unsqueezed C dim
        _x = _x[...,None]
    elif len(_x.shape) == 2:  #unsqueeze N and C dim
        _x = _x[None,:,:,None]
    return np.transpose(_x, (0, 3, 1, 2))

def get_unique_name(path):
    idx = 1
    _path = path
    while os.path.isdir(_path):
        _path = '{}_{}'.format(path, idx)
        idx += 1
    return _path

def init_model_dir(path, name):
    full_path = os.path.join(path, name)
    if not os.path.isdir(path):
        os.mkdir(path)
    else:
        full_path = get_unique_name(full_path)
    os.mkdir(full_path)
    os.mkdir(os.path.join(full_path, 'saved'))
    return full_path, os.path.join(full_path, 'saved')

def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    inds = tuple(slice(None, None) if i != dim
             else x.new(torch.arange(x.size(i)-1, -1, -1).tolist()).long()
             for i in range(x.dim()))
    return x[inds]

def bn(num_features):
    return nn.BatchNorm2d(num_features)

def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    downsampler = None

    if stride != 1 and downsample_mode != 'stride':
        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
           downsampler = nn.MaxPool2d(stride, stride)
        else:
            assert False
        stride = 1
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection' and False:
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)
    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)

def gaussian(ins, is_training, mean, stddev):
    if is_training:
        noise = stddev * torch.randn_like(ins) + mean
        return ins + noise
    return ins

#TODO(hillel): this is dangrouse NO default factor val!!!
def get_criterion(use_cuda=True, sc_factor=0.1):

    l1 = nn.L1Loss()
    l2 = nn.MSELoss()

    if use_cuda:
        l2 = l2.cuda()
        l1 = l1.cuda()
    
    def total_loss(inputs, target, sc_in, sc_tar):
        sc_in_embd = sc_in.mean(-1).mean(-1)
        sc_tar_embd = sc_tar.mean(-1).mean(-1)
        return l1(inputs, target) * (1 - sc_factor) + l2(sc_in_embd, sc_tar_embd) * (sc_factor)
    return total_loss


def psnr(im, recon, verbose=False):
    im.shape
    im = np.squeeze(im)
    recon = np.squeeze(recon)
    MSE = np.sum((im - recon)**2) / (im.shape[0] * im.shape[1])
    MAX = np.max(im)
    PSNR = 10 * np.log10(MAX ** 2 / MSE)
    if verbose:
        print('PSNR %f'%PSNR)
    return PSNR

def clean(save_path, save_count=10):
    import glob

    l = glob.glob(save_path)

    if len(l) < save_count:
        return 
    l.sort(key=os.path.getmtime) 
    for f in l[:-save_count]:
        print('removing', f)
        os.remove(f)
    
def save_train(path, model, optimizer, schedular=None, epoch=None):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    #TODO(hillel): fix this so we can save schedular state
    #if schedular is not None:
    #    state['schedular'] = schedular.state_dict()
    if epoch is not None:
        state['epoch'] = epoch
    torch.save(state, os.path.join(path, 'epoch_{}'.format(epoch)))
    return os.path.join(path, 'epoch_{}'.format(epoch))

def load_train(path, model, optimizer, schedular=None):
    state = torch.load(path)
    model.load_state_dict(state['model'])
    if 'optimizer' in state:
        optimizer.load_state_dict(state['optimizer'])
    else:
        print('Optimizer not inilized since no data for it exists in supplied path')
    if schedular is not None:
        if 'schedular' in state:
            schedular.load_state_dict(state['schedular'])
        else:
            print('Schedular not inilized since no data for it exists in supplied path')
    if 'epoch' in state:
        e = state['epoch']
    else: 
        e = 0
    return e

def save_eval(path, model):
    torch.save(model.state_dict(), path)

def load_eval(path, model):
    _model = torch.load(path)
    if 'model' in _model.keys():
        _model = _model['model']
    model.load_state_dict(_model)
    model.eval()
