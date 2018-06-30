from __future__ import division
import os 
import sys
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from convsparse_net import LISTAConvDictADMM
import common
from common import save_train, load_train, clean
from common import get_criterion, init_model_dir
from torch.utils.data import DataLoader
import arguments
from  mnist_datasets import get_train_valid_loader, get_test_loader
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt

USE_CUDA = torch.cuda.is_available() and False
FILE_PATH = os.path.dirname(os.path.abspath(__file__))

def add_noise(imgs, noise):
    noise = common.normalize(noise)
    return common.gaussian(imgs, is_training=True, mean=0, stddev=noise)

def step(model, img, img_n, optimizer=None, criterion=None):
    if optimizer is not None: 
       optimizer.zero_grad()
    output, sc_in = model(img_n)
    _, sc_target = model(img)
    if criterion is not None:
        loss = criterion(output, img, sc_in, sc_target.data)
        if optimizer:
            loss.backward()
            optimizer.step()
        return loss, output.cpu()
    return output.cpu()

def maybe_save_model(model, opt, schd, epoch, save_path, curr_val, other_values):
    path = ''
    def no_other_values(other_values):
        return len(other_values) == 0
    if no_other_values(other_values) or curr_val <  min(other_values):
        print('saving model...')
        path = save_train(save_path, model, opt, schd, epoch)
        clean(save_path, save_count=10)
    return path

def run_valid(model, data_loader, criterion, logdir, noise):
    loss = 0
    for img, _ in data_loader:
        if USE_CUDA:
            img = img.cuda()
        img_n = add_noise(img, noise) 

        _loss, _out = step(model, img, img_n, criterion=criterion)
        loss += _loss.data

    _, output = step(model, img, img_n, criterion=criterion)
    np.savez(os.path.join(logdir, 'images'), IN=img.data.cpu().numpy(),
        OUT=output.data.cpu().numpy(), NOISE=img_n.data.cpu().numpy())

    return float(loss) / len(data_loader)

def train(model, args):
    
    optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)
    criterion = get_criterion(use_cuda=True, sc_factor=args['sc_factor'])
    train_loader, valid_loader = get_train_valid_loader(batch_size=args['batch_size'], valid_size=0.01)
    print(args)

    print('train count: {}  valid count: {}'.format(len(train_loader), len(valid_loader)))

    if args['load_path'] != '':
        ld_p = args['load_path']
        print('loading from %s'%ld_p)
        load_train(ld_p, model, optimizer, scheduler)        
        print('Done!')
        
    _train_loss = []
    _valid_loss = []
    running_loss = 0
    valid_every = 10#int(0.1 * len(train_loader))

    itr = 0
    for e in range(args['epoch']):
        print('Epoch number {}'.format(e))
        for img, _ in train_loader:
            itr += 1

            if USE_CUDA:
                img = img.cuda()

            img_n = add_noise(img, args['noise'])
            
            _loss, _ = step(model, img, img_n, optimizer, criterion=criterion)
            running_loss += float(_loss)

            if itr % valid_every == 0:
                _train_loss.append(running_loss / valid_every)
                _v_loss = run_valid(model, valid_loader,
                        criterion, args['save_dir'], args['noise'])
                scheduler.step(_v_loss)
                model_path = maybe_save_model(model, optimizer,
                        scheduler, e, args['save_dir'],
                        _v_loss, _valid_loss)
                _valid_loss.append(_v_loss)
                print("epoch {} train loss: {} valid loss: {}".format(e,
                    running_loss / valid_every, _v_loss))
                running_loss = 0
                return model_path, _valid_loss[-1]

def build_model(args):
    model = LISTAConvDictADMM(
        num_input_channels=args['num_input_channels'],
        num_output_channels=args['num_output_channels'],
        kc=args['kc'], 
        ks=args['ks'],
        ista_iters=args['ista_iters'],
        iter_weight_share=args['iter_weight_share'],
    )
    if USE_CUDA:
        model = model.cuda()
    return model

def run(args_file):
    args = arguments.load_args(args_file)
    log_dir, save_dir = init_model_dir(args['train_args']['log_dir'], args['train_args']['name'])
    arguments.logdictargs(os.path.join(log_dir, 'params.json'), args)
    args['train_args']['save_dir'] = save_dir
    args['train_args']['log_dir'] = log_dir
    model = build_model(args['model_args'])
    model_path, valid_loss = train(model, args['train_args'])

    args['test_args']['load_path'] = model_path
    args['test_args']['embedd_model_path'] = model_path
    args['train_args']['final_loss'] = valid_loss

    args_fp = os.path.join(log_dir, 'params.json')
    arguments.logdictargs(args_fp, args)
    return args_fp


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--arg_file', default='')
    arg_file = parser.parse_args().arg_file

    run(arg_file)
