from __future__ import division
import os 
import sys
import numpy as np
from itertools import cycle
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from convsparse_net import LISTAConvDictADMM, LISTAConvDictMNISTSSL
import common
from common import save_train, load_train, clean
from common import get_sup_criterion, get_unsup_criterion, init_model_dir
from torch.utils.data import DataLoader
import arguments
from  mnist_datasets import semisup_mnist, get_test_loader
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt

USE_CUDA = torch.cuda.is_available()
FILE_PATH = os.path.dirname(os.path.abspath(__file__))

def add_noise(imgs, noise):
    noise = common.normalize(noise)
    return common.gaussian(imgs, is_training=True, mean=0, stddev=noise)

def train_step(model, sup_criterion, unsup_criterion, noise_strength, x, y=None):
    """
    model - nn model to optimize
    x, y - input and its label
    u - unlabeld input
    """
    logits, embedding = model(x)


    x_n = add_noise(x, noise_strength)
    logits_n, embedding_n = model(x_n)
    loss = unsup_criterion([logits, embedding], [logits_n.data, embedding_n.data])

    if y is not None:
        loss += sup_criterion(logits, y)

    #loss.backward()
    #optimizer.step()

    return loss, output.cpu()

def maybe_save_model(model, opt, schd, epoch, save_path, curr_val, other_values):
    path = ''
    def no_other_values(other_values):
        return len(other_values) == 0
    if no_other_values(other_values) or curr_val <  min(other_values):
        print('saving model...')
        path = save_train(save_path, model, opt, schd, epoch)
        print("new checkpoint at %s"%path)
        clean(save_path, save_count=10)
    return path

def run_valid(model, data_loader, criterion, logdir, noise):
    loss = 0
    for x, y in data_loader:

        if USE_CUDA:
            x = x.cuda()
            y = y.cuda()

        _logits, _ = model(x) 
        loss = criterion(_logits, target)
        loss += _loss.data
    return float(loss) / len(data_loader)

def train(model, args):
    
    optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)
    sup_criterion = get_sup_criterion(use_cuda=USE_CUDA)
    unsup_criterion = get_unsup_criterion(args['unsup_factors'])

    labeled_loader, unlabeled_loader = semisup_mnist(batch_size=args['batch_size'], lbl_cnt=args["label_count"])
    valid_loader = get_test_loader()

    print(args)

    print('labeld count: {}  unlabeled count: {}'.format(len(labeled_loader), len(unlabeled_loader)))

    if args['load_path'] != '':
        ld_p = args['load_path']
        print('loading from %s'%ld_p)
        load_train(ld_p, model, optimizer, scheduler)        
        print('Done!')
        
    _train_label_loss = []
    _valid_unlabel_loss = []

    running_label_loss = 0
    running_unlabel_loss = 0
    valid_every = int(0.1 * len(labeled_loader)) #TODO(hillel): better idea than using labeled data as valid as well?

    itr = 0
    for e in range(args['epoch']):
        print('Epoch number {}'.format(e))
        for (x, y), u in zip(cycle(labeled_loader), unlabeled_loader):
            itr += 1

            if USE_CUDA:
                x = x.cuda()
                y = y.cuda()
                u = u.cuda()


            optimizer.zero_grad()

            label_loss = train_step(model, sup_criterion, unsup_criterion, args['noise'], x, y)
            unlabel_loss = train_step(model, sup_criterion, unsup_criterion, args['noise'], u)

            _loss = label_loss + unlabel_loss
           
            _loss.backward()
            optimizer.step()

            running_label_loss += float(label_loss_loss)
            running_unlabel_loss += float(label_unloss_loss)

            if itr % valid_every == 0:
                _train_label_loss.append(running_label_loss / valid_every)
                _train_unlabel_loss.append(running_unlabel_loss / valid_every)

                _v_loss = run_valid(model, valid_loader,
                        sup_criterion, args['save_dir'], args['noise'])
                scheduler.step(_v_loss)
                _model_path = maybe_save_model(model, optimizer,
                        scheduler, e, args['save_dir'],
                        _v_loss, _valid_loss)
                if _model_path != '':
                    model_path = _model_path
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
    model = LISTAConvDictMNISTSSL(
        embedding_model=model,
        embedding_size=64,
        hidden_size=1000
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
    args['test_args']['log_dir'] = log_dir

    args_fp = os.path.join(log_dir, 'params.json')
    arguments.logdictargs(args_fp, args)
    print("writing {} to  {}".format(args_fp, args))
    return args_fp


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--arg_file', default='./default_ssl_e2e_args.json')
    arg_file = parser.parse_args().arg_file

    run(arg_file)
