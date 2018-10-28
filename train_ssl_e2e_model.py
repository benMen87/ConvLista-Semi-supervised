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
from common import save_train, load_train, load_eval, clean, _pprint
from common import get_sup_criterion, get_unsup_criterion, init_model_dir
import arguments
from  mnist_datasets import get_test_loader
from  split_mnist import get_train_loaders
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

    logits, embedding, reconstructed = model(x)

    x_n = add_noise(x, noise_strength)
    logits_n, embedding_n, reconstructed_n = model(x_n)
    np.savez('debug', X=x, X_n=x_n)
   # loss = unsup_criterion([reconstructed, reconstructed_n], [x, x])
    loss_unsup = unsup_criterion(
            [logits_n,        reconstructed_n,        embedding_n],
            [logits.detach(), reconstructed.detach(), embedding.detach()]
    )

    loss_sup = None
    if y is not None:
        loss_sup = sup_criterion(logits_n, y)

    return loss_unsup, loss_sup #, output.cpu()

def maybe_save_model(model, opt, schd, epoch, save_path, curr_val, other_values, old_path=''):
    path = old_path
    def no_other_values(other_values):
        return len(other_values) == 0

    if no_other_values(other_values) or curr_val <  min(other_values):
        print('saving model...')
        path = save_train(save_path, model, opt, schd, epoch)
        print("new checkpoint at %s"%path)
        clean(save_path, save_count=10)
    return path

def run_valid(model, data_loader, criterion):
    loss = 0
    acc = 0
    _iter = 0
    for x, y in data_loader:
        _iter += 1
        if USE_CUDA:
            x = x.cuda()
            y = y.cuda()
        _logits, _, _ = model(x) 
        _loss = criterion(_logits, y)
        loss += _loss.data
        acc += float(float((_logits.argmax(dim=1) == y).sum()) / _logits.shape[0])
    return float(loss) / len(data_loader), acc / len(data_loader)

def train(model, args):

    optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)
    sup_criterion = get_sup_criterion(use_cuda=USE_CUDA)
    unsup_criterion = get_unsup_criterion(args['unsup_factors'])

    labeled_loader, unlabeled_loader, valid_loader =\
    get_train_loaders(labeled_size=args["label_count"], valid_size=5000, batch_size=args['batch_size'], pin_memory=USE_CUDA)

    print('Running Train\ntrain args:\n')
    _pprint(args)

    print('labeld count: {}  unlabeled count: {}'.\
        format(len(labeled_loader), len(unlabeled_loader)))

    if args['load_path'] != '':
        ld_p = args['load_path']
        print('loading from %s'%ld_p)
        load_train(ld_p, model, optimizer, scheduler)
        print('Done!')

    _train_label_loss = []
    _valid_loss = []
    _train_unlabel_loss = []
    _model_path = ''

    running_label_loss = 0
    running_unlabel_loss = 0
    valid_every =\
     int(0.1 * (len(labeled_loader) + len(unlabeled_loader)))

    itr = 0
    unsupervised_epochs = args['unsupervised_epochs']
    for e in range(args['epoch']):
        print('Epoch number {}'.format(e))
        for (x, y), (u, _) in zip(cycle(labeled_loader), unlabeled_loader):
            itr += 1

            if USE_CUDA:
                x = x.cuda()
                y = y.cuda()
                u = u.cuda()

            optimizer.zero_grad()

            if e < unsupervised_epochs:
                y = None

            ll_unsup, loss_sup =\
                    train_step(model, sup_criterion, unsup_criterion, args['noise'], x, y)
            ul_unsup, _ =\
                     train_step(model, sup_criterion, unsup_criterion, args['noise'], u)

            loss_unsup = 0.5 * (ll_unsup + ul_unsup)
            _loss = loss_unsup + loss_sup

            _loss.backward()
            optimizer.step()

            running_label_loss += float(loss_sup)
            running_unlabel_loss += float(loss_unsup)

            if itr % valid_every == 0:
                _train_label_loss.append(running_label_loss / valid_every)
                _train_unlabel_loss.append(running_unlabel_loss / valid_every)

                _v_loss, acc = run_valid(
                    model,
                    valid_loader,
                    sup_criterion,
                )
                scheduler.step(_v_loss)

                _model_path = maybe_save_model(model, optimizer,
                        scheduler, e, args['save_dir'],
                        _v_loss, _valid_loss, _model_path)

                _valid_loss.append(_v_loss)

                if e >= unsupervised_epochs:
                    line = "epoch ssl {}:{} train loss labeld: {} "
                    line += "train unlabeld loss: {}valid loss: {} valid accuracy {}"
                    print(line.format(
                        e, args['epoch'],
                        running_label_loss / valid_every, running_unlabel_loss /
                        valid_every, _v_loss, acc))
                else: 
                    avg_train_loss = ((running_label_loss + running_unlabel_loss) /
                                        (valid_every * 2))
                    print("epoch unsupervised {}:{} train loss {} valid loss: {} valid accuracy {}".format(
                        e, args['epoch'], avg_train_loss , _v_loss, acc))

                running_label_loss = 0
                runninig_unlabel_loss = 0
                running_loss = 0

    _, acc = run_valid(
        model, valid_loader,
        sup_criterion
    )
    return _model_path, acc

def test(model, args):

    print("Running test\n")
    _pprint(args)

    ld_p = args['load_path']
    print('loading from %s'%ld_p)
    load_eval(ld_p, model)
    print('Done!')

    test_loader = get_test_loader(batch_size=32, pin_memory=USE_CUDA)
    sup_criterion = get_sup_criterion(use_cuda=USE_CUDA)
    loss, acc = \
        run_valid(model, data_loader=test_loader, criterion=sup_criterion)
    print('#'*10)
    print('final evaluation on test set - \nloss: {} \naccuracy: {}'.format(loss, acc))
    print('#'*10)
    return loss, acc

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
        embedding_size=args['kc'] * 28 ** 2,
        hidden_size=[120, 84]
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
    args['train_args']['valid_acc'] = valid_loss
    args['test_args']['log_dir'] = log_dir

    test_loss, test_acc = test(model, args['test_args'])

    args['test_args']['final_loss'] = test_loss
    args['test_args']['final_acc'] = test_acc

    args_fp = os.path.join(log_dir, 'params.json')
    arguments.logdictargs(args_fp, args)
    print("writing {} to  {}".format(args_fp, args))

    return args_fp


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--args_file', default='./default_ssl_e2e_args.json')
    args_file = parser.parse_args().args_file

    run(args_file)
