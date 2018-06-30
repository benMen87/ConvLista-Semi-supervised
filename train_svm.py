from train_embedding import build_model
from common import load_eval
import arguments
import mnist_datasets

def get_train_embeddings(model_args, train_args):
    labeled = [()] # list of d,l 
    unlabeled = []

    load_path = train_args['embedd_model_path']
    label_count = train_args['label_count']

    model = build_model(model_args)
    load_eval(load_path, model)

    dll, dlu, = mnist_datasets.semisup_mnist(label_count)

    for d, l in dll:
        _, conv_sc = model(d)
        #embedding is spatial mean thus sc dim: (C, W, H) --> emb dim: (C, 1)
        embedding  = conv_sc.mean(-1).mean(-1).cpu().data.numpy()
        labeled.append((embedding, l.cpu().data.numpy()))

    for d, in dlu:
        _, conv_sc = model(d)
        embedding  = conv_sc.mean(-1).mean(-1).cpu().data.numpy()
        unlabeled.append(embedding)

    return labeled, unlabeled

def train_sslsvm(labeled, unlabeled):
    pass

def main(args_file):
    args = arguments.load_args(args_file)
    labeld_embbdings, unlabeld_embbedings =\
        get_train_embeddings(args['model_args'], args['train_args'])

    train_sslsvm(labeld_embbdings, unlabeld_embbedings)

    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--arg_file', default='')
    arg_file = parser.parse_args().arg_file

    main(arg_file)


