import os
import torch
import numpy as np
from sklearn import svm, metrics, model_selection
from train_embedding import build_model
from common import load_eval
import arguments
import mnist_datasets


USE_CUDA = torch.cuda.is_available()


def load_model(args, saved_model_path):
    model = build_model(args)
    load_eval(saved_model_path, model)
    if USE_CUDA:
        model = model.cuda()
    return model

def get_train_embeddings_PCA(model, train_args):
    
    data_labels, data_u, = mnist_datasets.semisup_mnist(train_args['label_count'])

    labeled_data = []
    labels = []

    for batch_d, batch_l in data_labels:
        if USE_CUDA:
            batch_d = batch_d.cuda()
        output, conv_sc = model(batch_d)
        embedding =  conv_sc.cpu().data.numpy()
        labeled_data += [embd.flatten() for embd in embedding]
        labels += [l.data.numpy() for l in batch_l]
    labeled = (labeled_data, labels)

    unlabeled = []
    for d, in data_u:
        if USE_CUDA:
            d = d.cuda()
        _, conv_sc = model(d)
        embedding  = conv_sc.cpu().data.numpy()
        unlabeled += [embd.flatten() for embd in embedding]

    from sklearn import decomposition

    np_unlabeled = np.array(unlabeled)
    np_labeled = np.array(labeled[0])
    print("Runnig PCA to for embedding")
    pca = decomposition.PCA(n_components=train_args['PCA_components'])

    pca.fit(np_labeled)
    pca.transform(np_unlabeled)

    def pca_embbed(x):
        dim0 = x.shape[0]
        return pca.transform(x.reshape(dim0, -1))

    return (pca.transform(np_labeled), labeled[1]), pca.transform(np_unlabeled), pca_embbed


def get_train_embeddings_AVG(model, train_args):
    """Embed train data and split for ssl"""
    data_labels, data_u, = mnist_datasets.semisup_mnist(train_args['label_count'])

    labeled_data = []
    labels = []

    print("Runnig Spatial AVG to for embedding")

    for batch_d, batch_l in data_labels:
        if USE_CUDA:
            batch_d = batch_d.cuda()
        output, conv_sc = model(batch_d)
        #embedding is spatial mean thus sc dim: (C, W, H) --> emb dim: (C, 1)
        embedding =  conv_sc.mean(-1).mean(-1).cpu().data.numpy()
        labeled_data += [embd for embd in embedding]
        labels += [l.data.numpy() for l in batch_l]
    labeled = (labeled_data, labels)

    unlabeled = []
    for d, in data_u:
        if USE_CUDA:
            d = d.cuda()
        _, conv_sc = model(d)
        embedding  = conv_sc.mean(-1).mean(-1).cpu().data.numpy()
        unlabeled += [embd for embd in embedding]
    return labeled, unlabeled, lambda x: x.mean(axis=(-1, -2))

def get_test_embeddings(model, convert_fn):
    test_dataset = mnist_datasets.get_test_loader()
    test_embeddings = []

    labeled_data = []
    labels = []
    for batch_d, batch_l in test_dataset:
        if USE_CUDA:
            batch_d = batch_d.cuda()
        _, conv_sc = model(batch_d)
        # embedding is spatial mean thus sc dim: (C, W, H) --> emb dim: (C, 1)
        embedding  = convert_fn(conv_sc.cpu().data.numpy())
        labeled_data += [embd for embd in embedding]
        labels += [l.data.numpy() for l in batch_l]

    return (labeled_data, labels)


def train_svm(labeled, unlabeled):
    """
    train svm on embedding vectors
    TODO(hillel): add svm semi sup support
    args:
    labeled [tuple]: (data_list, label_list)
    unlabeled [list]: data_list
    """

    mean = np.mean(unlabeled)
    var = np.var(unlabeled)

    embeddings, labels = labeled
    embeddings = (embeddings - mean) / var
    print("training SVM on n_samples:%d"%len(embeddings))
    C_range = np.logspace(-2, 10, 5)
    gamma_range = np.logspace(-9, 3, 5)
    param_grid = dict(gamma=gamma_range, C=C_range, kernel=['rbf'])
    grid = model_selection.GridSearchCV(svm.SVC(), param_grid=param_grid, n_jobs=4)
    grid.fit(np.array(embeddings), np.array(labels))

    print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

#    clf  = svm.SVC(C=1e3, gamma=1e3, kernel='rbf')
#    clf.fit(np.array(embeddings), np.array(labels))
    print("SVM training done")
    return grid, mean, var
#    return clf

def test_svm(clf, test_data, mean, var, save_path):
    """
    trained classifier on embedding vectors
    args:
    clf [obj]: classifier implements predictions fn
    test_data [tuple]: (data_list, label_list)
    """
    embeddings, ground_truth = test_data
    embeddings = (embeddings - mean) / var

    predictions = clf.predict(embeddings)
    confusion_matrix = metrics.confusion_matrix(ground_truth, predictions, labels=list(range(10)))

    print(confusion_matrix)
    print(metrics.classification_report(ground_truth, predictions, labels=list(range(10))))
    return confusion_matrix
    
def run(args_file):
    args = arguments.load_args(args_file)
    print("loaded {} from file {}".format(args, args_file))
    model  = load_model(args['model_args'],
            args['test_args']['embedd_model_path'])

    if args['train_args']['embedd_type'] == 'AVG':
        labeld_embedding, unlabeled_embedding, embedd_fn =\
            get_train_embeddings_AVG(model, args['train_args'])
    elif args['train_args']['embedd_type'] == 'PCA':
        labeld_embedding, unlabeled_embedding, embedd_fn =\
            get_train_embeddings_PCA(model, args['train_args'])

    clf, mean, var= train_svm(labeld_embedding, unlabeled_embedding)

    test_embeddings = get_test_embeddings(model, embedd_fn)
    print("Running SVM on test set")
    confusion_matrix = test_svm(clf, test_embeddings, mean, var, args['test_args']['log_dir'])
    np.save(os.path.join(args['test_args']['log_dir'], 'confusion_matrix'), confusion_matrix)

    args['test_args']['accuracy'] = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    
    arguments.logdictargs(os.path.join(args['test_args']['log_dir'], 'params.json'), args)
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--arg_file', default='/home/hillel/projects/listaSemiSupervised/saved_models/debug_10k_label/params.json')
    arg_file = parser.parse_args().arg_file

    run(arg_file)


