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

def get_train_embeddings(model, train_args):
    """Embed train data and split for ssl"""
    data_labels, data_u, = mnist_datasets.semisup_mnist(train_args['label_count'])

    labeled_data = []
    labels = []
    for batch_d, batch_l in data_labels:
        if USE_CUDA:
            batch_d = batch_d.cuda()
        output, conv_sc = model(batch_d)
        #embedding is spatial mean thus sc dim: (C, W, H) --> emb dim: (C, 1)
        embedding  = conv_sc.mean(-1).mean(-1).cpu().data.numpy()
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

    return labeled, unlabeled

def get_test_embeddings(model):
    test_dataset = mnist_datasets.get_test_loader()
    test_embeddings = []

    labeled_data = []
    labels = []
    for batch_d, batch_l in test_dataset:
        if USE_CUDA:
            batch_d = batch_d.cuda()
        _, conv_sc = model(batch_d)
        # embedding is spatial mean thus sc dim: (C, W, H) --> emb dim: (C, 1)
        embedding  = conv_sc.mean(-1).mean(-1).cpu().data.numpy()
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
    embeddings, labels = labeled
    print("training SVM on n_samples:%d"%len(embeddings))
    C_range = np.logspace(-2, 10, 5)
    gamma_range = np.logspace(-9, 3, 5)
    param_grid = dict(gamma=gamma_range, C=C_range, kernel=('rbf', 'linear'))
    grid = model_selection.GridSearchCV(svm.SVC(), param_grid=param_grid, n_jobs=4)
    grid.fit(np.array(embeddings), np.array(labels))

    print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

#    clf  = svm.SVC(C=1e3, gamma=1e3, kernel='rbf')
#    clf.fit(np.array(embeddings), np.array(labels))
    print("SVM training done")
    return grid
#    return clf

def test_svm(clf, test_data, save_path):
    """
    trained classifier on embedding vectors
    args:
    clf [obj]: classifier implements predictions fn
    test_data [tuple]: (data_list, label_list)
    """
    embeddings, ground_truth = test_data
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
    labeld_embedding, unlabeled_embedding =\
        get_train_embeddings(model, args['train_args'])
    clf = train_svm(labeld_embedding, unlabeled_embedding)
    
    test_embeddings = get_test_embeddings(model)
    print("Running SVM on test set")
    confusion_matrix = test_svm(clf, test_embeddings, args['test_args']['log_dir'])
    np.save(os.path.join(args['test_args']['log_dir'], 'confusion_matrix'), confusion_matrix)
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--arg_file', default='/home/hillel/projects/listaSemiSupervised/saved_models/debug_10k_label/params.json')
    arg_file = parser.parse_args().arg_file

    run(arg_file)


