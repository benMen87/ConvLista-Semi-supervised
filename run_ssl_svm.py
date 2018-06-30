import train_embedding
import train_test_svm


def run(args_file):
    print("Training embedding model")
    updated_args_file = train_embedding.run(args_file)
    print("Training SVM classifier on embedding vectors")
    train_test_svm.run(updated_args_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--arg_file', default='')
    arg_file = parser.parse_args().arg_file

    run(arg_file)
