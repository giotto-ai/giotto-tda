import pickle as pkl


def read_pickle(path):
    with open(path, 'rb') as f:
        return pkl.load(f)


def write_pickle(path, array):
    with open(path, 'wb') as f:
        pkl.dump(array, f)
