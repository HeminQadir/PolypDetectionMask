from dataset import data_path
from sklearn.model_selection import KFold
import numpy as np


def get_split(fold, num_splits=5):
    train_path = data_path / 'train' / 'polyp' / 'images'

    train_file_names = np.array(sorted(list(train_path.glob('*'))))

    kf = KFold(n_splits=num_splits, random_state=2018)

    ids = list(kf.split(train_file_names))

    train_ids, val_ids = ids[fold]

    if fold == -1:
        return train_file_names, train_file_names
    else:
        return train_file_names[train_ids], train_file_names[val_ids]


if __name__ == '__main__':
    ids = get_split(0)
