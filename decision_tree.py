import os
import numpy as np

DATA_ROOT = '/Users/xana/Dev/data/'
HOT_DATA = 'ivy_l4_hot_entry'
NORMAL_DATA = 'ivy_l4_not_hot_entry'


def load_data_set(data_name):
    with open(os.path.join(DATA_ROOT, data_name)) as f:
        for i, line in enumerate(f):
            pass
    features = np.zeros((i + 1, 10))
    with open(os.path.join(DATA_ROOT, data_name)) as f:
        for i, line in enumerate(f):
            tokens = line.strip('\n').split(' ')
            tokens[4] = int(tokens[4] == 'True')
            tokens[6] = int(tokens[5] == 'M')
            tokens[9] = int(tokens[9] == 'True')
            features[i] = tokens
            if i % 1000 == 0:
                print(i)
    return features


if __name__ == '__main__':
    hot_data = load_data_set(HOT_DATA)
    normal_data = load_data_set(NORMAL_DATA)
