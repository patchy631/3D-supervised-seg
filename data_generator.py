import numpy as np
import h5py
import pandas as pd


def one_hot_encode(labels):
    targets = np.array(list(map(label_mapper, labels)))
    nb_classes = 5
    one_hot_targets = np.eye(nb_classes)[targets]
    return one_hot_targets


def label_mapper(label):
    label_map = {1.0: 0, 2.0: 1, 6.0: 2, 9.0: 3, 26.0: 4}
    return label_map[label]


def save_h5(h5_filename, data, label, data_dtype='float32', label_dtype='float32'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
        'data', data=data,
        compression='gzip', compression_opts=4,
        dtype=data_dtype)
    h5_fout.create_dataset(
        'label', data=label,
        compression='gzip', compression_opts=1,
        dtype=label_dtype)
    h5_fout.close()



def group_into_grids(points):
    pt_cloud_grids = []
    sorted_by_x = points[np.argsort(points[:, 0])]
    splits_along_x = np.array_split(sorted_by_x, 10)
    for split in splits_along_x:
        sorted_by_y = split[np.argsort(split[:, 1])]
        splits_along_y = np.array_split(sorted_by_y, 10)
        pt_cloud_grids.extend(splits_along_y)
    return pt_cloud_grids


def normalize_data(arr):
    arr[:, :-1] = arr[:, :-1] / 1000
    max_x, min_x = np.max(arr[:, 0]), np.min(arr[:, 0])
    max_y, min_y = np.max(arr[:, 1]), np.min(arr[:, 1])
    max_z, min_z = np.max(arr[:, 2]), np.min(arr[:, 2])
    arr[:, 0] = (arr[:, 0] - min_x) / (max_x - min_x)
    arr[:, 1] = (arr[:, 1] - min_y) / (max_y - min_y)
    arr[:, 2] = (arr[:, 2] - min_z) / (max_z - min_z)
    return arr


def main():
    df = pd.read_csv('records.csv', usecols=['X', 'Y', 'Z', 'raw_classification'])
    arr = df.to_numpy().astype(np.float32)
    arr = normalize_data(arr)
    inputs = arr[:, :-1]
    labels = arr[:, -1]
    inputs = inputs[labels != 2.0]
    labels = labels[labels != 2.0]
    labels = one_hot_encode(labels)
    save_h5('pt_cloud_data.h5', inputs, labels)
    print()

if __name__ == '__main__':
    main()