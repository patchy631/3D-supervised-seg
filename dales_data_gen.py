import glob

import laspy as lp
import numpy as np


def one_hot_encode(labels):
    targets = np.array(list(map(label_mapper, labels)))
    nb_classes = 9
    one_hot_targets = np.eye(nb_classes)[targets]
    return one_hot_targets


def group_into_grids(points, grid_dim):
    pt_cloud_grids = []
    sorted_by_x = points[np.argsort(points[:, 0])]
    splits_along_x = np.array_split(sorted_by_x, grid_dim)
    for split in splits_along_x:
        sorted_by_y = split[np.argsort(split[:, 1])]
        splits_along_y = np.array_split(sorted_by_y, grid_dim)
        pt_cloud_grids.extend(splits_along_y)
    return pt_cloud_grids


def label_mapper(label):
    """
    for dales
    Args:
        label:

    Returns:

    """
    label_map = {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, 4.0: 4, 5.0: 5, 6.0: 6, 7.0: 7, 8.0: 8}
    return label_map[label]


def norm_pts(x):
    """
    Normalise points to a
    Unit Sphere
    Args:
        x: input arr

    Returns: points normalised
    to Unit Sphere

    """
    x = x / 1000
    x -= np.mean(x, axis=0)
    dists = np.linalg.norm(x, axis=1)
    return x / np.max(dists)


def generate_samples(points, sample_size):
    inputs = []
    labels = []
    for i in range(10):
        indices = np.random.randint(points.shape[0], size=sample_size)
        sample = points[indices, :]
        inp = sample[:, :-1]
        label = one_hot_encode(sample[:, -1])
        inputs.append(inp)
        labels.append(label)
    return inputs, labels

t = np.load('da')

list_train_files = glob.glob('./dales_las/test/*.las')[:5]

input_samples = []
label_samples = []
count = 0
for file in list_train_files:
    point_cloud = lp.read(file)
    data = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z, point_cloud.classification)).transpose()
    grids = group_into_grids(data, 10)
    for grid in grids:
        grid[:, :3] = norm_pts(grid[:, :3])
        inputs, labels = generate_samples(grid, 2048)
        input_samples.extend(inputs)
        label_samples.extend(labels)
    count += 1
    print(f'{count} files processed')

np.save('dales_train/dales_inputs_test.npy', np.asarray(input_samples))
np.save('dales_train/dales_labels_test.npy', np.asarray(label_samples))
