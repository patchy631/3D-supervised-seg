import random
import laspy as lp
import numpy as np


def one_hot_encode(labels):
    targets = np.array(list(map(label_mapper, labels)))
    nb_classes = 5
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
    label_map = {1.0: 0, 2.0: 1, 6.0: 2, 9.0: 3, 26.0: 4}
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

def normalise_intensities(intensities):
    intensities = np.clip(intensities, a_min=1, a_max=600)
    intensities = (intensities - 1)/600
    return intensities

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


orig_point_cloud = lp.read('C_37EZ1.las')
orig_data = np.vstack((orig_point_cloud.x, orig_point_cloud.y, orig_point_cloud.z, orig_point_cloud.intensity,
                  orig_point_cloud.classification)).transpose()

# globally normalise intensities
orig_data[:, 3] = normalise_intensities(orig_data[:, 3])

orig_grids = group_into_grids(orig_data, 10)

orig_grids = random.sample(orig_grids, 30)

input_samples = []
label_samples = []
count = 0
for orig_grid in orig_grids:

    grids = group_into_grids(orig_grid, 10)
    for grid in grids:
        grid[:, :3] = norm_pts(grid[:, :3])
        inputs, labels = generate_samples(grid, 2048)
        input_samples.extend(inputs)
        label_samples.extend(labels)
    count += 1
    print(f'{count} files processed')

np.save('inputs.npy', np.asarray(input_samples))
np.save('labels.npy', np.asarray(label_samples))
