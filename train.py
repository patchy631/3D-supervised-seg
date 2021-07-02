import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from pointnet_model.pointnet import PointNet

tf.compat.v1.disable_eager_execution()


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)


def rotate_point_cloud(batch_data):
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    B, N, C = batch_data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data


'''
global variable
'''
# batch size
batch_size = 32
# number of points in each sample
num_points = 1000
# number of categories
k = 5
# epoch number
epo = 20
# define optimizer
adam = optimizers.Adam(lr=0.001, decay=0.7)

'''
load train and test data
'''
# load TRAIN points and labels
path = os.path.dirname(os.path.realpath(__file__))
train_path = os.path.join(path, "train_data")
filenames = [d for d in os.listdir(train_path)]
print(train_path)
print(filenames)
train_points = None
train_labels = None
for d in filenames:
    cur_points, cur_labels = load_h5(os.path.join(train_path, d))
    cur_points = cur_points[68682:]
    cur_labels = cur_labels[68682:]
    # cur_points = cur_points.reshape(1, -1, 3)
    # cur_labels = cur_labels.reshape(1, -1)
    if train_labels is None or train_points is None:
        train_labels = cur_labels
        train_points = cur_points
    else:
        train_labels = np.hstack((train_labels, cur_labels))
        train_points = np.hstack((train_points, cur_points))
train_points_r = train_points.reshape(-1, num_points, 3)
train_labels_r = train_labels.reshape(-1, num_points, k)

# load TEST points and labels
test_path = os.path.join(path, "test_data")
filenames = [d for d in os.listdir(test_path)]
print(test_path)
print(filenames)
test_points = None
test_labels = None
for d in filenames:
    cur_points, cur_labels = load_h5(os.path.join(test_path, d))
    cur_points = cur_points[68682:]
    cur_labels = cur_labels[68682:]
    if test_labels is None or test_points is None:
        test_labels = cur_labels
        test_points = cur_points
    else:
        test_labels = np.hstack((test_labels, cur_labels))
        test_points = np.hstack((test_points, cur_points))
test_points_r = test_points.reshape(-1, num_points, 3)
test_labels_r = test_labels.reshape(-1, num_points, k)

"""
load model
"""
model = PointNet(input_shape=1000, num_classes=k).build()

'''
train and evaluate the model
'''
# compile classification model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

"""
callbacks
"""
# Callbacks
logdir = "logs/train_data/"
tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=True)
modelCkptCallBack = ModelCheckpoint("./model.hdf5", monitor='val_loss', save_best_only=True, save_weights_only=True,
                                    verbose=1)

# train model
for i in range(epo):
    # augment the data by rotating and
    # jitter point cloud every epoch
    train_points_rotate = rotate_point_cloud(train_points_r)
    train_points_jitter = jitter_point_cloud(train_points_rotate)
    steps_per_epoch = int(train_points_jitter.shape[0] / batch_size)
    model.fit(train_points_jitter, train_labels_r, batch_size=batch_size,
              epochs=1, shuffle=True, verbose=1, callbacks=[tbCallBack])
    # evaluate model
    if i % 5 == 0:
        score = model.evaluate(test_points_r, test_labels_r, verbose=1)
        print('Test loss: ', score[0])
        print('Test accuracy: ', score[1])

final_model_path = "model_train_end.hdf5"
model.save(final_model_path)
