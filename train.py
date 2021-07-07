import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import losses
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
num_points = 2048
# number of categories
k = 9
# epoch number
epo = 20
# define optimizer
adam = optimizers.Adam(lr=0.001, decay=0.7)

'''
load train and test data
'''
train_points_r = np.load('dales_inputs.npy')
train_labels_r = np.load('dales_labels.npy')

test_points_r = np.load('dales_inputs_test.npy')
test_labels_r = np.load('dales_labels_test.npy')

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

# define categorical focal loss
loss = losses.categorical_focal_loss(alpha=0.25)

'''
train and evaluate the model
'''
# compile classification model
model.compile(optimizer='adam',
              loss=loss,
              metrics=['categorical_accuracy'])

# train model
for i in range(epo):
    # rotate and jitter point cloud every epoch
    train_points_rotate = rotate_point_cloud(train_points_r)
    train_points_jitter = jitter_point_cloud(train_points_rotate)
    model.fit(train_points_jitter, train_labels_r, batch_size=32, epochs=1, shuffle=True, verbose=1)

    # evaluate model
    if i % 2 == 0:
        score = model.evaluate(test_points_r, test_labels_r, verbose=1)
        if score[1] > 0.75:
            model.save('model_accurate.hdf5')
        print('Test loss: ', score[0])
        print('Test accuracy: ', score[1])
