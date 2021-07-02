import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Convolution1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda, concatenate
from tensorflow.keras.models import Model


class PointNet:
    """
    Tensorflow2 implementation of PointNet
    """

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def mat_mul(self, A, B):
        return tf.matmul(A, B)

    def exp_dim(self, global_feature, input_shape):
        return tf.tile(global_feature, [1, input_shape, 1])

    def build(self):
        """
        Pointnet Architecture
        """
        # input_Transformation_net
        input_points = Input(shape=(self.input_shape, 3))
        x = Convolution1D(64, 1, activation='relu',
                          input_shape=(self.input_shape, 3))(input_points)
        x = BatchNormalization()(x)
        x = Convolution1D(128, 1, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Convolution1D(1024, 1, activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=self.input_shape)(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(9, weights=[np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(x)
        input_T = Reshape((3, 3))(x)

        # forward net
        g = Lambda(self.mat_mul, arguments={'B': input_T})(input_points)
        g = Convolution1D(64, 1, input_shape=(self.input_shape, 3), activation='relu')(g)
        g = BatchNormalization()(g)
        g = Convolution1D(64, 1, input_shape=(self.input_shape, 3), activation='relu')(g)
        g = BatchNormalization()(g)

        # feature transformation net
        f = Convolution1D(64, 1, activation='relu')(g)
        f = BatchNormalization()(f)
        f = Convolution1D(128, 1, activation='relu')(f)
        f = BatchNormalization()(f)
        f = Convolution1D(1024, 1, activation='relu')(f)
        f = BatchNormalization()(f)
        f = MaxPooling1D(pool_size=self.input_shape)(f)
        f = Dense(512, activation='relu')(f)
        f = BatchNormalization()(f)
        f = Dense(256, activation='relu')(f)
        f = BatchNormalization()(f)
        f = Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])(f)
        feature_T = Reshape((64, 64))(f)

        # forward net
        g = Lambda(self.mat_mul, arguments={'B': feature_T})(g)
        seg_part1 = g
        g = Convolution1D(64, 1, activation='relu')(g)
        g = BatchNormalization()(g)
        g = Convolution1D(128, 1, activation='relu')(g)
        g = BatchNormalization()(g)
        g = Convolution1D(1024, 1, activation='relu')(g)
        g = BatchNormalization()(g)

        # global_feature
        global_feature = MaxPooling1D(pool_size=self.input_shape)(g)
        global_feature = Lambda(self.exp_dim, arguments={'input_shape': self.input_shape})(global_feature)

        # point_net_seg
        c = concatenate([seg_part1, global_feature])
        c = Convolution1D(512, 1, activation='relu')(c)
        c = BatchNormalization()(c)
        c = Convolution1D(256, 1, activation='relu')(c)
        c = BatchNormalization()(c)
        c = Convolution1D(128, 1, activation='relu')(c)
        c = BatchNormalization()(c)
        c = Convolution1D(128, 1, activation='relu')(c)
        c = BatchNormalization()(c)
        prediction = Convolution1D(self.num_classes, 1, activation='softmax')(c)
        """
        end of pointnet
        """

        # define model
        model = Model(inputs=input_points, outputs=prediction)
        print(model.summary())
        return model
