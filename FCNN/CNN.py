import tensorflow as tf
import numpy as np
import keras
from keras import backend as K


def Density():

    # Swish Activation Function
    def swish(x):
        return (K.sigmoid(x) * x)
    keras.utils.generic_utils.get_custom_objects().update({'swish': keras.layers.Activation(swish)})

    #Bilinear Upsampling
    class up2(keras.layers.UpSampling2D):
        def call(self, inputs):
            new_shape = tf.shape(inputs)[1:3]
            new_shape *= tf.constant(np.array([self.size[0], self.size[1]]).astype('int32'))
            return tf.image.resize_bilinear(inputs, new_shape)

    model = keras.Sequential()

    # Conv1
    model.add(
        keras.layers.Conv2D(filters=32, kernel_size=11, padding='same', activation=None, input_shape=(None, None, 1)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(swish))
    model.add(keras.layers.SpatialDropout2D(
        rate=0.2))  # Dropout randomly switches off filters to promote independence between them
    model.add(keras.layers.MaxPool2D(pool_size=2, padding='same'))

    # Conv2
    model.add(keras.layers.Conv2D(filters=64, kernel_size=9, padding='same', activation=None))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(swish))
    model.add(keras.layers.SpatialDropout2D(
        rate=0.2))  # Dropout randomly switches off filters to promote independence between them
    model.add(keras.layers.MaxPool2D(pool_size=2, padding='same'))

    # Conv3
    model.add(keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation=None))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(swish))
    model.add(keras.layers.SpatialDropout2D(
        rate=0.2))  # Dropout randomly switches off filters to promote independence between them

    # Conv4
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation=None))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(swish))

    # Conv5
    model.add(up2(size=2))
    #model.add(keras.layers.UpSampling2D(size=2))   # Uncomment for Nearest Neighbour Upsampling
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=None))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(swish))

    # Conv6
    model.add(up2(size=2))
    #model.add(keras.layers.UpSampling2D(size=2))
    model.add(keras.layers.Conv2D(filters=8, kernel_size=3, padding='same', activation=None))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(swish))

    # Conv7
    model.add(keras.layers.Conv2D(filters=1, kernel_size=3, padding='same', activation=None))
    model.add(keras.layers.Activation('relu'))

    def SSES_Loss(y_true, y_pred):
        return (K.square(K.sum(y_pred / 100) - K.sum(y_true / 100)) + K.sum(K.square(y_pred - y_true)))

    model.compile(optimizer=keras.optimizers.Adam(lr=1e-3), loss=SSES_Loss)

    return model





