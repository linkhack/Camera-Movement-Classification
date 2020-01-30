
import tensorflow.keras as keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf


def build_model():
    inputs = keras.Input((16, 224, 224, 3))
    base_model = VGG16(include_top=False, weights='imagenet')
    feature_extractor = keras.layers.TimeDistributed(
        keras.Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output),
        input_shape=(16, 224, 224, 3)
    )
    feature_extractor.trainable = False
    x = feature_extractor(inputs)
    x = keras.layers.TimeDistributed( keras.layers.Flatten())(x)
    #x = keras.layers.LSTM(128, return_sequences = True, recurrent_activation='sigmoid')(x)
    #x = keras.layers.LSTM(64, return_sequences = True, recurrent_activation='sigmoid')(x)
    #x = keras.layers.LSTM(32, return_sequences = True, recurrent_activation='sigmoid')(x)
    #x = keras.layers.LSTM(32, return_sequences = True)(x)
    #x =  keras.layers.TimeDistributed(keras.layers.Dense(128, activation='relu'))(x)
    #x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Conv1D(64,3, dilation_rate=2, activation='relu')(x)
    x = keras.layers.Conv1D(64, 3, dilation_rate=2, activation='relu')(x)
    x = keras.layers.Conv1D(64, 3, dilation_rate=2, activation='relu')(x)

    x = keras.layers.TimeDistributed(keras.layers.Dense(2, activation='softmax'))(x)
    output = keras.layers.GlobalAveragePooling1D()(x)
    model = keras.Model(inputs=inputs, outputs=output)
    return model


def build_inference_model(window_model):
    input = keras.Input((None,224,224,3))
    frames = keras.layers.Lambda(sliding_window)(input, 2, 3)
    values = keras.layers.TimeDistributed(window_model)(frames)
    output = keras.layers.GlobalAveragePooling1D()(values)
    model = keras.Model(inputs=input, outputs=output)
    return model


def sliding_window(shots, stride_model=2, stride_windows=3):
    shot_length = tf.shape(shots)[1]
    window_with = 16
    valid_length = shot_length-(stride_model*(window_with-1))+1

    start_indices = tf.range(valid_length, delta = stride_windows, dtype=tf.int32)
    windows = tf.map_fn(lambda t: shots[:,t:(t+window_with-1)], start_indices)
    windows = tf.transpose(windows, [1,0,2,3,4,5])

    return windows


