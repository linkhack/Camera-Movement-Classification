
import tensorflow.keras as keras
import tensorflow as tf


def build_model():
    inputs = keras.Input((16, 224, 224, 3))
    base_model = keras.applications.VGG19(include_top=False, weights='imagenet')
    feature_extractor = keras.layers.TimeDistributed(
        keras.Model(inputs=base_model.input, outputs=base_model.output),
        input_shape=(16, 224, 224, 3)
    )
    feature_extractor.trainable = False
    x = feature_extractor(inputs)
    x = keras.layers.TimeDistributed( keras.layers.Flatten())(x)
    x = keras.layers.LSTM(512)(x)
    x =  keras.layers.Dense(128, activation='relu')(x)
    #x = keras.layers.GlobalAveragePooling1D()(x)
    output = keras.layers.Dense(3, activation='softmax')(x)

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


