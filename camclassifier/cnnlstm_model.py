
import tensorflow.keras as keras


def build_model():
    inputs = keras.Input((16, 299, 299, 3))
    base_model = keras.applications.Xception(include_top=False, weights='imagenet')
    feature_extractor = keras.layers.TimeDistributed(
        keras.Model(inputs=base_model.input, outputs=base_model.get_layer('block5_sepconv1_act').input),
        input_shape=(16, 299, 299, 3)
    )(inputs)
    x = keras.layers.ConvLSTM2D(1024,(3, 3), padding='same')(feature_extractor)
    x = keras.layers.Flatten()(x)
    output = keras.layers.Dense(3, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=output)
    return model

