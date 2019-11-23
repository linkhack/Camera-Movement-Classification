
import tensorflow.keras as keras


def build_model():
    inputs = keras.Input((16, 299, 299, 3))
    base_model = keras.applications.VGG19(include_top=False, weights='imagenet')
    feature_extractor = keras.layers.TimeDistributed(
        keras.Model(inputs=base_model.input, outputs=base_model.output),
        input_shape=(16, 299, 299, 3)
    )(inputs)
    x = keras.layers.TimeDistributed( keras.layers.Flatten())(feature_extractor)
   # x = keras.layers.LSTM(255)(x)
    output = keras.layers.Dense(3, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=output)
    return model

