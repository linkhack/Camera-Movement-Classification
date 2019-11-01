import keras
import keras.layers as layers
from keras.models import Model, Sequential
from keras.layers import Layer, ConvLSTM2D, TimeDistributed
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16


class CNNLSTM(Model):
    def __init__(self):
        super().__init__()
        base_model = Xception(include_top=False, weights='imagenet')
        self.feature_extractor = TimeDistributed(
            Model(inputs=base_model.input, outputs=base_model.get_layer('block5_sepconv1_act').input),
            input_shape=(8, 299, 299, 3)
        )
        del base_model
        self.lstm_block = ConvLSTM2D(1024,(3, 3), padding='same')

    def call(self, inputs, mask=None):
        x = self.feature_extractor(inputs) # shape (batch, 16, 19,19,728)
        x = self.lstm_block(x)
        x = layers.Flatten()(x)
        x = layers.Dense(3, activation='softmax')(x)
        return x

