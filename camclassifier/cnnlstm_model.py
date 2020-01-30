from tensorflow.keras.layers import TimeDistributed, Flatten, LSTM, Conv1D, GlobalAveragePooling1D
from tensorflow.keras import Model, Input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from typing import Tuple, Optional, List, Any
import yaml

base_model_dict = {'VGG16': VGG16,
                   'VGG19': VGG19,
                   'ResNet': ResNet50,
                   'DenseNet': DenseNet121}


def build_model(input_size: Tuple[int, int, int] = (224, 224, 3),
                window_size: int = 16,
                base_model: str = 'VGG16',
                feature_layer: Optional[str] = None,
                trainable_features: bool = False,
                temporal: str = 'LSTM',
                LSTM_size: List[int] = None,
                CONV_filter: List[int] = None,
                CONV_filter_size: List[int] = None
                ):
    # Default Arguments
    if CONV_filter_size is None:
        CONV_filter_sizes = [3]
    if CONV_filter is None:
        CONV_filters = [64]
    if LSTM_size is None:
        LSTM_size = [32]

    if len(CONV_filter_size) != CONV_filter:
        raise ValueError('Length of CONV_filters and CONV_filter_sizes has to be equal')

    # Calculate input shape (frames, width, height, channels)
    input_shape = (window_size, input_size[0], input_size[1], input_size[2])

    # Select Feature extractor
    base_model = base_model_dict.get(base_model)(include_top=False, weights='imagenet')

    # Prepare feature extractor
    base_inputs = base_model.input
    if feature_layer is not None:
        base_outputs = base_model.get_layer(feature_layer).output
    else:
        base_outputs = base_model.outputs

    feature_extractor = TimeDistributed(Model(inputs=base_inputs, outputs=base_outputs), input_shape=input_shape)
    feature_extractor.trainable = trainable_features

    # Define Model
    inputs = Input(input_shape)
    x = feature_extractor(inputs)
    x = TimeDistributed(Flatten())(x)

    if temporal == "LSTM":
        # LSTM for temporal modeling
        for layer_size in LSTM_size:
            x = LSTM(layer_size, return_sequences=True)(x)
    elif temporal == 'CONV':
        # Timewise 1D convolution
        for filters, filter_size in zip(CONV_filter, CONV_filter_size):
            x = Conv1D(filter_size, filter_size, activation='relu')(x)

    output = GlobalAveragePooling1D()(x)
    model = Model(inputs=inputs, outputs=output)

    return model


def build_model_config(config_file: str):
    stream = open('config.yml', 'r')
    config = yaml.load(stream, Loader=yaml.BaseLoader)
    model_config = config.get('model')

    input_size = model_config.get('input_size')
    window_size = model_config.get('window_size')
    base_model = model_config.get('base_model')
    feature_layer = model_config.get('feature_layer')
    trainable_features = model_config.get('trainable_features')
    temporal = model_config.get('temporal')
    LSTM_size = model_config.get('LSTM_size')
    CONV_filter = model_config.get('CONV_filter')
    CONV_filter_size = model_config.get('CONV_filter_size')

    return build_model(input_size=input_size,
                       window_size=window_size,
                       base_model=base_model,
                       feature_layer=feature_layer,
                       trainable_features=trainable_features,
                       temporal=temporal,
                       LSTM_size=LSTM_size,
                       CONV_filter=CONV_filter,
                       CONV_filter_size=CONV_filter_size
                       )
