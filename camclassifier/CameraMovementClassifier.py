from tensorflow.keras.layers import TimeDistributed, Flatten, LSTM, Conv1D, GlobalAveragePooling1D, Dense
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
                CONV_filter_size: List[int] = None,
                nr_classes: int = 2,
                ):
    """
    Builds the classification model

    :param input_size: (width, height, channels)
    :param window_size: Nr of frames in detection window.
    :param base_model: Feature extractor to use. One of 'VGG16', 'VGG19'. 'ResNet', 'DenseNet'
    :param feature_layer: Layer name from which to extract features. Look into base model documentation for layer names.
                          If feature_layer is none, then only the fully connected classification layers get removed.
    :param trainable_features: If feature extractor is trainable
    :param temporal: How to model temporal component. One of 'LSTM', 'CONV'
    :param LSTM_size: Each list entry is a lstm layer with this number of units
    :param CONV_filter: Each list entry is a conv layer with this number of filters. Length has to be equal to CONV_filter_size
    :param CONV_filter_size: Each list entry is a conv layer with this filter size. Length has to be equal to CONV_filter
    :param nr_classes: Number of classes to classify
    :return: tensorflow keras model
    """
    # Default Arguments
    if CONV_filter_size is None:
        CONV_filter_size = [3]
    if CONV_filter is None:
        CONV_filter = [64]
    if LSTM_size is None:
        LSTM_size = [32]

    if len(CONV_filter_size) != len(CONV_filter):
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
    x = TimeDistributed(Dense(nr_classes, activation='softmax'))(x)
    output = GlobalAveragePooling1D()(x)
    model = Model(inputs=inputs, outputs=output)

    return model


def build_model_from_config(config_file: str):
    """
    Wraps build_model so that one can use a config file.

    Config file has to have the form:

        model:
            param1: value
            param2: value

    where params have to have to be the same as in build_model. If one omits parameters in the config file, then these
    parameters are filled by their default value

    :param config_file: Path to config
    :return: A tensorflow.keras model
    """
    stream = open(config_file, 'r')
    config = yaml.load(stream, Loader=yaml.SafeLoader)
    model_config = config.get('model', dict())

    input_size = tuple(model_config.get('input_size',(224, 224, 3)))
    window_size = model_config.get('window_size', 16)
    base_model = model_config.get('base_model', 'VGG16')
    feature_layer = model_config.get('feature_layer', None)
    trainable_features = model_config.get('trainable_features', False)
    temporal = model_config.get('temporal', 'LSTM')
    LSTM_size = model_config.get('LSTM_size', None)
    CONV_filter = model_config.get('CONV_filter', None)
    CONV_filter_size = model_config.get('CONV_filter_size', None)
    nr_classes = model_config.get('nr_classes', 2)

    return build_model(input_size=input_size,
                       window_size=window_size,
                       base_model=base_model,
                       feature_layer=feature_layer,
                       trainable_features=trainable_features,
                       temporal=temporal,
                       LSTM_size=LSTM_size,
                       CONV_filter=CONV_filter,
                       CONV_filter_size=CONV_filter_size,
                       nr_classes=nr_classes
                       )
