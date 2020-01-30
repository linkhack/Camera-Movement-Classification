# Camera Movement Classification: A fundamental base for automatic video analysis
## Introduction
This project tries to classify movie shots based on the camera movement. We wish to classify the video into one of the three classes pan, tilt and tracking. Pan is a horizontal rotation, tilt is a vertical rotation of the camera. Tracking is filming from a moving platform.

The data stems from the digitalization of historical videos from around the second world war. The quality and camera parameters can widely vary from shot to shot. The dataset can be found on http://efilms.ushmm.org/ and https://imediacities.hpc.cineca.it/.

## Package
### Installation
### CameraClassificationModel
This is the main model of this package. It classifies a window of n frames. The model achitecture is based on the paper "Long-Term Recurrent Convolutional Networks for Visual Recognition and Description" by Donahue et al. [[link](https://arxiv.org/abs/1411.4389)]. The model is configurable with the config.yml file.

The model is configurable through following parameters:
'''
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
'''

### Inference Model
### DataLoader

## Demo

## Develop
### Data Preparation
### Training
### Tensorboard
### Evaluation
