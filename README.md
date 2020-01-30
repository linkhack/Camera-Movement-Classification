# Camera Movement Classification: A fundamental base for automatic video analysis
## Introduction
This project tries to classify movie shots based on the camera movement. We wish to classify the video into one of the three classes pan, tilt and tracking. Pan is a horizontal rotation, tilt is a vertical rotation of the camera. Tracking is filming from a moving platform.

The data stems from the digitalization of historical videos from around the second world war. The quality and camera parameters can widely vary from shot to shot. The dataset can be found on http://efilms.ushmm.org/ and https://imediacities.hpc.cineca.it/.

## Package
### Installation
### CameraClassificationModel
This is the main model of this package. It classifies a window of n frames. The model achitecture is based on the paper "Long-Term Recurrent Convolutional Networks for Visual Recognition and Description" by Donahue et al. [[link](https://arxiv.org/abs/1411.4389)]. The model is configurable with the config.yml file. If one comments or deletes a line, then this parameter will be set to default values.

The model is configurable through following parameters:
- input_size: (width, height, channels). Default: (224,224,3)
- window_size: Number of frames in detection window. Default: 16
- base_model: Feature extractor to use. One of 'VGG16', 'VGG19'. 'ResNet', 'DenseNet'. Default: 'VGG16'
- feature_layer: Layer name from which to extract features. Look into documentation of the models for layer names.
                 If feature_layer is none, then only the fully connected classification layers get removed. 'block5_pool'
- trainable_features: If the feature extractor is trainable. Default: False
- temporal: How to model the temporal component. One of 'LSTM', 'CONV'. Default: 'LSTM'
- LSTM_size: Each list entry is a lstm layer with this number of units. Default: [32]
- CONV_filter: Each list entry is a 1d convolution layer with this number of filters. Length has to be equal to CONV_filter_size. Default: [64]
- CONV_filter_size: Each list entry is a 1d convolution layer with this filter size. Length has to be equal to CONV_filter. Default: [3]
- nr_classes: Number of classes to classify. Default: 2

As an example, if one sets `temporal: 'LSTM'`, `LSTM_size: [64, 64]`,  and `base_model:'VGG16'`. Then the model will use VGG16 to extract features and then feed this features to two stacked lstm layers with 64 units each. All the other parameters wil be set to their default value. 
```python
from camclassifier import CameraMovementClassifier

model = CameraMovementClassifier.build_model(base_model='VGG16', temporal='LSTM', LSTM_size=[64, 64])
```

One can also create a model directly from a config file with:
```python
from camclassifier import CameraMovementClassifier

model = CameraMovementClassifier.build_model_from_config('config.yml')
```

Prediction can be done with:
```python
model.predict(x, batch_size=batch_size)
```
Note that the shape of x has to be `(batch, window_size, width, height, channels)`. Therefore arbitrarily long shots are not supported by this model and a `window_size` long subshot has first to be extracted. This can be achieved with the included [DataLoader](#dataloader). As this is a tensorflow.keras model, the documentation of the keras api can be found [here](https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict).
### InferenceModel
This model is used for inference. It can classify a arbitrarily long shot. This model uses a sliding window approach. Each `window_size' long subshot is predicted using a CameraMovementClassifier base model. The average over all predictions is the final prediction. The model is configurable with the config.yml file. If one comments or deletes a line, then this parameter will be set to default values.

The model is configurable through following parameters:
- base_model: Model used to classify each window.
- window_size: Window size in frames. 
- window_stride: Take every `stride` frame.
- nr_classes: Number of classes to classify.

A InferenceModel can be constructed in the following way
```python
import tensorflow as tf
from camclassifier.InferenceModel import InferenceModel

base_model = tf.keras.models.load_model(base_model_path)
inference_model = InferenceModel(base_model=base_model, window_size=16, window_stride=3, nr_classes=2)
```
Alternatively it can also be directly created from a config file
```python
from camclassifier.InferenceModel import InferenceModel

inference_model = InferenceModel.build_model_from_config('config.yml')
```

Prediction can be done with:
```python
inference_model.predict(shot)
inference_model.batch_predict([shot1, shot2, shot3])
```
Shot should have the shape (frames, width, height, channels). The method `batch_predict(shots)` has as input a list of shots where each shot can have a different number of frames, but width, height and number of channels have to be the same.
### DataLoader
This class is a wrapper for a tf.data.Dataset. It also does all the preprocessing, batching, padding, random subshot selection, data augmentation and video loading. In essence the Dataset uses a list of data elements as inputs. These elements specify the file path, classification, start frame and end frame. The files get loaded and processed in parallel, than batched and then sent to the gpu or cpu as tensors or as numpy arrays, depending on which pipeline one uses.
#### Configuration
The DataLoader can be configured so that it fits with the definition of the [CameraClassificationModel](#cameraclassificationmodel) or [InferenceModel](#inferencemodel) additionally to some directly related to the dataset generation. The available parameters are:
- dataset_path: Path to flist file.
- frame_size: Output size of a frame (width, height).
- frame_number: Number of frames in a window.
- stride: Take every stride-th frame.
- preprocess_name: Name of the preprocess. The available preprocesses are given in preprocess_dict.
                                One of 'VGG16', 'VGG19'. 'ResNet', 'DenseNet' or ''.
- nr_classes: Number of classes in Dataset.
- nr_threads: Number of threads to use for video loading/processing. This allows parallel execution of the generation
                           of elements on the cpu.
                           
The class also offers an utility function to create the requaried keyword parameter dictionary for the training, validation and test set automatically from a configuration file. This can be used as follows.
 ```python
 from camclassifier.DataLoader import DataLoader

training_configs = DataLoader.get_args_from_config('config.yml')
training_set = DataLoader(**training_configs.get('training'))
validation_set = DataLoader(**training_configs.get('validation'))
test_set = DataLoader(**training_configs.get('test'))
```
#### flist
This is the file-format that is used to create datasets from. The general file is defined like this:
- Each line is a data element.
- Each value on a line is separated by a space.
- The file_path should contain the complete path (best absolute) including file name and file extension
- Classification is a integer corresponding to a class label. This number should be `0<=classification<nr_classes`
- Start_frame and End_Frame are the start and end of the shot in frames.
```
file_path1 classification start_frame end_frame
file_path2 classification start_frame end_frame
```
In the [Develop](#data-preparation) folder are several scripts that can be used to generate these flists. 

#### Pipelines
This class offers several pipelines adhering to the tensorflow keras api and can therefore be used directly with methods such as fit, predict etc. and a generator that can be used for the [InferenceModel](#inferencemodel). The batch size can be specified for the tensorflow pipelines. These pipelines return tensors. The generator returns numpy arrays.

The available pipelines are:
- `training_pipeline`: Iterates over the whole dataset. Used for training. Shots are of `window_size` length.
- `balanced_pipeline`: Resampled, repeating dataset, such that every class is equally likely to be sampled. Shots are of `window_size` length. `steps_per_epoch` has to be specified for training.
- `training_pipeline_repeating`: Iterates repeatedly over the dataset. Used if `steps_per_epoch` is used. Shots are of `window_size` length.
- `validation_pipeline`: Iterates over the whole dataset. Used for validation. Shots are of `window_size` length.
- `py_iterator`:  Python generator that extracts complete shots, iterating over the whole dataset. Shots have arbitrary length.

The basic usage is:
```python
from camclassifier.DataLoader import DataLoader

training_configs = DataLoader.get_args_from_config('config.yml')
test_set = DataLoader(**training_configs.get('test'))
test_pipeline = test_set.validation_pipeline(batch_size=16)
for shot, label in test_pipeline:
    # Do something

complete_shots_generator = test_set.py_iterator()

for shot, label, file_name in complete_shots_generator:
    # Do something
```
## Demo

## Develop
### Data Preparation
### Training
### Tensorboard
### Evaluation
