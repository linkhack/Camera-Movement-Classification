# Camera Movement Classification: A fundamental base for automatic video analysis
  * [Introduction](#introduction)
  * [Package](#package)
    + [Installation](#installation)
    + [CameraClassificationModel](#cameraclassificationmodel)
    + [InferenceModel](#inferencemodel)
    + [DataLoader](#dataloader)
      - [Configuration](#configuration)
      - [flist](#flist)
      - [Pipelines](#pipelines)
  * [Demo](#demo)
  * [Develop](#develop)
    + [Data Preparation](#data-preparation)
    + [Training](#training)
    + [Tensorboard](#tensorboard)
    + [Evaluation](#evaluation)
## Introduction
This project tries to classify movie shots based on the camera movement. We wish to classify the video into one of the three classes pan, tilt and tracking. Pan is a horizontal rotation, tilt is a vertical rotation of the camera. Tracking is filming from a moving platform.

The data stems from the digitalization of historical videos from around the second world war. The quality and camera parameters can widely vary from shot to shot. The dataset can be found on http://efilms.ushmm.org/ and https://imediacities.hpc.cineca.it/.

## Package
### Installation
As this package uses tensorflow 2.0.1 and gpu acceleration you have to have CUDA 10.0 installed. Please follow the tensorflow instructions [here](https://www.tensorflow.org/install/gpu), but instead of installing CUDA10.1, install CUDA10.0. Alternatively you can also try to install the tensorflow backand that fits your architecture after installing this package. It should be tensorflow 2.0 or 2.1 and the project wasn't tested on tensorflow 2.1.

The package can be installed in two ways. Either directly from your local machine or from github. If you want to install it directly from github use:
```
pip install git+https://github.com/linkhack/Camera-Movement-Classification
```
One can also install it from local files if you have cloned or downloaded the repoisitory. Assume that the project is in the folder `Camera-Movement-Classification` and this folder contains the `setup.py` script and nothing changed in the folder structure. Then you can install this package also in this way:
```
pip install path/to/Camera-Movement-Classification
```
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
- class_dict: Dictionary used to encode classes. Should have entries with class_name: integer. Where the integers are consecutive from 0 to nr_classes-1

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
prediction = inference_model.predict(shot, movie_id, shot_id, csv_file)
batch_prediction = inference_model.batch_predict([(shot1, movie_id1, shot_id1), (shot2, movie_id2, shot_id2), (shot3, movie_id3, shot_id3)], csv_file)
```
Shot should have the shape (frames, width, height, channels). The method `batch_predict(shots)` has as input a list of shots where each shot can have a different number of frames, but width, height and number of channels have to be the same. The id's are used to identify prediction. The result is a numpy array of the form `[movie_id, shot_id, prediction], where the prediction is decoded into a string(name of class). Moreover if one specifies the csv_file parameter, then the predictions will be appended to the specified csv file. If this parameter is not given, or `None`, then the predictions are not written into a file.

Additionally this class offers the function `evaluate(shot)`, which just returns the raw softmax score instead of the classification. This is used to evaluate the model.
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

One can also load a single file with the correct processing steps with the function `load_complete_shot(file_name)`. This function returns the shot as a numpy array.
## Demo
The demo consists of two scripts, one to classify single files and one to classify complete folders. In the demofolder is a minimal configuration file, where all unneeded fields were deleted. The config will work with the given folder structure. You can specify the `csv_file` field to save the results into a csv-file. If one omits this the results will just be printed on the console.
- `predict_single.py --file Data` classifies the shot `shot.mp4`.
- `predict_batch.py --folder Data` classifies all shots inside the folder `Data`. 
## Develop
In this folder are all development related scripts. Moreover a complete configuration file is also provided. With this file the model architecture, training procedure and evaluation mode can be completly defined.
### Data Preparation
This repository contains several script to prepare the date for this model. Their usage is explained in this section.
#### Flist Creation
There are two scripts that help with generating the flist files.

`create_flist.py --folder_path path/to/folder --annotation path/to/annotation.csv --output annotation/annotation.flist` This script is used to generate flists from the original dataset. 
- `--folder_path` is the path in which the video files are contained. 
- `--annotation` is the annotation csv file in which the annotation for the specified folder is stored.
-' `--output` is the output file in which the flist should be stored
Unfortunately one has to copy the annotation of the imc and efilms shots manually into one file.

`create_flist_and_training_split_from_shots.py --folder_path path/to/folders --output annotation.flist` This script is used to generate flist directly from folders containing the shots. It also automatically generates a 60%-20%-20% train/val/test split of the dataset. This script assumes following folder structure:
- path/to/folders
   - class 1
      - file1
      - file2
      - ...
 - class 2
      - file1
      - file2
      - ...
 - ...
#### Dataset correction
- `cut_exact_videos.py --folder_path data_folder --annotation annotation.csv --output training_folder`
- `clean_data.py --folder_path training_folder --annotation corrections.flist --output training_folder`
- `delete_files --folder_path training_folder/class --annotation corrections.flist` 
are used in combination to generate the cleaned dataset. If the arguments are the same then they correspond to the same folder or file.

`cut_exact_videos.py --folder_path data_folder --annotation annotation.csv --output training_folder` is used to extract the shots from the whole video files and save them individually in `training_folder/class`. Note that a folder for each class has to be created in `training_folder`. `--folder_path` is the folder with the original videos and `--annotation` is the original csv annotation for this folder.

After running this script we have folders of the form `training_folder/class` for each class containing all the shots of this class. Then one has to manually go through each shot and write the corrections into corrections.flist. This file contains also file_name classification start_frame end_frame as above.

This correction file is then used with `clean_data.py` to cut and reclassify the videos. After running this script a new shot file is generated for each correction. As this doesn't delete the wrong shots that were corrected, we have to run `delete_files.py` to delete the wrong files.
### Training
To run training just execute `train.py`. It uses automatically the provided config file, as long as it is called `config.yml` and is contained in the same folder as `train.py`. In the config file one has to define the [DataLoader](#dataloader) and [CameraClassificationModel](#cameraclassificationmodel) as explained above. Moreover one has to define following fields in `config.yml`under `training:`
- training_set: Training set flist file.
- validation_set: Validation set flist file.
- test_set: Test: set flist file.
- balanced_training: If the dataset should be resampled, such that every class is equally likely.
- use_class_weights: If class weights should be used. This gives classes with less data more weight when calculating the loss.
- adam_lr: Learning rate of adam optimizer.
- adam_epsilon: Epsilon of adam optimizer.
- max_epochs: Maximum number of epochs.
- steps_per_epoch: How many datapoints are one epoch.
- early_stopping_patience: How often(epochs) can the loss not improve without stopping the training.
- model_logs_basepath: Basepath for model logs, used for tensorboard.
- model_checkpoints_basepath: Basepath for model checkpoints, used for saving model and optimizer weights.
This script automatically creates unique subfolders in the logging/checkpoint folders for each training run. The format of these folders is extractor_layer_trainable_temporal_layers_info_stride_dateuid. This helps in organizing the different runs.
### Tensorboard
A tensorboard visualization can be started with
```
tensorboard --logdir model_logs_basepath
```
which is then served on localhost.
### Evaluation

