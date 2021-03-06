import tensorflow as tf
from skimage import exposure, util

import random
from typing import Tuple, List, Optional
import numpy as np
import cv2
import keras.applications.resnet50 as resnet50
import keras.applications.densenet as densenet
import keras.applications.vgg16 as vgg16
import keras.applications.vgg19 as vgg19
from keras_applications import imagenet_utils

import yaml

class DataLoader:
    """
    Class that wraps tf.data.Dataset for video shots and offers different iterators.
    """

    preprocess_dict= {'VGG16': vgg16.preprocess_input,
                      'VGG19': vgg19.preprocess_input,
                      'ResNet': resnet50.preprocess_input,
                      'DenseNet': densenet.preprocess_input,
                      '':lambda x: imagenet_utils.preprocess_input(x, mode='tf')}

    def __init__(self, dataset_path: Optional[str] = None, frame_size: Tuple[int, int]=(224,224), frame_number: int = 16, stride:int = 1, preprocess_name:str='VGG16', nr_classes:int = 2, nr_threads:int = 2):
        """
        Used to create a Dataset. Iterators/Pipelines can be created with the different pipeline/iterator function.
        This class handles the loading of shots from video files. The complete preprocessing is handled by this class.
        Also the extraction and padding such that each shot has the correct length.

        The dataset should be given in .flist format. This format contains one line for each element of the dataset
        A line should be as follows:

        file_path classification start_frame end_frame

        File_path is the path to the file(for example data/pan/1823_23893719.mp4). These values are separated by spaces.
        Classification should be an integer corresponding to the class with 0<=classification<nr_classes

        :param dataset_path: Path to flist file.
        :param frame_size: Output size of a frame (width, height).
        :param frame_number: Number of frames in a window.
        :param stride: Take every stride-th frame.
        :param preprocess_name: Name of the preprocess. The available preprocess are given in preprocess_dict.
                                One of 'VGG16', 'VGG19'. 'ResNet', 'DenseNet' or ''.
        :param nr_classes: Number of classes in Dataset.
        :param nr_threads: Number of threads to use for video loading/processing. This allows parallel execution of the generation
                           of elements on the cpu.
        """
        if dataset_path is not None:
            self.inputs, self.labels = self.process_flist(dataset_path)
        else:
            self.inputs = self.labels = []
        # labels and counts
        unique, counts = np.unique(self.labels, return_counts=True)
        print(f"Elemets per class {counts}")

        self.class_probabilities = list(counts/len(self.inputs))
        print(f"Probability per class {self.class_probabilities}")

        #Statistics
        self.frame_size = frame_size
        self.frame_number = int(frame_number)
        self.stride = int(stride)
        self.counts = counts
        self.length = len(self.inputs)
        self.nr_classes = nr_classes

        #Preprocess
        self.preprocess_fn = DataLoader.preprocess_dict.get(preprocess_name)
        self.nr_threads = nr_threads

        # Create flist dataset
        self.dataset = tf.data.Dataset.from_tensor_slices(self.inputs)

        # Each class in own dataset for resampling
        self.classes = self.split_classes(self.inputs) #split dataset into classes
        self.tfdataset_classes_list = [tf.data.Dataset.from_tensor_slices(self.classes.get(label, [])) for label in self.classes]


    def process_flist(self, dataset_path: str) -> Tuple[List[Tuple[str, ...]], List[str]]:
        """
        Parses flist content. Creates List of Tuples (path,classification,start_frame, end_frame) and a List of labels for each element.
        :param dataset_path: Path to flist of dataset
        :return:
        """
        with open(dataset_path,'r') as file:
            content = file.read()
            lines = content.splitlines()
            lines = [tuple(line.split()) for line in lines]
            labels = [line[1] for line in lines]

        return lines, labels

    def balanced_pipeline(self, batch_size: int):
        """ Resampled pipeline, such that every class is equally likely to be sampled. Shots are of window_size length."""
        dataset = tf.data.experimental.sample_from_datasets(self.tfdataset_classes_list,[1/self.nr_classes]*len(self.tfdataset_classes_list))
        return dataset.map(self.process_file, num_parallel_calls=4).repeat().batch(batch_size).prefetch(1)

    def training_pipeline(self, batch_size: int):
        """ Iterates over whole dataset. Used for training. Shots are of window_size length."""
        return self.dataset.shuffle(self.length).map(self.process_file, num_parallel_calls=self.nr_threads).batch(
        batch_size).prefetch(1)

    def training_pipeline_repeating(self, batch_size: int):
        """ Iterates repeatedly over dataset. Used if steps_per_epoch is set in model.fit / config. Shots are of window_size length."""
        return self.dataset.repeat().shuffle(self.length).map(self.process_file, num_parallel_calls=self.nr_threads).batch(
        batch_size).prefetch(1)

    def validation_pipeline(self, batch_size: int):
        """ Iterates over whole dataset. Used for validation. Shots are of window_size length."""
        return self.dataset.map(self.process_file, num_parallel_calls=self.nr_threads).batch(batch_size).prefetch(1)

    def py_iterator(self):
        """ Python generator that extracts complete shots. Shots have arbitrary length."""
        for item in self.inputs:
            file_name = item[0]
            label = item[1]
            shot = self._load_complete_file_py(item)
            shot = self.preprocess_fn(shot)
            yield (shot, tf.one_hot(int(label),self.nr_classes), file_name)

    def process_file(self, input: Tuple[str, int, int, int]):
        """Extracts subshot. Input tuple is (file_path, classification. start_frame, end_frame)"""
        vid_shape = [self.frame_number,self.frame_size[0], self.frame_size[1],3]
        shot = tf.py_function(self._process_file_py, [input],tf.float32)
        shot.set_shape(vid_shape)
        shot = self.preprocess_fn(shot)
        return shot, tf.one_hot(int(input[1]),self.nr_classes)

    def split_classes(self, inputs):
        """Splits the dataset according to the class label."""
        result = dict()
        for element in inputs:
            result.get(int(element[1]),[]).append(element)
        return result

    def get_class_weights(self):
        return 1./self.counts*(np.max(self.counts))

    def _load_complete_file_py(self, input):
        """
        automatically pad for windowing.
        :param input: Data record
        :return: Loaded shot as numpy array
        """

        file_name = input[0]
        cap = cv2.VideoCapture(file_name)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frameStart = int(input[2])
        frameEnd = int(input[3])
        start_shot = 1000.*frameStart/fps

        duration = min(frameEnd-frameStart,32*self.stride*(self.frame_number-1))

        padded_duration = max(duration, self.stride*(self.frame_number-1)+1)
        buf = np.empty((padded_duration, self.frame_size[0], self.frame_size[1], 3), np.dtype('uint8'))

        cap.set(cv2.CAP_PROP_POS_MSEC,start_shot)
        output_fc=0
        ret = True
        while (output_fc < duration and ret):
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame,self.frame_size)
                buf[output_fc]=util.img_as_ubyte(exposure.equalize_hist(frame))
            output_fc+=1

        cap.release()
        return buf

    def load_complete_shot(self, file_name):
        """
        Used to load a single file and return its content as numpy array
        :param file_name:
        :return:
        """
        cap = cv2.VideoCapture(file_name)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frameEnd = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameStart = 0

        start_shot = 1000. * frameStart / fps

        duration = min(frameEnd - frameStart, 32 * self.stride * (self.frame_number - 1))

        padded_duration = max(duration, self.stride * (self.frame_number - 1) + 1)
        buf = np.empty((padded_duration, self.frame_size[0], self.frame_size[1], 3), np.dtype('uint8'))

        cap.set(cv2.CAP_PROP_POS_MSEC, start_shot)
        output_fc = 0
        ret = True
        while (output_fc < duration and ret):
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, self.frame_size)
                buf[output_fc] = util.img_as_ubyte(exposure.equalize_hist(frame))
            output_fc += 1

        cap.release()
        buf = self.preprocess_fn(buf)
        return buf

    def _process_file_py(self, input):

        file_name = input[0]
        cap = cv2.VideoCapture(file_name.numpy().decode('UTF-8'))
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frameStart = int(input[2])
        frameEnd = int(input[3])
        start_shot = 1000. * frameStart / fps
        end_shot = 1000. * frameEnd / fps
        duration = 1000. * (self.stride * (self.frame_number - 1)) / fps
        start_time = random.uniform(start_shot, end_shot - duration)
        max_frames = min(int(np.ceil((frameEnd-frameStart)/self.stride)),self.frame_number)
        buf = np.empty((max_frames, self.frame_size[0], self.frame_size[1], 3), np.dtype('uint8'))

        cap.set(cv2.CAP_PROP_POS_MSEC, start_time)
        fc = 0
        output_fc = 0
        ret = True
        stride_counter = 0
        reversed_x = random.choice([True, False])
        reversed_y = random.choice([True, False])
        while (output_fc < max_frames and ret):
            ret, frame = cap.read()
            if (stride_counter % self.stride == 0):
                if ret:
                    frame = cv2.resize(frame, self.frame_size)
                    buf[output_fc] = util.img_as_ubyte(exposure.equalize_hist(frame))
                output_fc += 1
            stride_counter += 1
            fc += 1

        cap.release()

        buf = np.pad(buf,((0,self.frame_number-max_frames),(0,0),(0,0),(0,0)),'reflect')

        if reversed_x:
            buf = np.flip(buf, 2)
        if reversed_y:
            buf = np.flip(buf, 1)

        return buf

    @staticmethod
    def get_args_from_config(config_file: str):
        """
        Parser config file and returns a dict that holds the keyword - parameter dictionary for the training, validation and
        test set.
        Usage:
            training_configs = DataLoader.get_args_from_config('config.yml')
            training_set = DataLoader(**training_configs.get('training'))
            validation_set = DataLoader(**training_configs.get('validation'))
            test_set = DataLoader(**training_configs.get('test'))

        :param config_file: Path to config file
        :return: Dictionary of keyword - parameter dictionaries for training, test, validation and loading without Dataset
        """
        # Load config
        stream = open('config.yml', 'r')
        config = yaml.load(stream, yaml.SafeLoader)
        model_config = config.get('model', dict())
        training_config = config.get('training', dict())

        # Preprocess Name
        preprocess_name = training_config.get('preprocess_name', None)
        if preprocess_name is None:
            # preprocess_name is None
            preprocess_name = model_config.get('base_model', '')
        if preprocess_name is None:
            # base model is None
            preprocess_name = ''

        base = {
            'stride' : model_config.get('stride', 3),
            'frame_size' : tuple(model_config.get('input_size', (224, 224, 3)))[0:2],
            'frame_number' : model_config.get('window_size', 16),
            'nr_classes' : model_config.get('nr_classes', 2),
            'preprocess_name': preprocess_name,
            'nr_threads': training_config.get('nr_threads', 2)
        }


        training_set_path = training_config.get('training_set','')
        validation_set_path = training_config.get('validation_set','')
        test_set_path = training_config.get('test_set','')

        training = {**base, 'dataset_path':training_set_path}
        validation = {**base, 'dataset_path':validation_set_path}
        test = {**base, 'dataset_path':test_set_path}

        result = {
            'training': training,
            'validation': validation,
            'test': test,
            'base': base
        }

        return result


