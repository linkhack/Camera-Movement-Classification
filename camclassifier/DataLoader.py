import tensorflow as tf
import pandas as pd
import typing
import csv
import os
from skimage import exposure, util
from pathlib import Path
import random
from typing import Tuple, List
import numpy as np
import cv2
import keras
import keras.backend as K


class DataLoader:
    def __init__(self, dataset_path: str, frame_size: Tuple[int, int], frame_number: int = 16, stride:int = 1):
        self.labels = []
        self.inputs = self.process_flist(dataset_path)
        unique, counts = np.unique(self.labels, return_counts=True)
        print(counts)
        self.class_probabilities = list(counts/len(self.inputs))
        print(self.class_probabilities)
        self.classes = self.split_classes(self.inputs)
        self.frame_size = frame_size
        self.frame_number = int(frame_number)
        self.stride = int(stride)
        self.counts = counts
        self.length = len(self.inputs)
        self.dataset = tf.data.Dataset.from_tensor_slices(self.inputs)
        self.pan = tf.data.Dataset.from_tensor_slices(self.classes[0])
        self.tilt = tf.data.Dataset.from_tensor_slices(self.classes[1])
        self.tracking = tf.data.Dataset.from_tensor_slices(self.classes[2])

    def process_flist(self, dataset_path: str) -> List[Tuple[str, str]]:
        shots = []
        annotation = []
        with open(dataset_path,'r') as file:
            content = file.read()
            lines = content.splitlines()
            lines = [tuple(line.split()) for line in lines]
        self.labels = [line[1] for line in lines]
        return lines

    def training_pipeline(self, batch_size: int):
        dataset = tf.data.experimental.sample_from_datasets([self.pan, self.tilt, self.tracking],[0.5,0.5,0.5])
        return dataset.map(self.process_file, num_parallel_calls=4).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    def validation_pipeline(self, batch_size: int):
        return self.dataset.repeat().shuffle(self.length).map(self.process_file, num_parallel_calls=4).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    def test_pipeline(self, batch_size: int):
        return self.dataset.map(self.load_whole_file, num_parallel_calls=4).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    def py_iterator(self):
        for item in self.inputs:
            file_name = item[0]
            label = item[1]
            shot = self._load_complete_file_py(item)
            yield (shot, tf.one_hot(int(label),3))

    def process_file(self, input: Tuple[str, int, int, int]):

        vid_shape = [self.frame_number,self.frame_size[0], self.frame_size[1],3]
        shot = tf.py_function(self._process_file_py, [input],tf.float32)
        shot.set_shape(vid_shape)

        return shot, tf.one_hot(int(input[1]),3)

    def load_whole_file(self, input):
        vid_shape = [None, self.frame_size[0], self.frame_size[1],3]
        shot = tf.py_function(self._load_complete_file_py, [input],tf.float32)
        shot.set_shape(vid_shape)
        shot = keras.applications.vgg19.preprocess_input(shot)
        return shot, tf.one_hot(int(input[1]),3)

    def split_classes(self, inputs):
        result = [[],[],[]]
        for element in inputs:
            result[int(element[1])].append(element)
        return result

    def get_class_weights(self):
        return 1./self.counts*(np.max(self.counts))

    def _load_complete_file_py(self, input):
        """
        automatically pad for windowing
        :param input:
        :return:
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
        print(duration)
        padded_duration = max(duration, self.stride*(self.frame_number-1)+1)
        print(padded_duration)
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
        buf =buf.astype(dtype=np.float32)
        print('Loaded')
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
        buf = np.empty((self.frame_number, self.frame_size[0], self.frame_size[1], 3), np.dtype('uint8'))

        cap.set(cv2.CAP_PROP_POS_MSEC, start_time)
        fc = 0
        output_fc = 0
        ret = True
        stride_counter = 0
        reversed_x = random.choice([True, False])
        reversed_y = random.choice([True, False])
        while (output_fc < self.frame_number and ret):
            ret, frame = cap.read()
            if (stride_counter % self.stride == 0):
                if ret:
                    frame = cv2.resize(frame, self.frame_size)
                    buf[output_fc] = util.img_as_ubyte(exposure.equalize_hist(frame))
                output_fc += 1
            stride_counter += 1
            fc += 1

        cap.release()

        if reversed_x:
            buf = np.flip(buf, 2)
        if reversed_y:
            buf = np.flip(buf, 1)

        return buf

