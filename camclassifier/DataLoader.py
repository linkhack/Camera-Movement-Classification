import tensorflow as tf
import pandas as pd
import typing
import csv
import os
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
        return self.dataset.map(self.process_file, num_parallel_calls=4).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    def process_file(self, input: Tuple[str, int, int, int]):

        vid_shape = [self.frame_number,self.frame_size[0], self.frame_size[1],3]
        shot = tf.py_function(self._process_file_py, [input],tf.float32)
        shot.set_shape(vid_shape)

        return shot, tf.one_hot(int(input[1]),3)

    def split_classes(self, inputs):
        result = [[],[],[]]
        for element in inputs:
            result[int(element[1])].append(element)
        return result

    def get_class_weights(self):
        return 1./self.counts*(self.length)/3.


    def _process_file_py(self, input):

        file_name = input[0]
        cap = cv2.VideoCapture(file_name.numpy().decode('UTF-8'))
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frameStart = int(input[2])
        frameEnd = int(input[3])
        start_shot = 1000.*frameStart/fps
        end_shot = 1000.*frameEnd/fps
        duration = 1000.*(self.stride*(self.frame_number-1))/fps
        start_time = random.uniform(start_shot,end_shot-duration)
        buf = np.empty((self.frame_number, self.frame_size[0], self.frame_size[1], 3), np.dtype('uint8'))

        cap.set(cv2.CAP_PROP_POS_MSEC,start_time)
        fc = 0
        output_fc=0
        ret = True
        stride_counter = 0
        while (output_fc < self.frame_number and ret):
            ret, frame = cap.read()
            if (stride_counter%self.stride == 0):
                if ret:
                    frame = cv2.resize(frame,self.frame_size)
                    buf[output_fc]=frame
                output_fc+=1
            stride_counter+=1
            fc+=1

        cap.release()

        return buf/122.5-1

