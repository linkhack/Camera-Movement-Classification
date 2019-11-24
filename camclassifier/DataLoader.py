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
        self.inputs = self.process_flist(dataset_path)
        self.frame_size = frame_size
        self.frame_number = int(frame_number)
        self.stride = int(stride)
        self.labels = []
        self.dataset = tf.data.Dataset.from_tensor_slices(self.inputs)

    def process_flist(self, dataset_path: str) -> List[Tuple[str, str]]:
        shots = []
        annotation = []
        with open(dataset_path,'r') as file:
            content = file.read()
            lines = content.splitlines()
            lines = [tuple(line.split()) for line in lines]
        self.labels = [line[1] for line in lines]
        return lines

    def pipeline(self, batch_size: int):
        return self.dataset.shuffle(len(self.inputs)).map(self.process_file, num_parallel_calls=4).batch(batch_size).prefetch(1)

    def process_file(self, input: Tuple[str, int, int, int]):

        vid_shape = [self.frame_number,self.frame_size[0], self.frame_size[1],3]
        shot = tf.py_function(self._process_file_py, [input],tf.float32)
        shot.set_shape(vid_shape)

        return shot, tf.one_hot(int(input[1]),3)

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
        end_shot = frameEnd/fps
        duration = (self.stride*(self.frame_number-1))/fps
        start_time = random.uniform(start_shot,end_shot-duration)
        buf = np.empty((self.frame_number, self.frame_size[0], self.frame_size[1], 3), np.dtype('uint8'))/255.

        cap.set(cv2.CAP_PROP_POS_MSEC,start_time)
        fc = 0
        output_fc=0
        ret = True
        stride_counter = 0
        while (output_fc < self.frame_number and ret):
            ret, frame = cap.read()
            if stride_counter%self.stride == 0:
                frame = cv2.resize(frame,self.frame_size)
                buf[output_fc]=frame
                output_fc+=1
            stride_counter+=1
            fc+=1

        cap.release()

        return buf