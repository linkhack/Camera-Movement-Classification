import tensorflow as tf
import pandas as pd
import typing
import csv
import random
from typing import Tuple, List
import numpy as np
import cv2

class DataLoader:
    def __init__(self, dataset_path: str, frame_size: Tuple[int, int], frame_number: int = 16, stride:int = 1):
        self.inputs = self.process_flist(dataset_path)
        self.frame_size = frame_size
        self.frame_number = int(frame_number)
        self.stride = int(stride)
        self.dataset = tf.data.Dataset.from_tensor_slices(self.inputs)

    def process_flist(self, dataset_path: str) -> List[Tuple[str, str]]:
        shots = []
        annotation = []
        with open(dataset_path,'r') as file:
            content = file.read()
            lines = content.splitlines()
            lines = [tuple(line.split()) for line in lines]
        return lines

    def pipeline(self, batch_size: int):
        return self.dataset.shuffle(len(self.inputs)).map(self.process_file, num_parallel_calls=4).batch(batch_size).prefetch(1)

    def process_file(self, input: Tuple[str, str, str, str]):

        vid_shape = [self.frame_number,self.frame_size[0], self.frame_size[1],3]
        shot = tf.py_function(self._process_file_py, [input],tf.float32)
        shot.set_shape(vid_shape)

        return (shot, input[1])

    def _process_file_py(self, input):

        file_name = input[0]

        cap = cv2.VideoCapture(str(file_name))
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frameStart = int(input[2])
        frameEnd = int(input[3])
        buf = np.empty((self.frame_number, frameHeight, frameWidth, 3), np.dtype('uint8'))/255.
        start_frame = int(random.uniform(frameStart, int(frameEnd-int(self.stride*int(self.frame_number-1)))))

        # TODO: use msec if less errors
        cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
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

        return (buf, input[1])