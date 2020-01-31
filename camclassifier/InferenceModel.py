import numpy as np
import yaml
import tensorflow as tf
from typing import Dict
import csv

class InferenceModel:

    def __init__(self, base_model, window_size=16, window_stride=3, nr_classes=2, class_dict:Dict[str,int]=None):
        """
        Creates a Model for inference. Used tp Classify an arbitrarily long shot using a sliding window approach.

        :param base_model: Model used to classify each window
        :param window_size: Window size in frames
        :param window_stride: Take every stride-th frame.
        :param nr_classes: nr_classes: Number of classes to classify.
        """
        self.base_model = base_model
        self.window_size = window_size
        self.window_stride = window_stride
        self.nr_classes = nr_classes
        self.class_dict = class_dict
        # reverse so we have encoding: name
        self.class_dict = {v: k for k,v in self.class_dict.items()}

    def predict(self, shot, movie_id:str = None, shot_id: str = None, csv_file:str = None):
        """
        Classifies arbitrarily long shot using a sliding window approach. The final score is the average over all window scores

        :param shot: Arbitrary long shot. Has shape (frames, width, height, channels)
        :return: Softmax score
        """
        shot = np.squeeze(shot)
        length = shot.shape[0]
        nr_windows = length-(self.window_size-1)*self.window_stride
        average = np.zeros((1,self.nr_classes))
        for i in range(min(nr_windows,128)):
            window = shot[i:i+(self.window_size-1)*self.window_stride+1:self.window_stride]
            window = np.expand_dims(window,0)
            result = self.base_model.predict(window, batch_size=1)
            average=average+(result-average)/(i+1)
        label = np.argmax(average[0])
        label = self.class_dict.get(label,'label not found')
        result = np.array((movie_id,shot_id, label))
        if csv_file is not None:
            with open(csv_file, 'a+', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(result)

        return result

    def batch_predict(self, shots, csv_file:str = None):
        """
        Classifies a batch of shots. Applies predict to each shot.

        :param shots: List of a tuple with arbitrarily long shots, movie_id and shot_id
        :return: List of softmax scores.
        """
        scores = []
        for shot in shots:
            scores.append(self.predict(*shot, csv_file))
        scores = np.array(scores)
        return scores

    @staticmethod
    def build_model_from_config(config_file: str):
        # load config
        stream = open('config.yml', 'r')
        config = yaml.load(stream, yaml.SafeLoader)
        model_config = config.get('model', dict())

        # parse config
        base_model_path = model_config.get('load_weights')
        stride = model_config.get('stride', 3)
        window_size = model_config.get('window_size', 16)
        nr_classes = model_config.get('nr_classes', 2)
        class_dict = model_config.get('class_dict',dict())

        # load base model
        base_model = tf.keras.models.load_model(base_model_path)

        return InferenceModel(base_model,window_size,stride,nr_classes,class_dict)