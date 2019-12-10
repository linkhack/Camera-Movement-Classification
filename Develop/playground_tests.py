from camclassifier.cnnlstm_model import build_model
from camclassifier import DataLoader
import camclassifier
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import keras

#model = tf.keras.models.load_model('./model_checkpoints/20191126193611240278/mymodel_5.h5')

dataset = DataLoader.DataLoader('test.flist',(224,224),stride=3)

for x,y in dataset.py_iterator():
    print(x.shape)
    print(y)