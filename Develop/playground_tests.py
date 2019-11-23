from camclassifier.cnnlstm_model import CNNLSTM, build_model
from camclassifier import DataLoader
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import keras

images = np.random.rand(100,16,299,299,3)
label = np.random.randint(0,3,100)
label = keras.utils.to_categorical(label,3)

dataset = tf.data.Dataset.from_tensor_slices((images, label))
dataset = dataset.batch(16).prefetch(1)

print(dataset)

dataset2 = DataLoader.DataLoader('annotation.flist', (299,299)).pipeline(16)
print(dataset2)

model = build_model()
print(model.summary())
model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.CategoricalCrossentropy(), metrics=[keras.metrics.CategoricalAccuracy()])

history = model.fit(x=dataset,  validation_data=dataset2, validation_steps=10)