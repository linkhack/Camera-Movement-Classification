from camclassifier import DataLoader
from camclassifier.cnnlstm_model import CNNLSTM
import tensorflow as tf
import keras

model = CNNLSTM()
training_set = DataLoader.DataLoader('annotation.flist', (299,299)).pipeline(16)
validation_set = DataLoader.DataLoader('val.flist', (299,299)).pipeline(16)

model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.CategoricalCrossentropy(), metrics=keras.metrics.CategoricalAccuracy())

history = model.fit(x=training_set, epochs=3)

model.evaluate(validation_set)
