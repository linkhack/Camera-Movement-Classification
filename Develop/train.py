from camclassifier import DataLoader
from camclassifier.cnnlstm_model import build_model
import tensorflow as tf
from tensorflow import keras

tf.keras.backend.clear_session()

model = build_model()
model.summary()
training_set = DataLoader.DataLoader('annotation.flist', (299,299)).pipeline(1)
validation_set = DataLoader.DataLoader('val.flist', (299,299)).pipeline(1)

print(validation_set)
model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.CategoricalCrossentropy(), metrics=[keras.metrics.CategoricalAccuracy()])

history = model.fit(training_set, epochs=3, validation_data=validation_set)

model.evaluate(validation_set)
