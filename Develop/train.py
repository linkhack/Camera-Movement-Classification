from camclassifier import DataLoader
from camclassifier.cnnlstm_model import build_model
import tensorflow as tf
from tensorflow import keras
import numpy as np
tf.keras.backend.clear_session()
import os

model = build_model()
model.summary()
training_set = DataLoader.DataLoader('annotation.flist', (224,224)).pipeline(2)
validation_set = DataLoader.DataLoader('val.flist', (224,224)).pipeline(1)
test_set = DataLoader.DataLoader('test.flist',(224,224)).pipeline(1)

print(validation_set)


model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.CategoricalCrossentropy(), metrics=[keras.metrics.CategoricalAccuracy(), keras.metrics.Recall(), keras.metrics.Precision()])

log_dir = os.path.join('model_logs','batch2','efilms_only')
model_dir = os.path.join('model_checkpoints')

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir,'mymodel_{epoch}.h5'),
        # Path where to save the model
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the `val_loss` score has improved.
        save_best_only=True,
        monitor='val_loss',
        verbose=1),
    keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=200, profile_batch=0, histogram_freq=1),
    keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor='val_loss',
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-2,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=3,
        verbose=1)
]

history = model.fit(training_set, epochs=25, validation_data=validation_set, callbacks=callbacks ,validation_steps=423, verbose=2)

cm = np.zeros((3,3))
for x,y in test_set:
    y_predict = model.predict(x)
    y_predict = np.argmax(y_predict[0])
    y_true = np.argmax(y[0])
    cm[y_predict,y_true]+=1

print(cm)
