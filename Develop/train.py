from camclassifier import DataLoader
from camclassifier.cnnlstm_model import build_model
import camclassifier.utils.utils as utils
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.metrics as metrics
import tensorflow.keras.losses as losses
import numpy as np
tf.keras.backend.clear_session()
import os,datetime

model = build_model()
model.summary()

stride=5
training_set = DataLoader.DataLoader('annotation/train_shots.flist', (224,224), stride=stride)
trainings_pipeline = training_set.training_pipeline(4)



class_weight = training_set.get_class_weights()
#class_weight = None
print(f"Class weights: {class_weight}")
validation_set = DataLoader.DataLoader('annotation/val_shots.flist', (224,224), stride=stride).validation_pipeline(1)
test_set = DataLoader.DataLoader('annotation/test_shots.flist',(224,224), stride=stride).validation_pipeline(1)



model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001, epsilon=0.1), loss=losses.BinaryCrossentropy(), metrics=[metrics.BinaryAccuracy(), keras.metrics.Recall(), keras.metrics.Precision()])

log_dir = os.path.join('model_logs', utils.date_uid())
model_dir = os.path.join('model_checkpoints', utils.date_uid())

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir,'mymodel_{epoch}.h5'),
        # Path where to save the model
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the `val_loss` score has improved.
        save_best_only=False,
        monitor='val_loss',
        verbose=1),
    keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=200, profile_batch=0, histogram_freq=1),
    keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor='val_loss',
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-3,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=15,
        restore_best_weights=True,
        verbose=1)
]

history = model.fit(trainings_pipeline, epochs=100, validation_data=validation_set, callbacks=callbacks,validation_steps=366, verbose=2, class_weight=class_weight)

cm = np.zeros((2,2))
for x,y in test_set:
    y_predict = model.predict(x, batch_size=1)
    y_predict = np.argmax(y_predict[0])
    y_true = np.argmax(y[0])
    cm[y_predict,y_true]+=1

print(cm)
acc = np.trace(cm)/np.sum(cm)