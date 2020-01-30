from camclassifier.DataLoader import DataLoader
from camclassifier.Camera_Movement_Classifier import build_model_from_config
import camclassifier.utils.utils as utils
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.metrics as metrics
import tensorflow.keras.losses as losses
import numpy as np

tf.keras.backend.clear_session()
import os
import yaml

# load config
stream = open('config.yml', 'r')
config = yaml.load(stream, yaml.SafeLoader)
model_config = config.get('model', dict())
training_config = config.get('training', dict())

# build model
restore_weights = model_config.get('load_weights', None)
if restore_weights is not None:
    model = tf.keras.models.load_model(restore_weights)
else:
    model = build_model_from_config('config.yml')
model.summary()

# build data pipeline
# parse config
dataset_args = DataLoader.get_args_from_config('config.yml')
batch_size = training_config.get('batch_size', 4)

# create datasets
training_set = DataLoader(**dataset_args.get('training'))
validation_set = DataLoader(**dataset_args.get('validation'))
test_set = DataLoader(**dataset_args.get('test'))

# create dataset iterators
trainings_pipeline = training_set.training_pipeline(batch_size)
validation_pipeline = validation_set.validation_pipeline(batch_size)
test_pipeline = validation_set.validation_pipeline(batch_size)

# Get class weights
use_class_weights = training_config.get('use_class_weights', True)
if use_class_weights:
    class_weight = training_set.get_class_weights()
else:
    class_weight = None
print(f"Class weights: {class_weight}")

# Parse Optimizer settings
learning_rate = training_config.get('adam_lr', 0.0001)
epsilon = training_config.get('adam_epsilon', 0.1)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon),
              loss=losses.BinaryCrossentropy(),
              metrics=[metrics.BinaryAccuracy(), keras.metrics.Recall(), keras.metrics.Precision()])

# Setup logging
# Create identifier from config
identifier = model_config.get('base_model', 'VGG16') + "_" + model_config.get('feature_layer', None) + "_" + str(
    model_config.get('trainable_features', False)) + "_" + model_config.get('temporal', 'LSTM')
if model_config.get('temporal', 'LSTM') == 'LSTM':
    identifier = identifier + "_" + str(model_config.get('LSTM_size', None)).replace(', ', '_')
elif model_config.get('temporal', 'LSTM') == 'CONV':
    identifier = identifier + "_" + str(model_config.get('CONV_filter', None)).replace(', ', '_') + "_" + str(
        model_config.get('CONV_filter_sizes', None)).replace(', ', '_')
identifier = identifier + "_" + str(model_config.get('stride', 3))

model_logs_basepath = training_config.get('model_logs_basepath', 'model_logs')
model_checkpoints_basepath = training_config.get('model_checkpoints_basepath', 'model_checkpoints')

log_dir = os.path.join(model_logs_basepath, identifier + '_' + utils.date_uid())
model_dir = os.path.join(model_checkpoints_basepath, identifier + '_' + utils.date_uid())

# create folders if needed
if not os.path.exists(model_logs_basepath):
    os.mkdir(model_logs_basepath)
if not os.path.exists(model_checkpoints_basepath):
    os.mkdir(model_checkpoints_basepath)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

print(log_dir)
print(model_dir)

# early stopping and epochs
max_epochs = training_config.get('max_epochs', 100)
steps_per_epoch = training_config.get('max_epochs', None)
patience = training_config.get('early_stopping_patience', 15)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, 'mymodel_{epoch}.h5'),
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
        patience=patience,
        restore_best_weights=True,
        verbose=1)
]

history = model.fit(trainings_pipeline, epochs=max_epochs, steps_per_epoch=steps_per_epoch,
                    validation_data=validation_pipeline, callbacks=callbacks, validation_steps=validation_set.length,
                    verbose=2, class_weight=class_weight)

# Test on test set
nr_classes = model_config.get('nr_classes', 2)
cm = np.zeros((nr_classes, nr_classes))
for x, y in test_pipeline:
    y_predict = model.predict(x, batch_size=batch_size)
    y_predict = np.argmax(y_predict[0])
    y_true = np.argmax(y[0])
    cm[y_predict, y_true] += 1

print("Confusion matrix on test set:")
print(cm)
acc = np.trace(cm) / np.sum(cm)
print("Overall accuracy:")
print(acc)
