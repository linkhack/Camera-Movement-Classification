from camclassifier.InferenceModel import InferenceModel
from camclassifier.DataLoader import DataLoader
import tensorflow as tf
import numpy as np
import tqdm
import yaml

# load config
stream = open('config.yml', 'r')
config = yaml.load(stream, yaml.SafeLoader)
model_config = config.get('model', dict())
inference_config = config.get('inference', dict())
training_config = config.get('training', dict())

# parse config
base_model_path = model_config.get('load_weights')
stride = model_config.get('stride', 3)
window_size = model_config.get('window_size', 16)
dataset_path = inference_config.get('dataset_path')
frame_size = tuple(model_config.get('input_size', [224, 224, 3])[0:2])
preprocess_name = training_config.get('preprocess_name', None)
if preprocess_name is None:
    # preprocess_name is None
    preprocess_name = model_config.get('base_model', '')
if preprocess_name is None:
    # base model is None
    preprocess_name = ''
nr_classes = model_config.get('nr_classes', 2)
nr_threads = training_config.get('nr_threads', 2)
inference_model = inference_config.get('inference_model', True)
batch_size = training_config.get('batch_size', 4)

# initialize dataset
dataset = DataLoader(dataset_path=dataset_path, frame_size=frame_size, frame_number=window_size, stride=stride,
                     preprocess_name=preprocess_name, nr_classes=nr_classes, nr_threads=nr_threads)
# Load base model
base_model = tf.keras.models.load_model(base_model_path)

# Evaluate
cm = np.zeros((nr_classes, nr_classes))
if inference_model:
    # Classification from whole shot
    test_set = dataset.py_iterator()
    end_model = InferenceModel(base_model=base_model,window_size=window_size,window_stride=stride)
    for x, y, file_name in tqdm.tqdm(test_set, total=dataset.length):
        y_predict = end_model.predict(x)
        y_predict = np.argmax(y_predict[0])
        y_true = np.argmax(y)
        cm[y_predict, y_true] += 1
else:
    # Classification from random subshot
    test_set = dataset.validation_pipeline(batch_size=batch_size)
    for x, y in tqdm.tqdm(test_set, total=int(dataset.length/batch_size)):
        y_predict = base_model.predict(x, batch_size=batch_size)
        y_predict = np.argmax(y_predict, axis=1)
        y_true = np.argmax(y, axis=1)
        for predict, true in zip(y_predict,y_true):
            cm[predict, true] += 1

print(cm)
acc = np.trace(cm) / np.sum(cm)
print(acc)
