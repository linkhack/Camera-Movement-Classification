from camclassifier.InferenceModel import InferenceModel
import camclassifier.DataLoader
import tensorflow as tf
import numpy as np
import tqdm

# load base_model
base_model_path = r"E:\repos\Python\Camera-Movement-Classification\Develop\model_checkpoints\20191205131529604919\mymodel_4.h5"

loaded_model = tf.keras.models.load_model(base_model_path)

#create inference model
end_model = InferenceModel(loaded_model, window_stride=3)

# test set
dataset = camclassifier.DataLoader.DataLoader('test.flist',(224,224), stride=3)
test_set = dataset.py_iterator()
test_squence = dataset.training_pipeline(4)

cm_seq=np.zeros((3,3))

for x,y in tqdm.tqdm(test_squence, total=dataset.length/4):
    y_predict = loaded_model.predict(x, batch_size=4)
    y_predict = np.argmax(y_predict[0])
    y_true = np.argmax(y[0])
    cm_seq[y_predict,y_true]+=1

cm = np.zeros((3,3))
for x,y in tqdm.tqdm(test_set, total=dataset.length):
    y_predict = end_model.predict(x)
    y_predict = np.argmax(y_predict[0])
    y_true = np.argmax(y)
    cm[y_predict,y_true]+=1

print(cm)
acc = np.trace(cm)/np.sum(cm)
print(acc)

print(cm_seq)
acc = np.trace(cm_seq)/np.sum(cm_seq)
print(acc)

