from camclassifier.InferenceModel import InferenceModel
import camclassifier.DataLoader
import tensorflow as tf
import numpy as np

# load base_model
base_model_path = r"E:\repos\Python\Camera-Movement-Classification\Develop\model_checkpoints\20191205131529604919\mymodel_4.h5"

loaded_model = tf.keras.models.load_model(base_model_path)

#create inference model
end_model = InferenceModel(loaded_model, window_stride=3)

# test set
test_set = camclassifier.DataLoader.DataLoader('test.flist',(224,224), stride=3).py_iterator()

cm = np.zeros((3,3))
for x,y in test_set:
    y_predict = end_model.predict(x)
    y_predict = np.argmax(y_predict[0])
    y_true = np.argmax(y[0])
    cm[y_predict,y_true]+=1

print(cm)
acc = np.trace(cm)/np.sum(cm)
print(acc)