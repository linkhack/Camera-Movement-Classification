from camclassifier.InferenceModel import InferenceModel
import camclassifier.DataLoader
import tensorflow as tf
import numpy as np
import tqdm

# load base_model
base_model_path = r"model_checkpoints/20200127133615815460/mymodel_15.h5"

loaded_model = tf.keras.models.load_model(base_model_path)

#create inference model
end_model = InferenceModel(loaded_model, window_stride=3)

# test set
dataset = camclassifier.DataLoader.DataLoader('annotation/test_shots.flist',(224,224), stride=3)
test_set = dataset.py_iterator()

cm_seq=np.zeros((2,2))

cm = np.zeros((2,2))
for x,y in tqdm.tqdm(test_set, total=dataset.length):
    y_predict = end_model.predict(x)
    y_predict = np.argmax(y_predict[0])
    y_true = np.argmax(y)
    cm[y_predict,y_true]+=1

print(cm)
acc = np.trace(cm)/np.sum(cm)
print(acc)


