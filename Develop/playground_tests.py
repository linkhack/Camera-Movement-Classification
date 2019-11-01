from camclassifier import CNNLSTM
import cv2
import numpy as np
import tensorflow as tf

testi = CNNLSTM()

im = cv2.imread("./Data/test_img/Thinking-of-getting-a-cat.png")

im = np.ones((1, 8, 299, 299, 3))


blub = testi.predict(im)

print(blub)
print(blub.shape)