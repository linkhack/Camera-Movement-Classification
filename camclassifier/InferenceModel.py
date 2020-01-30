import numpy as np

class InferenceModel:

    def __init__(self, base_model=None, window_size=16, window_stride=1):
        self.base_model = base_model
        self.window_size = window_size
        self.window_stride = window_stride

    def predict(self, shot):
        shot = np.squeeze(shot)
        length = shot.shape[0]
        nr_windows = length-(self.window_size-1)*self.window_stride
        average = np.zeros((1,2))
        for i in range(min(nr_windows,32)):
            window = shot[i:i+(self.window_size-1)*self.window_stride+1:self.window_stride]
            window = np.expand_dims(window,0)
            result = self.base_model.predict(window, batch_size=1)
            average=average+(result-average)/(i+1)

        return average
