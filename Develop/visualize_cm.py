import numpy as np
import pandas as pd
from camclassifier.utils.confusion_matrix_pretty_print import pretty_plot_confusion_matrix

if __name__ == '__main__':
    cm = np.array([[421., 72., 151.],
                   [46., 27., 13.],
                   [2., 0., 14.]]
                  ).T
    cmap = 'PuRd'
    cm_df = pd.DataFrame(cm, index=['Pan','Tilt','Tracking'], columns=['Pan','Tilt','Tracking'])
    pretty_plot_confusion_matrix(cm_df, cmap=cmap, pred_val_axis='y', show_null_values=True)
