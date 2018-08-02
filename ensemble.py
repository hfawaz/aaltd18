import keras
import numpy as np
from utils.utils import calculate_metrics
import gc

class Classifier_ENSEMBLE:
    def __init__(self, output_directory, input_shape, nb_classes, verbose=False):
        self.output_directory = output_directory
        self.model1 = keras.models.load_model(self.output_directory.
                                    replace('ensemble','resnet')
                                    +'best_model.hdf5')
        self.model2 = keras.models.load_model(self.output_directory.
                                    replace('ensemble','resnet_augment')
                                    +'best_model.hdf5')
        if (verbose == True):
            self.model1.summary()
            self.model2.summary()
        self.verbose = verbose

    def fit(self, x_test, y_true):
        # no training since models are pre-trained

        y_pred1 = self.model1.predict(x_test)
        y_pred2 = self.model2.predict(x_test)

        y_pred = (y_pred1+y_pred2)/2

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        df_metrics = calculate_metrics(y_true, y_pred, 0.0)

        df_metrics.to_csv(self.output_directory+'df_metrics.csv', index=False)

        keras.backend.clear_session()

        gc.collect()