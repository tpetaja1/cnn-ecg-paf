
import numpy as np
from keras.callbacks import Callback
from keras import backend as K


class EmptyCallback(Callback):
    pass


class SVMCallBack(Callback):

    def __init__(self, training_x, training_y, test_x, test_y, svm):
        self.training_x = training_x
        self.training_y = np.argmax(training_y, axis=1)
        self.test_x = test_x
        self.test_y = np.argmax(test_y, axis=1)
        self.svm = svm

    def on_epoch_end(self, epoch, logs={}):
        feature_extractor = \
            K.function([self.model.input] + [K.learning_phase()],
                       [self.model.get_layer("features").output])
        """ Train SVM """
        x = self.training_x
        y = self.training_y
        features = feature_extractor([x])[0]
        train_acc = self.gaussian_svm(features, y, train=True)

        """ Test SVM """
        x = self.test_x
        y = self.test_y
        features = feature_extractor([x])[0]
        test_acc = self.gaussian_svm(features, y, train=False)

        print()
        print("Gaussian SVM Training accuracy: {:.2f}".format(train_acc))
        print("Gaussian SVM Test accuracy: {:.2f}".format(test_acc))

    def gaussian_svm(self, features, y, train=True):
        if train:
            self.svm.fit(features, y)
        predictions = self.svm.predict(features)
        accuracy = np.sum(np.equal(predictions, y))
        return accuracy
