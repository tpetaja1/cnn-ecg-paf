
import datetime

from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import MaxPooling1D, AveragePooling1D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.regularizers import l2
from keras.optimizers import Adagrad, Adam, SGD
from keras.callbacks import CSVLogger

from data_handler_keras import DataHandler
from callbacks import EmptyCallback, SVMCallBack

from sklearn.svm import SVC


class KerasPAFClassifier:

    def __init__(self,
                 number_of_channels=6,
                 number_of_epochs=10,
                 mini_batch_size=10,
                 number_of_filters=128,
                 filter_length=32,
                 regularization_coefficient=2*1e-1,
                 pool_size=128,
                 dropout_rate=0.25,
                 fully_connected_layer_neurons=64,
                 learning_rate=0.001,
                 momentum=0.9,
                 pooling_type="max",
                 optimizer="adam",
                 convolution_activation="relu",
                 fully_connected_activation="relu",
                 output_activation="softmax",
                 classifier="mlp",
                 low_cutoff=1,
                 high_cutoff=40):

        self.filename = \
            "logs/log_{}.csv".format(
                datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        self.verbose = True
        self.log = True
        self.number_of_channels = number_of_channels
        self.data_handler = \
            DataHandler(number_of_channels=1,
                        number_of_negative_sets=50,
                        number_of_positive_sets=50,
                        number_of_test_sets=50,
                        low_cutoff=low_cutoff,
                        high_cutoff=high_cutoff,
                        verbose=self.verbose)
        self.data_handler.load_training_data()
        self.data_handler.load_test_data()
        self.data_handler.preprocess_data()

        self.mini_batch_size = mini_batch_size
        self.number_of_epochs = number_of_epochs

        self.training_errors = []
        self.test_errors = []

        self.number_of_filters = number_of_filters
        self.filter_length = filter_length
        self.regularization_coefficient = regularization_coefficient
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        self.fully_connected_layer_neurons = fully_connected_layer_neurons
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.pooling_type = pooling_type
        self.optimizer = optimizer
        self.convolution_activation = convolution_activation
        self.fully_connected_activation = fully_connected_activation
        self.output_activation = output_activation
        self.history = None

        self.csv_logger = EmptyCallback()
        self.svm_classifier = EmptyCallback()

        self.classifier = classifier
        self.define_classifier()
        self.define_callbacks()

        if self.log:
            self.data_handler.write_information(self)

        self.model = Sequential()

    def define_classifier(self):
        if self.classifier == "mlp":
            self.mlp_neurons = 32
        elif self.classifier == "gaussian_svm":
            self.svm = SVC(C=11., kernel="rbf", gamma=1./(2*2.85))

    def define_callbacks(self):
        if self.log:
            self.csv_logger = CSVLogger(self.filename, append=True)
        if self.classifier == "gaussian_svm":
            self.svm_classifier = SVMCallBack(self.data_handler.training_data,
                                              self.data_handler.training_labels,
                                              self.data_handler.test_data,
                                              self.data_handler.test_labels,
                                              self.svm)

    def create_model(self):

        self.model.add(Conv1D(input_shape=(38400, self.number_of_channels),
                              filters=self.number_of_filters,
                              kernel_size=self.filter_length,
                              padding="causal",
                              activation=self.convolution_activation,
                              use_bias=True,
                              kernel_initializer="glorot_normal",
                              bias_initializer="zeros",
                              kernel_regularizer=l2(self.regularization_coefficient)
                              ))

        if self.pooling_type == "max":
            self.model.add(MaxPooling1D(pool_size=self.pool_size,
                                        strides=None,
                                        padding="valid"))
        elif self.pooling_type == "mean":
            self.model.add(AveragePooling1D(pool_size=self.pool_size,
                                            strides=None,
                                            padding="valid"))

        self.model.add(Dropout(rate=self.dropout_rate))

        self.model.add(Flatten())

        self.model.add(Dense(units=self.fully_connected_layer_neurons,
                             activation=self.fully_connected_activation,
                             use_bias=True,
                             kernel_initializer="glorot_normal",
                             bias_initializer="zeros",
                             kernel_regularizer=l2(self.regularization_coefficient),
                             name="features"
                             ))

        self.model.add(Dropout(rate=self.dropout_rate))

        if self.classifier == "mlp":
            self.model.add(Dense(units=self.mlp_neurons,
                                 activation=self.fully_connected_activation,
                                 use_bias=True,
                                 kernel_initializer="glorot_normal",
                                 bias_initializer="zeros",
                                 kernel_regularizer=l2(self.regularization_coefficient)
                                 ))
            self.model.add(Dropout(rate=self.dropout_rate))

        self.model.add(Dense(units=2,
                             activation=self.output_activation,
                             use_bias=True,
                             kernel_initializer="glorot_normal",
                             bias_initializer="zeros",
                             kernel_regularizer=l2(self.regularization_coefficient)
                             ))

        print(self.model.summary())

    def run(self):

        if self.optimizer == "adam":
            optimizer = Adam(lr=self.learning_rate,
                             beta_1=0.9,
                             beta_2=0.999,
                             epsilon=1e-4)

        elif self.optimizer == "sgd":
            optimizer = SGD(lr=self.learning_rate,
                            momentum=self.momentum)

        elif self.optimizer == "adagrad":
            optimizer = Adagrad(lr=self.learning_rate,
                                epsilon=1e-4)

        self.model.compile(loss="binary_crossentropy",
                           optimizer=optimizer,
                           metrics=["accuracy"])

        self.history = \
            self.model.fit(self.data_handler.training_data,
                           self.data_handler.training_labels,
                           epochs=self.number_of_epochs,
                           batch_size=self.mini_batch_size,
                           verbose=1,
                           validation_data=(self.data_handler.test_data,
                                            self.data_handler.test_labels),
                           callbacks=[self.csv_logger, self.svm_classifier])


if __name__ == "__main__":
    p = KerasPAFClassifier(number_of_epochs=10)
    p.create_model()
    p.run()
