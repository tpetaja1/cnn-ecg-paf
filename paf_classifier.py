
import theano
import theano.tensor as T
import numpy as np
import time
from sklearn.svm import SVC

from cnn import CNN
from data_handler import DataHandler


class PAFClassifier:

    def __init__(self, number_of_epochs=10):
        self.verbose = True
        self.number_of_channels = 2
        self.data_handler = \
            DataHandler(number_of_channels=self.number_of_channels,
                        number_of_negative_sets=50,
                        number_of_positive_sets=50,
                        number_of_test_sets=50,
                        verbose=self.verbose)
        self.data_handler.load_training_data()
        self.data_handler.load_test_data()
        self.data_handler.preprocess_data()

        self.mini_batch_size = 1
        self.model = CNN(number_of_channels=self.number_of_channels,
                         number_of_filters=12,
                         regularization_coefficient=1e0,
                         learning_rate=0.001,
                         filter_length=12,
                         pool_size=512,
                         fully_connected_layer_neurons=8,
                         momentum=0.9,
                         perform_normalization="no",
                         update_type="adam",
                         pool_mode="average_exc_pad")
        self.number_of_epochs = number_of_epochs

        self.training_errors = []
        self.test_errors = []
        self.classifier = SVC(C=11., kernel="rbf", gamma=1./(2*2.85))

    def create_models(self):

        if self.verbose:
            print("Creating Training model...")

        x = T.tensor3('x', dtype=theano.config.floatX)
        y = T.wvector('y')  # int16

        self.model.create_computational_graph(x, y)

        index = T.wscalar()  # int16

        self.train_model = \
            theano.function(
                [index],
                [self.model.cost, self.model.error,
                 self.model.negative_log_likelihood, self.model.penalty,
                 self.model.sensitivity, self.model.specificity],
                updates=self.model.updates,
                givens={x:
                        self.data_handler.training_data[
                            index * self.mini_batch_size:
                            (index + 1) * self.mini_batch_size],
                        y:
                        self.data_handler.training_labels[
                            index * self.mini_batch_size:
                            (index + 1) * self.mini_batch_size]
                        }
            )
        if self.verbose:
            print("Training model created.")

        if self.verbose:
            print("Creating Test model...")
        self.test_model = \
            theano.function(
                [x, y],
                [self.model.error,
                 self.model.sensitivity, self.model.specificity,
                 self.model.fully_connected_layer_output])

        if self.verbose:
            print("Test model created.")

        self.feature_extractor = theano.function(
            [x],
            self.model.fully_connected_layer_output)

    def train_neural_network(self):

        if self.verbose:
            print("Training the model...")

        for epoch_index in range(1, self.number_of_epochs + 1):

            start_time = time.time()

            if self.verbose:
                print("*** Epoch {}/{} ***\n".
                      format(epoch_index, self.number_of_epochs))

            """ Train model """
            if self.verbose:
                print("    ** Training **\n")
            train_errors = 0
            sensitivities = []
            specificities = []
            for training_set_index in range(
                    self.data_handler.total_number_of_training_data //
                    self.mini_batch_size):
                train_cost, train_error, nll, pen, \
                    sensitivity, specificity = \
                    self.train_model(training_set_index)
                train_errors += train_error

                sensitivities.append(sensitivity)
                specificities.append(specificity)

                if self.verbose and training_set_index % 300 == 0:
                    print("    NLL: {}".format(nll))
                    print("    Pen: {}".format(pen))
                    print()
            self.training_errors.append(train_errors)

            features = self.feature_extractor(
                self.data_handler.training_data.get_value(borrow=True))
            svm_errors = self.gaussian_svm(
                features,
                self.data_handler.training_labels.get_value(borrow=True))

            if self.verbose:
                print("    Train Errors: {}/{}, {:.2f}%".
                      format(train_errors,
                             self.data_handler.total_number_of_training_data,
                             100 * train_errors /
                             self.data_handler.total_number_of_training_data))
                print("    Train Sensitivity: {:.2f}%".
                      format(100 * np.mean(sensitivities)))
                print("    Train Specificity: {:.2f}%".
                      format(100 * np.mean(specificities)))
                print("    Train Cost: {}".format(train_cost))
                print("    NLL: {}".format(nll))
                print("    Pen: {}".format(pen))
                print("    Train SVM Errors: {}/{}, {:.2f}%".
                      format(svm_errors,
                             self.data_handler.total_number_of_training_data,
                             100 * svm_errors /
                             self.data_handler.total_number_of_training_data))
                print()

            """ Test model """
            if self.verbose:
                print("    ** Testing **\n")
            test_errors, sensitivity, specificity, features = \
                self.test_model(
                    self.data_handler.test_data.get_value(borrow=True),
                    self.data_handler.test_labels.get_value(borrow=True))

            self.test_errors.append(test_errors)

            svm_errors = self.gaussian_svm(
                features,
                self.data_handler.test_labels.get_value(borrow=True),
                train=False)

            if self.verbose:
                print("    Test Errors: {}/{}, {:.2f}%".
                      format(test_errors,
                             self.data_handler.total_number_of_test_data,
                             100 * test_errors /
                             self.data_handler.total_number_of_test_data))
                print("    Test Sensitivity: {:.2f}%".
                      format(100 * sensitivity))
                print("    Test Specificity: {:.2f}%".
                      format(100 * specificity))
                print("    Test SVM Errors: {}/{}, {:.2f}%".
                      format(svm_errors,
                             self.data_handler.total_number_of_test_data,
                             100 * svm_errors /
                             self.data_handler.total_number_of_test_data))
                print()

            epoch_time = time.time() - start_time

            if self.verbose:
                print("Epoch time: {:.2f}s\n".format(epoch_time))

        if self.verbose:
            print("Model trained.")

    def gaussian_svm(self, x, y, train=True):
        if train:
            self.classifier.fit(x, y)
        predictions = self.classifier.predict(x)
        error = np.sum(np.not_equal(predictions, y))
        return error


if __name__ == "__main__":
    p = PAFClassifier(number_of_epochs=50)
    p.create_models()
    p.train_neural_network()
