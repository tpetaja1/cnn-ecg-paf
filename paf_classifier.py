
import theano
import theano.tensor as T
import numpy as np

from cnn import CNN
from data_handler import DataHandler


class PAFClassifier():

    def __init__(self, number_of_epochs=10):
        self.verbose = True
        self.data_handler = DataHandler(number_of_negative_sets=50,
                                        number_of_positive_sets=50,
                                        number_of_test_sets=50,
                                        verbose=self.verbose)
        self.data_handler.load_training_data()
        self.data_handler.load_test_data()

        self.mini_batch_size = 50
        self.model = CNN(number_of_filters=128,
                         regularization_coefficient=1,
                         learning_rate=0.09,
                         filter_length=32,
                         mini_batch_size=self.mini_batch_size,
                         pool_size=128,
                         fully_connected_layer_neurons=64,
                         momentum=0.9,
                         perform_normalization="only input")
        self.number_of_epochs = number_of_epochs

        self.training_errors = []
        self.test_errors = []

    def create_models(self):

        if self.verbose:
            print("Creating Training model...")

        x = T.dmatrix('x')
        y = T.ivector('y')

        self.model.create_computational_graph(x, y)

        index = T.lscalar()

        self.train_model = \
            theano.function(
                [index],
                (self.model.cost, self.model.error,
                 self.model.negative_log_likelihood, self.model.penalty,
                 self.model.sensitivity, self.model.specificity),
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
                [index],
                (self.model.cost, self.model.error,
                 self.model.negative_log_likelihood, self.model.penalty,
                 self.model.sensitivity, self.model.specificity),
                givens={x:
                        self.data_handler.test_data[
                            index * self.mini_batch_size:
                            (index + 1) * self.mini_batch_size],
                        y:
                        self.data_handler.test_labels[
                            index * self.mini_batch_size:
                            (index + 1) * self.mini_batch_size]
                        }
            )
        if self.verbose:
            print("Test model created.")

    def train_neural_network(self):

        if self.verbose:
            print("Training the model...")

        for epoch_index in range(1, self.number_of_epochs + 1):

            if self.verbose:
                print("    *** Epoch {}/{} ***\n".
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

                if self.verbose and training_set_index % 5 == 0:
                    print("    NLL: {}".format(nll))
                    print("    Pen: {}".format(pen))
                    print()
            self.training_errors.append(train_errors)

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
                print()

            """ Test model """
            if self.verbose:
                print("    ** Testing **\n")
            test_errors = 0
            sensitivities = []
            specificities = []
            for test_set_index in range(
                    self.data_handler.total_number_of_test_data //
                    self.mini_batch_size):
                test_cost, test_error, nll, pen,\
                    sensitivity, specificity = \
                    self.test_model(test_set_index)
                test_errors += test_error
                sensitivities.append(sensitivity)
                specificities.append(specificity)

                if self.verbose and test_set_index % 5 == 0:
                    print("    NLL: {}".format(nll))
                    print("    Pen: {}".format(pen))
                    print()
            self.test_errors.append(test_errors)

            if self.verbose:
                print("    Test Errors: {}/{}, {:.2f}%".
                      format(test_errors,
                             self.data_handler.total_number_of_test_data,
                             100 * test_errors /
                             self.data_handler.total_number_of_test_data))
                print("    Test Sensitivity: {:.2f}%".
                      format(100 * np.mean(sensitivities)))
                print("    Test Specificity: {:.2f}%".
                      format(100 * np.mean(specificities)))
                print("    Test Cost: {}".format(test_cost))
                print("    NLL: {}".format(nll))
                print("    Pen: {}".format(pen))
                print()

        if self.verbose:
            print("Model trained.")


if __name__ == "__main__":
    p = PAFClassifier(number_of_epochs=50)
    p.create_models()
    p.train_neural_network()
