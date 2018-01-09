
import theano
import theano.tensor as T

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
        self.sub_ecg_length = 38400  # 5 min
        self.number_of_sub_ecgs = 6
        self.mini_batch_size = 10
        self.model = CNN(number_of_filters=16,
                         regularization_coefficient=1e-1,
                         learning_rate=0.01,
                         filter_length=16,
                         mini_batch_size=self.mini_batch_size,
                         pool_size=2,
                         fully_connected_layer_neurons=16,
                         momentum=0.9)
        self.number_of_epochs = number_of_epochs

    def create_models(self):

        if self.verbose:
            print("Creating Training model...")

        x = T.dmatrix('x')
        y = T.ivector('y')

        self.model.create_computational_graph(x, y)

        set_index = T.lscalar()
        index = T.lscalar()

        self.train_model = \
            theano.function(
                [set_index, index],
                (self.model.cost, self.model.error),
                updates=self.model.updates,
                givens={x:
                        self.data_handler.training_data[
                            set_index * self.mini_batch_size:
                            (set_index + 1) * self.mini_batch_size,
                            index * self.sub_ecg_length:
                            (index + 1) * self.sub_ecg_length],
                        y:
                        self.data_handler.training_labels[
                            set_index * self.mini_batch_size:
                            (set_index + 1) * self.mini_batch_size]
                        }
            )
        if self.verbose:
            print("Training model created.")

        if self.verbose:
            print("Creating Test model...")
        self.test_model = \
            theano.function(
                [set_index, index],
                (self.model.cost, self.model.error),
                givens={x:
                        self.data_handler.test_data[
                            set_index * self.mini_batch_size:
                            (set_index + 1) * self.mini_batch_size,
                            index * self.sub_ecg_length:
                            (index + 1) * self.sub_ecg_length],
                        y:
                        self.data_handler.test_labels[
                            set_index * self.mini_batch_size:
                            (set_index + 1) * self.mini_batch_size]
                        }
            )
        if self.verbose:
            print("Test model created.")

    def train_neural_network(self):

        if self.verbose:
            print("Training the model...")

        for epoch_index in range(1, self.number_of_epochs + 1):

            if self.verbose:
                print("    *** Epoch {}/{} ***".
                      format(epoch_index, self.number_of_epochs))

            """ Train model """
            train_errors = 0
            for training_set_index in range(
                    self.data_handler.number_of_training_sets //
                    self.mini_batch_size):
                for minibatch_index in range(self.number_of_sub_ecgs):
                    train_cost, train_error = \
                        self.train_model(training_set_index, minibatch_index)
                    train_errors += train_error

            if self.verbose:
                print("    Train Errors: {}/{}, {:.2f}%".
                      format(train_errors,
                             self.data_handler.number_of_training_sets *
                             self.number_of_sub_ecgs,
                             100 * train_errors /
                             (self.data_handler.number_of_training_sets *
                              self.number_of_sub_ecgs)))
                print("    Train Cost: {}".format(train_cost))

            """ Test model """
            test_errors = 0
            for test_set_index in range(
                    self.data_handler.number_of_test_sets //
                    self.mini_batch_size):
                for minibatch_index in range(self.number_of_sub_ecgs):
                    test_cost, test_error = \
                        self.test_model(test_set_index, minibatch_index)
                    test_errors += test_error

            if self.verbose:
                print("    Test Errors: {}/{}, {:.2f}%".
                      format(test_errors,
                             self.data_handler.number_of_test_sets *
                             self.number_of_sub_ecgs,
                             100 * test_errors /
                             (self.data_handler.number_of_test_sets *
                              self.number_of_sub_ecgs)))
                print("    Test Cost: {}".format(test_cost))

        if self.verbose:
            print("Model trained.")


if __name__ == "__main__":
    p = PAFClassifier(number_of_epochs=50)
    p.create_models()
    p.train_neural_network()
