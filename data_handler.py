
import numpy as np
import theano


class DataHandler():

    MAX_POSITIVE_TRAINING_SETS = 50
    MAX_NEGATIVE_TRAINING_SETS = 50
    MAX_TEST_SETS = 100
    TRAINING_DATA_DIR = "training-data/"
    TEST_DATA_DIR = "test-data/"
    DATA_TYPE = np.int16

    def __init__(self,
                 number_of_positive_sets=50,
                 number_of_negative_sets=50,
                 number_of_test_sets=100,
                 verbose=True):
        self.verbose = verbose

        self.number_of_positive_sets = number_of_positive_sets
        self.number_of_negative_sets = number_of_negative_sets
        self.number_of_training_sets = \
            self.number_of_negative_sets + self.number_of_positive_sets
        self.number_of_test_sets = number_of_test_sets

        self.training_data = \
            np.zeros((self.number_of_training_sets, 38400*6),
                     dtype=self.DATA_TYPE)
        self.training_labels = \
            np.zeros(self.number_of_training_sets, dtype=self.DATA_TYPE)

        self.test_data = \
            np.zeros((self.number_of_test_sets, 38400*6),
                     dtype=self.DATA_TYPE)
        self.test_labels = \
            np.zeros(self.number_of_test_sets, dtype=self.DATA_TYPE)

        self.training_shuffled = np.arange(self.number_of_training_sets)
        np.random.shuffle(self.training_shuffled)
        self.test_shuffled = np.arange(self.number_of_test_sets)
        np.random.shuffle(self.test_shuffled)

    def load_training_data(self):

        set_index = 0

        if self.verbose:
            print("Loading positive training data...")

        for data_set_index in range(1, self.number_of_positive_sets + 1):
            if data_set_index < 10:
                file_name = self.TRAINING_DATA_DIR + "p0" \
                            + str(data_set_index) + ".dat"
            else:
                file_name = self.TRAINING_DATA_DIR + "p" + \
                            str(data_set_index) + ".dat"
            data = np.fromfile(file_name, dtype=self.DATA_TYPE)
            ecg_data = data[::2] - data[1::2]
            self.training_data[set_index, :] = ecg_data
            self.training_labels[set_index] = 1
            set_index += 1

        if self.verbose:
            print("Loading negative training data...")

        for data_set_index in range(1, self.number_of_negative_sets + 1):
            if data_set_index < 10:
                file_name = self.TRAINING_DATA_DIR + "n0" + \
                            str(data_set_index) + ".dat"
            else:
                file_name = self.TRAINING_DATA_DIR + "n" + \
                            str(data_set_index) + ".dat"
            data = np.fromfile(file_name, dtype=self.DATA_TYPE)
            ecg_data = data[::2] - data[1::2]
            self.training_data[set_index, :] = ecg_data
            self.training_labels[set_index] = 0
            set_index += 1

        """ Shuffle Training data """
        self.training_data = self.training_data[self.training_shuffled]
        self.training_labels = self.training_labels[self.training_shuffled]

        """ Create Theano symbolic variables from training data """
        self.training_data = theano.shared(
            np.asarray(self.training_data,
                       dtype=np.float64),
            borrow=True)
        self.training_labels = theano.shared(
            np.asarray(self.training_labels,
                       dtype=np.int32),
            borrow=True)

        if self.verbose:
            print("Training data loaded.")

    def load_test_data(self):

        if self.verbose:
            print("Loading test data....")

        for data_set_index in range(1, self.number_of_test_sets + 1):
            if data_set_index < 10:
                file_name = self.TEST_DATA_DIR + "t0" \
                            + str(data_set_index) + ".dat"
            else:
                file_name = self.TEST_DATA_DIR + "t" + \
                            str(data_set_index) + ".dat"
            data = np.fromfile(file_name, dtype=self.DATA_TYPE)
            ecg_data = data[::2] - data[1::2]
            self.test_data[data_set_index - 1, :] = ecg_data

        file_name = self.TEST_DATA_DIR + "labels.txt"
        test_labels = np.loadtxt(file_name, dtype=self.DATA_TYPE)
        self.test_labels = test_labels[:self.number_of_test_sets, 1]

        """ Shuffle Test data """
        self.test_data = self.test_data[self.test_shuffled]
        self.test_labels = self.test_labels[self.test_shuffled]

        """ Create Theano symbolic variables from test data """
        self.test_data = theano.shared(
            np.asarray(self.test_data,
                       dtype=np.float64),
            borrow=True)
        self.test_labels = theano.shared(
            np.asarray(self.test_labels,
                       dtype=np.int32),
            borrow=True)

        if self.verbose:
            print("Test data loaded.")
