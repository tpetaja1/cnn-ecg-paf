
import numpy as np
import theano
import scipy.signal as sgn


class DataHandler:

    MAX_POSITIVE_TRAINING_SETS = 50
    MAX_NEGATIVE_TRAINING_SETS = 50
    MAX_TEST_SETS = 50
    TRAINING_DATA_DIR = "../training-data/"
    TEST_DATA_DIR = "../test-data/"
    DATA_TYPE = np.int16

    def __init__(self,
                 number_of_channels=2,
                 number_of_positive_sets=50,
                 number_of_negative_sets=50,
                 number_of_test_sets=100,
                 number_of_sub_ecgs=6,
                 length_of_sub_ecg=38400,
                 verbose=True):
        self.verbose = verbose
        self.number_of_channels = number_of_channels

        self.number_of_sub_ecgs = number_of_sub_ecgs
        self.length_of_sub_ecg = length_of_sub_ecg

        self.number_of_positive_sets = number_of_positive_sets
        self.number_of_negative_sets = number_of_negative_sets
        self.number_of_training_sets = \
            self.number_of_negative_sets + self.number_of_positive_sets
        self.number_of_test_sets = number_of_test_sets

        self.total_number_of_training_data = \
            self.number_of_sub_ecgs * self.number_of_training_sets
        self.total_number_of_test_data = \
            self.number_of_sub_ecgs * self.number_of_test_sets

        if self.number_of_channels == 2:
            self.training_data = \
                np.array([[], []], dtype=self.DATA_TYPE).transpose()
            self.test_data = \
                np.array([[], []], dtype=self.DATA_TYPE).transpose()
        elif self.number_of_channels == 1:
            self.training_data = \
                np.array([[]], dtype=self.DATA_TYPE).transpose()
            self.test_data = \
                np.array([[]], dtype=self.DATA_TYPE).transpose()

        self.training_labels_raw = np.array([], dtype=self.DATA_TYPE)
        self.test_labels_raw = np.array([], dtype=self.DATA_TYPE)

        self.training_shuffled = \
            np.arange(self.total_number_of_training_data)
        np.random.shuffle(self.training_shuffled)

        self.test_shuffled = \
            np.arange(self.total_number_of_test_data)
        np.random.shuffle(self.test_shuffled)

        self.sampling_frequency = 128
        self.high_pass_filter = sgn.firwin2(
            251,
            [0, 0.5, 2, self.sampling_frequency/2],
            [0, 0, 1, 1],
            fs=self.sampling_frequency)

    def load_training_data(self):

        if self.verbose:
            print("Loading positive training data...")

        for data_set_index in range(1, self.number_of_positive_sets + 1):
            if data_set_index < 10:
                file_name = self.TRAINING_DATA_DIR + "p0" \
                            + str(data_set_index) + ".dat"
            else:
                file_name = self.TRAINING_DATA_DIR + "p" + \
                            str(data_set_index) + ".dat"
            data = np.fromfile(file_name, dtype=self.DATA_TYPE)/200

            if self.number_of_channels == 2:
                ecg_2_channel = data.reshape(data.shape[0] // 2, 2)
                self.training_data = \
                    np.append(self.training_data, ecg_2_channel, axis=0)
            elif self.number_of_channels == 1:
                ecg0 = data[::2]
                ecg0 = ecg0.reshape(ecg0.shape[0], 1)
                self.training_data = \
                    np.append(self.training_data, ecg0, axis=0)
            self.training_labels_raw = \
                np.append(self.training_labels_raw,
                          np.ones(self.number_of_sub_ecgs,
                                  dtype=self.DATA_TYPE))

        if self.verbose:
            print("Loading negative training data...")

        for data_set_index in range(1, self.number_of_negative_sets + 1):
            if data_set_index < 10:
                file_name = self.TRAINING_DATA_DIR + "n0" + \
                            str(data_set_index) + ".dat"
            else:
                file_name = self.TRAINING_DATA_DIR + "n" + \
                            str(data_set_index) + ".dat"
            data = np.fromfile(file_name, dtype=self.DATA_TYPE)/200

            if self.number_of_channels == 2:
                ecg_2_channel = data.reshape(data.shape[0] // 2, 2)
                self.training_data = \
                    np.append(self.training_data, ecg_2_channel, axis=0)
            elif self.number_of_channels == 1:
                ecg0 = data[::2]
                ecg0 = ecg0.reshape(ecg0.shape[0], 1)
                self.training_data = \
                    np.append(self.training_data, ecg0, axis=0)
            self.training_labels_raw = \
                np.append(self.training_labels_raw,
                          np.zeros(self.number_of_sub_ecgs,
                                   dtype=self.DATA_TYPE))

        if self.verbose:
            print("Training data loaded.")

    def load_test_data(self):

        if self.verbose:
            print("Loading test data....")

        for data_set_index in range(1, 2 * self.number_of_test_sets, 2):
            if data_set_index < 10:
                file_name = self.TEST_DATA_DIR + "t0" \
                            + str(data_set_index) + ".dat"
            else:
                file_name = self.TEST_DATA_DIR + "t" + \
                            str(data_set_index) + ".dat"
            data = np.fromfile(file_name, dtype=self.DATA_TYPE)/200

            if self.number_of_channels == 2:
                ecg_2_channel = data.reshape(data.shape[0] // 2, 2)
                self.test_data = \
                    np.append(self.test_data, ecg_2_channel, axis=0)
            elif self.number_of_channels == 1:
                ecg0 = data[::2]
                ecg0 = ecg0.reshape(ecg0.shape[0], 1)
                self.test_data = \
                    np.append(self.test_data, ecg0, axis=0)

        file_name = self.TEST_DATA_DIR + "labels0.txt"
        test_labels = np.loadtxt(file_name, dtype=self.DATA_TYPE)
        self.test_labels_raw = test_labels[:self.number_of_test_sets, 1]

        if self.verbose:
            print("Test data loaded.")

    def preprocess_data(self):

        if self.verbose:
            print("Starting to preprocess data...")

        """ Preprocess training data """
        self.training_data = self.filter_data(self.training_data.transpose())

        self.training_data_mean = self.training_data.mean(axis=0)
        self.training_data_std = self.training_data.std(axis=0)

        self.training_data = \
            (self.training_data - self.training_data_mean)\
            / self.training_data_std

        self.training_data = self.training_data.reshape(
            self.total_number_of_training_data,
            self.length_of_sub_ecg,
            self.number_of_channels)

        """ Transpose channels into rows (previously on columns) """
        self.training_data = self.training_data.transpose((0, 2, 1))

        """ Shuffle Training data """
        self.training_data = self.training_data[self.training_shuffled]
        self.training_labels_raw = \
            self.training_labels_raw[self.training_shuffled]

        """ Create Theano symbolic variables from training data """
        self.training_data = theano.shared(
            np.asarray(self.training_data,
                       dtype=theano.config.floatX),
            borrow=True)
        self.training_labels = theano.shared(
            np.asarray(self.training_labels_raw,
                       dtype=np.int16),
            borrow=True)

        """ Preprocess test data """
        self.test_data = self.filter_data(self.test_data.transpose())

        self.test_data = \
            (self.test_data - self.training_data_mean)\
            / self.training_data_std

        self.test_data = self.test_data.reshape(
            self.total_number_of_test_data,
            self.length_of_sub_ecg,
            self.number_of_channels)
        self.test_labels_raw = \
            np.repeat(self.test_labels_raw, self.number_of_sub_ecgs)

        """ Transpose channels into rows (previously on columns) """
        self.test_data = self.test_data.transpose((0, 2, 1))

        """ Shuffle Test data """
        self.test_data = self.test_data[self.test_shuffled]
        self.test_labels_raw = self.test_labels_raw[self.test_shuffled]

        """ Create Theano symbolic variables from test data """
        self.test_data = theano.shared(
            np.asarray(self.test_data,
                       dtype=theano.config.floatX),
            borrow=True)
        self.test_labels = theano.shared(
            np.asarray(self.test_labels_raw,
                       dtype=np.int16),
            borrow=True)

        if self.verbose:
            print("Data preprocessed.")

    def filter_data(self, data):
        return sgn.filtfilt(self.high_pass_filter, 1, data).transpose()
