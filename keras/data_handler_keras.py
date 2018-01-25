
import numpy as np
import scipy.signal as sgn
from matplotlib import pyplot as plt


class DataHandler:

    MAX_POSITIVE_TRAINING_SETS = 50
    MAX_NEGATIVE_TRAINING_SETS = 50
    MAX_TEST_SETS = 50
    TRAINING_DATA_DIR = "../training-data/"
    TEST_DATA_DIR = "../test-data/"
    DATA_TYPE = np.int16

    def __init__(self,
                 number_of_channels=1,
                 number_of_positive_sets=50,
                 number_of_negative_sets=50,
                 number_of_test_sets=100,
                 number_of_sub_ecgs=6,
                 length_of_sub_ecg=38400,
                 low_cutoff=1,
                 high_cutoff=40,
                 verbose=True):
        self.verbose = verbose
        self.number_of_channels = number_of_channels
        self.scaling_factor = 1

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

        self.training_labels = np.array([], dtype=self.DATA_TYPE)
        self.test_labels = np.array([], dtype=self.DATA_TYPE)

        self.sampling_frequency = 128
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        if self.low_cutoff:
            self.high_pass_filter = sgn.firwin2(
                1051,
                [0, self.low_cutoff - 0.1, self.low_cutoff, self.sampling_frequency/2],
                [0, 0, 1, 1],
                fs=self.sampling_frequency
            )
        if self.high_cutoff:
            self.low_pass_filter = sgn.firwin2(
                251,
                [0, self.high_cutoff, self.high_cutoff + 2, self.sampling_frequency/2],
                [1, 1, 0, 0],
                fs=self.sampling_frequency
            )

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
            self.training_labels = \
                np.append(self.training_labels,
                          np.ones(1,
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
            self.training_labels = \
                np.append(self.training_labels,
                          np.zeros(1,
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
        self.test_labels = test_labels[:self.number_of_test_sets, 1]

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
            / self.training_data_std * self.scaling_factor

       # plt.plot(self.training_data[30000:33000])
       # plt.show()

        self.training_data = self.training_data.reshape(
            self.number_of_training_sets,
            self.length_of_sub_ecg,
            6)

        self.training_labels = self.onehot(self.training_labels)

        self.training_data = np.asarray(self.training_data,
                                        dtype=np.float32)
        self.training_labels = np.asarray(self.training_labels,
                                          dtype=np.float32)

        """ Preprocess test data """
        self.test_data = self.filter_data(self.test_data.transpose())

        self.test_data = \
            (self.test_data - self.training_data_mean)\
            / self.training_data_std * self.scaling_factor

       # plt.plot(self.test_data[30000:33000])
       # plt.show()

        self.test_data = self.test_data.reshape(
            self.number_of_test_sets,
            self.length_of_sub_ecg,
            6)

        self.test_labels = self.onehot(self.test_labels)

        self.test_data = np.asarray(self.test_data,
                                    dtype=np.float32)
        self.test_labels = np.asarray(self.test_labels,
                                      dtype=np.float32)

        if self.verbose:
            print("Data preprocessed.")

    def filter_data(self, data):
        if self.low_cutoff:
            data = sgn.filtfilt(self.high_pass_filter, 1, data)
        if self.high_cutoff:
            data = sgn.filtfilt(self.low_pass_filter, 1, data)
        return data.transpose()

    def onehot(self, input_data):
        n = input_data.shape[0]
        onehot = np.zeros((n, 2))
        for i in range(n):
            label = input_data[i]
            onehot[i, label] = 1
        return onehot

    def write_information(self, paf):
        with open(paf.filename, "w") as f:
            f.write("# Information\n")
            f.write("# Number of channels,{}\n".format(self.number_of_channels))
            f.write("# Mini-batch size,{}\n".format(paf.mini_batch_size))
            f.write("# Number of filters,{}\n".format(paf.number_of_filters))
            f.write("# Filter length,{}\n".format(paf.filter_length))
            f.write("# Regularization Coefficient,{}\n".format(paf.regularization_coefficient))
            f.write("# Pool size,{}\n".format(paf.pool_size))
            f.write("# Dropout rate,{}\n".format(paf.dropout_rate))
            f.write("# FC layer neurons,{}\n".format(paf.fully_connected_layer_neurons))
            f.write("# Learning rate,{}\n".format(paf.learning_rate))
            f.write("# Pooling type,{}\n".format(paf.pooling_type))
            f.write("# Optimizer,{}\n".format(paf.optimizer))
            f.write("# Conv activation,{}\n".format(paf.convolution_activation))
            f.write("# FC activation,{}\n".format(paf.fully_connected_activation))
            f.write("# Output activation,{}\n\n".format(paf.output_activation))
            f.write("epoch,acc,loss,val_acc,val_loss\n")
