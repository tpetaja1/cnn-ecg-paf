
import numpy as np
import theano
import theano.tensor as T

from theano.tensor.nnet.bn import batch_normalization_train
from theano.tensor.nnet.abstract_conv import causal_conv1d
from theano.tensor.signal import pool


class CNN:

    def __init__(self,
                 number_of_channels=2,
                 learning_rate=0.01,
                 data_length_in_mini_batch=38400,
                 batch_size=None,
                 number_of_filters=1,
                 filter_length=15,
                 activation=T.tanh,
                 classification_method=T.nnet.softmax,
                 pool_mode="max",
                 regularization_coefficient=0.001,
                 fully_connected_layer_neurons=16,
                 pool_size=2,
                 momentum=0.9,
                 perform_normalization="all",
                 update_type="adam"):

        """ Initialize hyperparameters """
        self.number_of_channels = number_of_channels
        self.learning_rate = learning_rate
        self.data_length_in_mini_batch = data_length_in_mini_batch
        self.batch_size = batch_size
        self.number_of_filters = number_of_filters
        self.filter_length = filter_length
        self.activation = activation
        self.classification_method = classification_method
        self.pool_mode = pool_mode
        self.regularization_coefficient = regularization_coefficient
        self.momentum = momentum
        self.perform_normalization = perform_normalization
        self.update_type = update_type

        """ Initialize shapes """
        self.convolution_input_shape = \
            (self.batch_size,
             self.number_of_channels,
             self.data_length_in_mini_batch)

        self.filter_shape = \
            (self.number_of_filters,
             self.number_of_channels,
             self.filter_length)

        self.pool_shape = (1, pool_size)

        self.fully_connected_layer_neurons = fully_connected_layer_neurons

        self.fully_connected_layer_shape = \
            (self.number_of_filters * (
                self.data_length_in_mini_batch // pool_size),
             fully_connected_layer_neurons)

        self.output_layer_shape = (fully_connected_layer_neurons, 2)

        """ Initialize parameters """
        self.parameters = []
        self.updates = []

    def create_computational_graph(self, x, y):
        # x = input to neural network
        # y = labels of input data

        """ Computational Graph """
        # *** Input Layer ***
        # Output Shape:
        # batch size  x  no. of channels  x  length of data
        input_data = x

        # *** Convolutional Layer ***
        # Output Shape:
        # batch size  x  no. of filters  x  length of data
        convolution_layer_output = self.convolution_layer(input_data)

        # *** Subsampling Layer ***
        # Output Shape:
        # batch size  x  no. of filters  x  (length of data / pool size)
        subsampling_layer_output = \
            self.subsampling_layer(convolution_layer_output)

        # *** Fully Connected Layer ***
        # Input Shape:
        # batch size  x  (no. of filters * length of data / pool size)
        fully_connected_layer_input = subsampling_layer_output.flatten(2)

        # *** Fully Connected Layer ***
        # Output Shape:
        # batch size  x  no. of fully connected layer neurons
        self.fully_connected_layer_output = \
            self.fully_connected_layer(fully_connected_layer_input)

        # *** Output Layer ***
        # Output Shape:
        # batch size  x  2
        self.probabilities = \
            self.output_layer(self.fully_connected_layer_output)

        # Predictions:
        self.predictions = T.argmax(self.probabilities, axis=1)

        """ Cost functions """
        self.regularizer = \
            self.parameters[0].norm(L=2) +\
            self.parameters[2].norm(L=2) +\
            self.parameters[4].norm(L=2)

        self.negative_log_likelihood = \
            -T.mean(T.log(self.probabilities)[T.arange(y.shape[0]), y])

        self.penalty = \
            self.regularization_coefficient * \
            self.regularizer / (2 * y.shape[0])

        self.cost = self.negative_log_likelihood + self.penalty

        """ Metrics """
        self.error = T.sum(T.neq(self.predictions, y))

        self.sensitivity = self.calculate_sensitivity(self.predictions, y)

        self.specificity = self.calculate_specificity(self.predictions, y)

        """ Backpropagation """
        self.parameter_updates()

    def input_layer(self, input_data):

        if self.perform_normalization == "all"\
           or self.perform_normalization == "only input":
            gamma = theano.shared(1.)
            bias = theano.shared(0.)
            running_mean = theano.shared(0.)
            running_var = theano.shared(0.)

            normalized_input_data, _, _,\
                new_running_mean, new_running_var = \
                batch_normalization_train(input_data,
                                          gamma,
                                          bias,
                                          axes=(0, 1),
                                          running_mean=running_mean,
                                          running_var=running_var)

            output = \
                normalized_input_data.reshape(self.convolution_input_shape)

            self.updates.append((running_mean, new_running_mean))
            self.updates.append((running_var, new_running_var))

        else:
            output = input_data.reshape(self.convolution_input_shape)

        return output

    def convolution_layer(self, input_data):

        """ Weight and bias for the layer """
        filter_weights = \
            theano.shared(
                np.asarray(
                    np.random.normal(0,
                                     1,
                                     size=self.filter_shape),
                    dtype=theano.config.floatX),
                name="Filter weights",
                borrow=True)

        bias_convolution = \
            theano.shared(np.zeros((self.number_of_filters,),
                                   dtype=theano.config.floatX),
                          borrow=True)

        """ Convolution """
        convolution = causal_conv1d(input=input_data,
                                    filters=filter_weights,
                                    filter_shape=self.filter_shape,
                                    input_shape=self.convolution_input_shape)

        convolution_output = \
            convolution + bias_convolution.dimshuffle("x", 0, "x")

        if self.perform_normalization == "all":
            gamma = theano.shared(1.)
            bias = theano.shared(0.)
            running_mean = theano.shared(0.)
            running_var = theano.shared(0.)

            normalized_output, _, _,\
                new_running_mean, new_running_var = \
                batch_normalization_train(convolution_output,
                                          gamma,
                                          bias,
                                          axes=(0, 1, 2),
                                          running_mean=running_mean,
                                          running_var=running_var)

            self.updates.append((running_mean, new_running_mean))
            self.updates.append((running_var, new_running_var))

            activation_output = \
                self.activation(normalized_output)

        else:
            activation_output = \
                self.activation(
                    convolution_output)

        """ Add parameters to be updated """
        self.parameters.append(filter_weights)
        self.parameters.append(bias_convolution)

        return activation_output

    def subsampling_layer(self, input_data):

        pool_out = pool.pool_2d(input=input_data,
                                ws=self.pool_shape,
                                ignore_border=False,
                                mode=self.pool_mode)

        return pool_out

    def fully_connected_layer(self, input_data):

        """ Weight and bias for the layer """
        W_fully_connected = \
            theano.shared(
                np.asarray(
                    np.random.normal(0,
                                     1,
                                     size=self.fully_connected_layer_shape),
                    dtype=theano.config.floatX),
                name="W fully connected",
                borrow=True)

        bias_fully_connected = \
            theano.shared(np.zeros((self.fully_connected_layer_shape[1],),
                                   dtype=theano.config.floatX),
                          name="bias fully connected",
                          borrow=True)

        dot_output = \
            T.dot(input_data, W_fully_connected) + bias_fully_connected

        if self.perform_normalization == "all":
            gamma = theano.shared(1.)
            bias = theano.shared(0.)
            running_mean = theano.shared(0.)
            running_var = theano.shared(0.)

            normalized_output, _, _,\
                new_running_mean, new_running_var = \
                batch_normalization_train(dot_output,
                                          gamma,
                                          bias,
                                          axes=(0, 1),
                                          running_mean=running_mean,
                                          running_var=running_var)

            self.updates.append((running_mean, new_running_mean))
            self.updates.append((running_var, new_running_var))

            output_fully_connected = self.activation(
                normalized_output)

        else:
            output_fully_connected = self.activation(dot_output)

        """ Add parameters to be updated """
        self.parameters.append(W_fully_connected)
        self.parameters.append(bias_fully_connected)

        return output_fully_connected

    def output_layer(self, input_data):

        """ Weight and bias for the layer """
        W_output_layer = \
            theano.shared(
                np.asarray(
                    np.random.normal(0,
                                     1,
                                     size=self.output_layer_shape),
                    dtype=theano.config.floatX),
                name="W output",
                borrow=True)

        bias_output_layer = \
            theano.shared(np.zeros((self.output_layer_shape[1],),
                                   dtype=theano.config.floatX),
                          name="bias output",
                          borrow=True)

        dot_output = T.dot(input_data, W_output_layer) + bias_output_layer

        if self.perform_normalization:
            gamma = theano.shared(1.)
            bias = theano.shared(0.)
            running_mean = theano.shared(0.)
            running_var = theano.shared(0.)

            normalized_output, _, _,\
                new_running_mean, new_running_var = \
                batch_normalization_train(dot_output,
                                          gamma,
                                          bias,
                                          axes=(0, 1),
                                          running_mean=running_mean,
                                          running_var=running_var)

            self.updates.append((running_mean, new_running_mean))
            self.updates.append((running_var, new_running_var))

            output = self.classification_method(normalized_output)

        else:
            output = self.classification_method(dot_output)

        """ Add parameters to be updated """
        self.parameters.append(W_output_layer)
        self.parameters.append(bias_output_layer)

        return output

    def parameter_updates(self):

        self.gradients = T.grad(self.cost, self.parameters)

        """ Momentum gradient update """
        if self.update_type == "momentum":

            for parameter, gradient in zip(self.parameters, self.gradients):

                velocity = theano.shared(parameter.get_value(borrow=True) * 0.)

                new_velocity = \
                    self.momentum * velocity - self.learning_rate * gradient
                self.updates.append(
                    (velocity, new_velocity))
                self.updates.append(
                    (parameter,
                     parameter + new_velocity))

        elif self.update_type == "adam":

            eps = 1e-4  # small constant used for numerical stabilization.
            beta1 = 0.9
            beta2 = 0.999

            for parameter, gradient in zip(self.parameters, self.gradients):
                # Shared variables
                t = theano.shared(1.)
                s = theano.shared(parameter.get_value(borrow=True) * 0.)
                r = theano.shared(parameter.get_value(borrow=True) * 0.)

                # Correct bias
                s_hat = s / (1. - beta1 ** t)
                r_hat = r / (1. - beta2 ** t)

                # Update moment estimates
                self.updates.append(
                    (s, beta1 * s + (1. - beta1) * gradient))
                self.updates.append(
                    (r, beta2 * r + (1. - beta2) * T.sqr(gradient)))

                # Update parameter
                self.updates.append(
                    (parameter,
                     T.cast(parameter -
                            self.learning_rate * s_hat / T.sqrt(r_hat + eps),
                            theano.config.floatX)))

                # Update time step
            self.updates.append((t, t + 1))

    def calculate_sensitivity(self, x, y):
        true_positives = T.sum(T.and_(T.eq(x, 1), T.eq(y, 1)))
        sensitivity = true_positives / T.sum(T.eq(y, 1))
        return sensitivity

    def calculate_specificity(self, x, y):
        true_negatives = T.sum(T.and_(T.eq(x, 0), T.eq(y, 0)))
        specificity = true_negatives / T.sum(T.eq(y, 0))
        return specificity
