
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool


class CNN():

    def __init__(self, learning_rate=0.01,
                 data_length_in_mini_batch=38400,
                 number_of_filters=1,
                 filter_length=15,
                 activation=T.tanh,
                 classification_method=T.nnet.softmax,
                 pool_mode="max",
                 regularization_coefficient=0.001):

        """ Initialize hyperparameters """
        self.data_length_in_mini_batch = data_length_in_mini_batch
        self.number_of_filters = number_of_filters
        self.filter_length = filter_length
        self.activation_in_conv_layer = activation
        self.learning_rate = learning_rate
        self.pool_mode = "max"
        self.classification_method = classification_method
        self.regularization_coefficient = regularization_coefficient

        """ Initialize shapes """
        self.convolution_input_shape = \
            (1, 1, self.data_length_in_mini_batch, 1)
        self.filter_shape = (self.number_of_filters, 1, self.filter_length, 1)
        self.pool_shape = (2, 1)
        self.fully_connected_layer_shape = \
            (int(self.number_of_filters*(
                self.data_length_in_mini_batch + self.filter_length - 1) /
             self.pool_shape[0]),
             2)

        """ Initialize parameters """
        self.filter_weights = None
        self.bias_convolution = None
        self.W_fully_connected = None
        self.bias_fully_connected = None

    def create_computational_graph(self):

        # Symbolic Theano variables
        self.x = T.dvector('x')
        self.y = T.iscalar('y')

        # Shape: 1  x  1  x  length_of_data  x  1
        input_data = self.input_layer(self.x)

        # Shape: 1  x  number_of_filters  x
        #        (length_of_data + length_of_filter - 1)  x  1
        convolution_layer_output = self.convolution_layer(input_data)

        # Shape: 1  x  number_of_filters  x
        #        (convolution_output / subsampling_coefficient)  x  1
        subsampling_layer_output = \
            self.subsampling_layer(convolution_layer_output)

        # Shape: 1  x  (number_of_filters * subsampling_output)
        fully_connected_layer_input = subsampling_layer_output.flatten(2)

        # Shape: 1  x  2
        fully_connected_layer_output = \
            self.fully_connected_layer(fully_connected_layer_input)

        # Shape: 1  x  2
        probabilities = self.output_layer(fully_connected_layer_output)

        prediction = T.argmax(probabilities, axis=1)

        regularizer = \
            T.sqr(self.filter_weights.norm(L=2)) +\
            T.sqr(self.W_fully_connected.norm(L=2))

        self.cost = \
            -T.mean(T.log(probabilities)[0, self.y]) +\
            self.regularization_coefficient * regularizer

        self.error = T.sum(T.neq(prediction, self.y))

        self.parameters = [self.filter_weights,
                           self.bias_convolution,
                           self.W_fully_connected,
                           self.bias_fully_connected]

        self.gradients = T.grad(self.cost, self.parameters)

        self.updates = \
            [(parameter, parameter - self.learning_rate * gradient)
             for parameter, gradient in zip(self.parameters, self.gradients)]

    def input_layer(self, input_data):

        return input_data.reshape(self.convolution_input_shape)

    def convolution_layer(self, input_data):

        self.filter_weights = \
            theano.shared(np.asarray(
                np.random.normal(0,
                                 0.1,
                                 size=self.filter_shape)),
                          name="Filter weights",
                          borrow=True)

        self.bias_convolution = \
            theano.shared(np.zeros((self.number_of_filters,),
                                   dtype=theano.config.floatX),
                          borrow=True)

        output = conv2d(input=input_data,
                        filters=self.filter_weights,
                        filter_shape=self.filter_shape,
                        input_shape=self.convolution_input_shape,
                        border_mode="full")

        activation_output = \
            self.activation_in_conv_layer(
                output + self.bias_convolution.dimshuffle("x", 0, "x", "x"))

        return activation_output

    def subsampling_layer(self, input_data):

        pool_out = pool.pool_2d(input=input_data,
                                ws=self.pool_shape,
                                ignore_border=True,
                                mode=self.pool_mode)

        return pool_out

    def fully_connected_layer(self, input_data):

        self.W_fully_connected = \
            theano.shared(np.asarray(
                np.random.normal(0,
                                 0.1,
                                 size=self.fully_connected_layer_shape)),
                          name="W fully connected",
                          borrow=True)

        self.bias_fully_connected = \
            theano.shared(np.zeros((self.fully_connected_layer_shape[1],),
                                   dtype=theano.config.floatX),
                          name="bias fully connected",
                          borrow=True)

        output_fully_connected = \
            T.dot(input_data, self.W_fully_connected) + \
            self.bias_fully_connected

        return output_fully_connected

    def output_layer(self, input_data):
        output = self.classification_method(input_data)
        return output
