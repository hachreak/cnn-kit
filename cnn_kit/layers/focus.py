
"""Focus Layer."""

import numpy as np

from keras import backend as K, activations, initializers, regularizers, \
    constraints
from keras.engine.topology import Layer
from keras.utils import conv_utils


def _gaussian(kernel_width, sigma):
    """Create a gaussian tensor."""
    k = kernel_width // 2
    probs = [np.exp(-z*z/(2*sigma*sigma))/np.sqrt(2*np.pi*sigma*sigma)
             for z in range(-k, k+1)]
    # FIXME kernel filters more than 1!
    return np.outer(probs, probs)


def _build_matrix(gaussian, x_shape, input_shape):
    _, x_width, x_height, x_depth = x_shape
    k_width, k_height = gaussian.shape

    head_shape = x_width * x_height

    g = np.repeat(gaussian, x_depth * input_shape[2])
    g = np.repeat(g, head_shape)

    g = np.reshape(
        g, (head_shape, k_width * k_height * input_shape[2], x_depth)
    )

    return g


def _expand_weights(weights, gaussian_shape, x_shape):
    k_width, k_height = gaussian_shape
    x_width, x_height, x_depth = x_shape

    return K.repeat_elements(weights, k_width * k_height * x_depth, 1)


def shape(input_shape):
    return reduce(lambda x, y: x*y, input_shape)


class Focus(Layer):
    """Focalize neural network attention."""

    def __init__(self, kernel_width=3, sigma=1,
                 filters=1, strides=(1, 1), padding='valid',
                 data_format=None,
                 kernel_initializer='uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 use_bias=True,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 activity_regularizer=None,
                 activation=None,
                 **kwargs):
        #  self._output_dim = output_dim
        self._kernel_width = kernel_width
        self.kernel_size = (kernel_width, kernel_width)
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        if self.padding != 'valid':
            raise ValueError('Invalid border mode for LocallyConnected2D '
                             '(only "valid" is supported): ' + padding)
        self.activation = activations.get(activation)
        self._gaussian = _gaussian(kernel_width, sigma)
        self.filters = filters
        self.data_format = K.normalize_data_format(data_format)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        super(Focus, self).__init__(**kwargs)

    def _get_input_shape(self, input_shape):
        """Set input_shape (rows, cols, filters)."""
        if self.data_format == 'channels_last':
            input_row, input_col = input_shape[1:-1]
            input_filter = input_shape[3]
        else:
            input_row, input_col = input_shape[2:]
            input_filter = input_shape[1]

        if input_row is None or input_col is None:
            raise ValueError('The spatial dimensions of the inputs to '
                             ' a LocallyConnected2D layer '
                             'should be fully-defined, but layer received '
                             'the inputs shape ' + str(input_shape))
        return input_row, input_col, input_filter

    def build(self, input_shape):
        """Build layer."""
        self._input_shape = self._get_input_shape(input_shape)
        self._output_shape = self._get_output_shape(input_shape)

        self._weights_shape = shape(self._output_shape[1:3]), 1, self.filters

        self._weights = self.add_weight(
            name='kernel',
            shape=self._weights_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True
        )

        if self.use_bias:
            self._bias_shape = self._output_shape[1], self._output_shape[2], \
                self.filters
            self.bias = self.add_weight(
                    shape=self._bias_shape,
                    initializer=self.bias_initializer,
                    name='bias',
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint)
        else:
            self.bias = None

        super(Focus, self).build(input_shape)

    def call(self, x):
        """To focus eyes."""
        ga = _build_matrix(
            self._gaussian, self._output_shape, self._input_shape
        )
        we = _expand_weights(
            self._weights, self._gaussian.shape, self._input_shape
        )

        kernel = K.tf.multiply(we, ga)

        output = K.local_conv2d(
            x,
            kernel=kernel,
            kernel_size=self.kernel_size,
            strides=self.strides,
            output_shape=self._output_shape[1:3],
            data_format=self.data_format
        )

        if self.use_bias:
            output = K.bias_add(
                output, self.bias, data_format=self.data_format
            )

        return self.activation(output)

    def _get_output_shape(self, input_shape):
        input_row, input_col, _ = self._get_input_shape(input_shape)
        output_row = conv_utils.conv_output_length(
            input_row, self.kernel_size[0], self.padding, self.strides[0]
        )
        output_col = conv_utils.conv_output_length(
            input_col, self.kernel_size[1], self.padding, self.strides[1]
        )
        return (input_shape[0], output_row, output_col, self.filters)

    def compute_output_shape(self, input_shape):
        b, r, c, f = self._get_output_shape(input_shape)

        if self.data_format == 'channels_first':
            return (b, f, r, c)
        elif self.data_format == 'channels_last':
            return (b, r, c, f)

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
        }
        base_config = super(Focus, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
