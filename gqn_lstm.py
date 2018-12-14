from collections import namedtuple
import tensorflow as tf
from utils import create_sub_scope


class GQNLSTMCell(tf.contrib.rnn.RNNCell):
    def __init__(self, input_shape, output_channels, kernel_size=5, use_bias=True, forget_bias=1., hidden_state_name='h', name='GQNCell'):
        super(GQNLSTMCell, self).__init__(name=name)
        if len(input_shape) != 3:
            raise ValueError('Invalid input_shape {}.'.format(input_shape))

        self._input_shape = input_shape
        self._output_channels = output_channels
        self._kernel_size = kernel_size
        self._use_bias = use_bias
        self._forget_bias = forget_bias
        self._hidden_state_name = hidden_state_name
        state_size = tf.TensorShape(self._input_shape[:-1] + [output_channels])
        self._output_size = state_size
        self._state_size = tf.contrib.rnn.LSTMStateTuple(state_size, state_size)

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._state_size

    def _conv(self, inputs):
        conv_outputs = []
        for k in inputs:
            conv_outputs.append(tf.layers.conv2d(inputs[k], filters=4*self._output_channels, kernel_size=self._kernel_size, strides=1, padding='SAME', use_bias=self._use_bias, activation=None, name='{}_LSTMConv'.format(k)))
        return tf.add_n(conv_outputs)

    def call(self, inputs, state, scope=None):
        cell_state, hidden_state = state
        inputs[self._hidden_state_name] = hidden_state
        with tf.name_scope('InputConv'):
            new_hidden = self._conv(inputs)
            gates = tf.split(value=new_hidden, num_or_size_splits=4, axis=-1)
        input_gate, new_input, forget_gate, output_gate = gates
        with tf.name_scope('Forget'):
            new_cell = tf.nn.sigmoid(forget_gate + self._forget_bias)*cell_state
        with tf.name_scope('Update'):
            new_cell += tf.nn.sigmoid(input_gate) * tf.nn.tanh(new_input)
        with tf.name_scope('Output'):
            output = tf.nn.tanh(new_cell)*tf.nn.sigmoid(output_gate)
        new_state = tf.contrib.rnn.LSTMStateTuple(new_cell, output)
        return output, new_state


_GeneratorCellInput = namedtuple('GeneratorCellInput', ['representation', 'query_pose', 'z'])
_GeneratorCellOutput = namedtuple('GeneratorCellOutput', ['canvas', 'lstm'])  # lstm = h
_GeneratorCellState = namedtuple('GeneratorCellState', ['canvas', 'lstm'])  # lstm = (c, h)


class GeneratorLSTMCell(tf.contrib.rnn.RNNCell):
    def __init__(self, input_shape, output_channels, canvas_channels, kernel_size=5, use_bias=True, forget_bias=1., name='GeneratorLSTMCell'):
        super(GeneratorLSTMCell, self).__init__(name=name)
        if len(input_shape) != 3:
            raise ValueError('Invalid input_shape {}.'.format(input_shape))

        self._gqn_cell = GQNLSTMCell(input_shape, output_channels, kernel_size, use_bias, forget_bias, hidden_state_name='h_g', name='{}_GQNCell'.format(name))
        canvas_size = tf.TensorShape([4*x for x in input_shape[:-1]] + [canvas_channels])
        self._canvas_channels = canvas_channels
        self._output_size = _GeneratorCellOutput(canvas_size, self._gqn_cell.output_size)
        self._state_size = _GeneratorCellState(canvas_size, self._gqn_cell.state_size)

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._state_size

    def call(self, inputs, state, scope=None):
        canvas, (cell_state, hidden_state) = state
        sub_state = tf.contrib.rnn.LSTMStateTuple(cell_state, hidden_state)
        sub_output, new_sub_state = self._gqn_cell(inputs._asdict(), sub_state, scope=create_sub_scope(scope, 'GQNCell'))
        new_canvas = canvas + tf.layers.conv2d_transpose(sub_output, filters=self._canvas_channels, kernel_size=4, strides=4, name='UpsampleGeneratorOutput')
        new_output = _GeneratorCellOutput(new_canvas, sub_output)
        new_state = _GeneratorCellState(new_canvas, new_sub_state)
        return new_output, new_state


_InferenceCellInput = namedtuple('InferenceCellInput', ['representation', 'query_pose', 'query_image', 'canvas', 'h_g'])
_InferenceCellOutput = namedtuple('InferenceCellOutput', ['lstm'])
_InferenceCellState = namedtuple('InferenceCellState', ['lstm'])


class InferenceLSTMCell(tf.contrib.rnn.RNNCell):
    def __init__(self, input_shape, output_channels, kernel_size=5, use_bias=True, forget_bias=1., name='InferenceLSTMCell'):
        super(InferenceLSTMCell, self).__init__(name=name)
        if len(input_shape) != 3:
            raise ValueError('Invalid input_shape {}.'.format(input_shape))
        self._gqn_cell = GQNLSTMCell(input_shape, output_channels, kernel_size, use_bias, forget_bias, hidden_state_name='h_e', name='{}_GQNCell'.format(name))
        self._output_size = _InferenceCellOutput(self._gqn_cell.output_size)
        self._state_size = _InferenceCellState(self._gqn_cell.state_size)

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._state_size

    def call(self, inputs, state, scope=None):
        cell_state, hidden_state = state.lstm
        input_dict = inputs._asdict()
        query_image = input_dict.pop('query_image')
        canvas = input_dict.pop('canvas')
        input_canvas_and_images = tf.layers.conv2d(tf.concat([query_image, canvas], axis=-1), filters=self.output_size.lstm[-1], kernel_size=4, strides=4, padding='VALID', use_bias=False, name='DownsampleInferenceInputCanvasAndImage')
        hidden_state += input_canvas_and_images
        state = tf.contrib.rnn.LSTMStateTuple(cell_state, hidden_state)
        output, new_state = self._gqn_cell(input_dict, state, scope=create_sub_scope(scope, 'GQNCell'))
        return _InferenceCellOutput(output), _InferenceCellState(new_state)
