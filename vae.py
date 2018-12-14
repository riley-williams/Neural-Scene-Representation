import tensorflow as tf
from config import CONFIG as C
from utils import broadcast_pose


def vae_simple_encoder(net, scope='VAESimpleEncoder'):
    with tf.variable_scope(scope):
        net = tf.layers.conv2d(net, filters=512, kernel_size=3, padding='SAME', activation=tf.nn.relu)
        net = tf.layers.conv2d(net, filters=64, kernel_size=3, strides=2, padding='SAME', activation=tf.nn.relu)
        net = tf.layers.conv2d(net, filters=128, kernel_size=3, strides=2, padding='SAME', activation=tf.nn.relu)
        net = tf.layers.conv2d(net, filters=512, kernel_size=5, strides=2, padding='SAME', activation=tf.nn.relu)
        net = tf.layers.conv2d(net, filters=16, kernel_size=5, strides=2, padding='SAME', activation=tf.nn.relu)
        net = tf.space_to_depth(net, block_size=4)
        return net, {}


def vae_simple_decoder(net, scope='VAESimpleDecoder'):
    def _upsample_conv2d(net, factor, filters, **kwargs):
        net = tf.layers.conv2d(net, filters=filters*(factor**2), **kwargs)
        net = tf.depth_to_space(net, block_size=factor)
        return net

    with tf.variable_scope(scope):
        net = _upsample_conv2d(net, factor=16, filters=128, kernel_size=3, padding='SAME', activation=tf.nn.relu)
        net = _upsample_conv2d(net, factor=2, filters=512, kernel_size=3, padding='SAME', activation=tf.nn.relu)
        net = _upsample_conv2d(net, factor=2, filters=512, kernel_size=3, padding='SAME', activation=tf.nn.relu)
        net = tf.layers.conv2d(net, filters=3, kernel_size=3, padding='SAME')
        return net, {}


def vae_tower_decoder(net, query_pose, output_channels=C.LSTM_CANVAS_CHANNELS, scope='VAETowerDecoder'):
    with tf.variable_scope(scope):
        net = tf.layers.conv2d(net, filters=256, kernel_size=1, strides=1, padding='SAME', activation=tf.nn.relu)
        net = tf.layers.conv2d(net, filters=128, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu)

        height, width = tf.shape(net)[1], tf.shape(net)[2]
        query_pose = broadcast_pose(query_pose, height, width)
        net = tf.concat([net, query_pose], axis=-1)
        skip1 = tf.layers.conv2d(net, filters=256, kernel_size=1, strides=1, padding='SAME', activation=None)
        net = tf.layers.conv2d(net, filters=256, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu)
        net += skip1
        net = tf.layers.conv2d(net, filters=256, kernel_size=2, strides=2, padding='VALID', activation=tf.nn.relu)
        net = tf.image.resize_bilinear(net, size=(2*height, 2*width))
        net = tf.layers.conv2d(net, filters=256, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu)
        skip2 = tf.layers.conv2d(net, filters=128, kernel_size=1, strides=1, padding='SAME', activation=None)
        net = tf.layers.conv2d(net, filters=128, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu)
        net += skip2
        net = tf.layers.conv2d(net, filters=256, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu)
        net = tf.image.resize_bilinear(net, size=(2*height, 2*width))
        net = tf.layers.conv2d(net, filters=128, kernel_size=1, strides=1, padding='SAME', activation=tf.nn.relu)
        net = tf.layers.conv2d(net, filters=256, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu)
        net = tf.layers.conv2d(net, filters=output_channels, kernel_size=1, strides=1, padding='SAME', activation=None)
        return net, {}
