import tensorflow as tf
from utils import broadcast_pose


def tower_encoder(frames, poses, scope='TowerEncoder'):
    with tf.variable_scope(scope):
        net = tf.layers.conv2d(frames, filters=256, kernel_size=2, strides=2, padding='VALID', activation=tf.nn.relu)
        skip1 = tf.layers.conv2d(net, filters=128, kernel_size=1, strides=1, padding='SAME', activation=None)
        net = tf.layers.conv2d(net, filters=128, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu)
        net += skip1
        net = tf.layers.conv2d(net, filters=256, kernel_size=2, strides=2, padding='VALID', activation=tf.nn.relu)
        height, width = tf.shape(net)[1], tf.shape(net)[2]
        poses = broadcast_pose(poses, height, width)
        net = tf.concat([net, poses], axis=3)
        skip2 = tf.layers.conv2d(net, filters=128, kernel_size=1, strides=1, padding='SAME', activation=None)
        net = tf.layers.conv2d(net, filters=128, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu)
        net += skip2
        net = tf.layers.conv2d(net, filters=256, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu)
        net = tf.layers.conv2d(net, filters=256, kernel_size=1, strides=1, padding='SAME', activation=tf.nn.relu)
        return net, {}


def pool_encoder(frames, poses, scope='PoolEncoder'):
    net, endpoints = tower_encoder(frames, poses, scope)
    with tf.variable_scope(scope):
        net = tf.reduce_mean(net, axis=[1, 2], keepdims=True)
    return net, endpoints


if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only run on GPU 1
    import numpy as np
    from config import CONFIG as C
    _BATCH_SIZE = 3
    _CONTEXT_SIZE = 5
    _DIM_POSE = C.POSE_CHANNELS
    _DIM_H_IMG = C.IMG_HEIGHT
    _DIM_W_IMG = C.IMG_WIDTH
    _DIM_C_IMG = C.IMG_CHANNELS
    _DIM_C_ENC = C.ENC_CHANNELS

    context_poses = tf.placeholder(tf.float32, [_BATCH_SIZE, _CONTEXT_SIZE, _DIM_POSE])
    context_frames = tf.placeholder(tf.float32, [_BATCH_SIZE, _CONTEXT_SIZE, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG])
    context_poses_packed = tf.reshape(context_poses, shape=[-1, _DIM_POSE])
    context_frames_packed = tf.reshape(context_frames, shape=[-1, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG])

    r_encoder_batch, ep_encoding = pool_encoder(context_frames_packed, context_poses_packed)
    r_encoder_batch = tf.reshape(r_encoder_batch, shape=[_BATCH_SIZE, _CONTEXT_SIZE, 1, 1, _DIM_C_ENC])
    r_encoder = tf.reduce_sum(r_encoder_batch, axis=1)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    r = sess.run(r_encoder, feed_dict={context_poses: np.random.rand(_BATCH_SIZE, _CONTEXT_SIZE, _DIM_POSE), context_frames: np.random.rand(_BATCH_SIZE, _CONTEXT_SIZE, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG)})
    # print(r)
    print(r.shape)
