import tensorflow as tf
from config import CONFIG as C
from encoder import tower_encoder, pool_encoder
from draw import inference_rnn, generator_rnn
from utils import broadcast_encoding, compute_eta_and_sample_z
from vae import vae_tower_decoder


_ENC_FUNCTIONS = {'pool': pool_encoder, 'tower': tower_encoder}


def _pack_context(poses, frames, model_params):
    _DIM_POSE = model_params.POSE_CHANNELS
    _DIM_H_IMG = model_params.IMG_HEIGHT
    _DIM_W_IMG = model_params.IMG_WIDTH
    _DIM_C_IMG = model_params.IMG_CHANNELS

    poses_packed = tf.reshape(poses, shape=[-1, _DIM_POSE])
    frames_packed = tf.reshape(frames, shape=[-1, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG])
    return poses_packed, frames_packed


def _reduce_packed_representation(enc_r_packed, model_params):
    _CONTEXT_SIZE = model_params.CONTEXT_SIZE
    _DIM_C_ENC = model_params.ENC_CHANNELS

    height, width = tf.shape(enc_r_packed)[1], tf.shape(enc_r_packed)[2]
    enc_r_unpacked = tf.reshape(enc_r_packed, shape=[-1, _CONTEXT_SIZE, height, width, _DIM_C_ENC])
    enc_r = tf.reduce_sum(enc_r_unpacked, axis=1)
    return enc_r


def _encode_context(encoder_fn, poses, frames, model_params):
    endpoints = {}
    poses_packed, frames_packed = _pack_context(poses, frames, model_params)
    enc_r_packed, endpoints_psi = encoder_fn(frames_packed, poses_packed)
    endpoints.update(endpoints_psi)
    enc_r = _reduce_packed_representation(enc_r_packed, model_params)
    endpoints['enc_r'] = enc_r
    return enc_r, endpoints


def gqn_draw(query_pose, target_frame, poses, frames, model_params, is_training=True, scope='GQN'):
    _ENC_TYPE = model_params.ENC_TYPE
    _DIM_H_ENC = model_params.ENC_HEIGHT
    _DIM_W_ENC = model_params.ENC_WIDTH
    _DIM_C_ENC = model_params.ENC_CHANNELS
    _SEQ_LENGTH = model_params.SEQ_LENGTH

    with tf.variable_scope(scope):
        endpoints = {}
        enc_r, endpoints_enc = _encode_context(_ENC_FUNCTIONS[_ENC_TYPE], poses, frames, model_params)
        endpoints.update(endpoints_enc)

        if _ENC_TYPE == 'pool':
            enc_r_broadcast = broadcast_encoding(enc_r, _DIM_H_ENC, _DIM_W_ENC)
        else:
            enc_r_broadcast = tf.reshape(enc_r, [-1, _DIM_H_ENC, _DIM_W_ENC, _DIM_C_ENC])

        if is_training:
            mu_target, endpoints_rnn = inference_rnn(enc_r_broadcast, query_pose, target_frame, _SEQ_LENGTH)
        else:
            mu_target, endpoints_rnn = generator_rnn(enc_r_broadcast, query_pose, _SEQ_LENGTH)
        endpoints.update(endpoints_rnn)
        return mu_target, endpoints


def gqn_vae(query_pose, poses, frames, model_params, scope='GQN_VAE'):
    with tf.variable_scope(scope):
        endpoints = {}
        enc_r, endpoints_enc = _encode_context(tower_encoder, poses, frames, model_params)
        endpoints.update(endpoints_enc)

        mu_z, sigma_z, z = compute_eta_and_sample_z(enc_r, channels=model_params.Z_CHANNELS, scope='Sample_eta')
        endpoints['mu_q'] = mu_z
        endpoints['sigma_q'] = sigma_z

        mu_target, decoder_ep = vae_tower_decoder(z, query_pose)
        endpoints.update(decoder_ep)
        return mu_target, endpoints


if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only run on GPU 1
    import numpy as np
    _BATCH_SIZE = 1
    _CONTEXT_SIZE = C.CONTEXT_SIZE
    _DIM_POSE = C.POSE_CHANNELS
    _DIM_H_IMG = C.IMG_HEIGHT
    _DIM_W_IMG = C.IMG_WIDTH
    _DIM_C_IMG = C.IMG_CHANNELS
    _SEQ_LENGTH = C.SEQ_LENGTH

    query_pose = tf.placeholder(tf.float32, [_BATCH_SIZE, _DIM_POSE])
    target_frame = tf.placeholder(tf.float32, [_BATCH_SIZE, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG])
    poses = tf.placeholder(tf.float32, [_BATCH_SIZE, _CONTEXT_SIZE, _DIM_POSE])
    frames = tf.placeholder(tf.float32, [_BATCH_SIZE, _CONTEXT_SIZE, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG])

    net, ep_gqn = gqn_draw(query_pose, target_frame, poses, frames, C, True)
    net_vae, ep_gqn_vae = gqn_vae(query_pose, poses, frames, C)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        mu, mu_vae = sess.run([net, net_vae], feed_dict={query_pose: np.random.rand(_BATCH_SIZE, _DIM_POSE), target_frame: np.random.rand(_BATCH_SIZE, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG), poses: np.random.rand(_BATCH_SIZE, _CONTEXT_SIZE, _DIM_POSE), frames: np.random.rand(_BATCH_SIZE, _CONTEXT_SIZE, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG)})
    print(mu.shape, mu_vae.shape)
