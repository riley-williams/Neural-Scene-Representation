import tensorflow as tf


def gqn_draw_elbo(mu_target, sigma_target, mu_q, sigma_q, mu_pi, sigma_pi, target_frame, scope='GQN_Draw_ELBO'):
    with tf.variable_scope(scope):
        endpoints = {}
        target_normal = tf.distributions.Normal(loc=mu_target, scale=sigma_target)
        target_llh = tf.identity(input=-tf.reduce_sum(tf.reduce_mean(target_normal.log_prob(target_frame), axis=0)), name='target_log_likelihood')
        endpoints['target_llh'] = target_llh
        kl_div_list = []
        for mu_q_l, sigma_q_l, mu_pi_l, sigma_pi_l in zip(mu_q, sigma_q, mu_pi, sigma_pi):
            posterior_normal_l = tf.distributions.Normal(loc=mu_q_l, scale=sigma_q_l)
            prior_normal_l = tf.distributions.Normal(loc=mu_pi_l, scale=sigma_pi_l)
            kl_div_l = tf.distributions.kl_divergence(posterior_normal_l, prior_normal_l)
            kl_div_list.append(kl_div_l)
        kl_regularizer = tf.identity(input=tf.reduce_sum(tf.reduce_mean(tf.add_n(kl_div_list), axis=0)), name='kl_regularizer')
        endpoints['kl_regularizer'] = kl_regularizer
        elbo = target_llh + kl_regularizer
        return elbo, endpoints


def gqn_vae_elbo(mu_target, sigma_target, mu_q, sigma_q, target_frame, scope='GQN_VAE_ELBO'):
    with tf.variable_scope(scope):
        target_normal = tf.distributions.Normal(loc=mu_target, scale=sigma_target)
        target_llh = tf.identity(input=-tf.reduce_sum(tf.reduce_mean(target_normal.log_prob(target_frame), axis=0)), name='target_log_likelihood')
        posterior_normal = tf.distributions.Normal(loc=mu_q, scale=sigma_q)
        prior_normal = tf.distributions.Normal(loc=tf.zeros_like(mu_q), scale=tf.ones_like(sigma_q))
        kl_div = tf.distributions.kl_divergence(posterior_normal, prior_normal)
        kl_regularizer = tf.identity(input=tf.reduce_sum(tf.reduce_mean(tf.add_n(kl_div), axis=0)), name='kl_regularizer')
        elbo = target_llh + kl_regularizer
        return elbo


if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only run on GPU 1
    import numpy as np
    from config import CONFIG as C
    from graph import gqn_draw
    _BATCH_SIZE = 1
    _CONTEXT_SIZE = C.CONTEXT_SIZE
    _DIM_POSE = C.POSE_CHANNELS
    _DIM_H_IMG = C.IMG_HEIGHT
    _DIM_W_IMG = C.IMG_WIDTH
    _DIM_C_IMG = C.IMG_CHANNELS
    _SEQ_LENGTH = C.SEQ_LENGTH
    MAX_TRAIN_STEP = 50

    query_pose = tf.placeholder(tf.float32, shape=[_BATCH_SIZE, _DIM_POSE])
    target_frame = tf.placeholder(tf.float32, shape=[_BATCH_SIZE, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG])
    poses = tf.placeholder(tf.float32, [_BATCH_SIZE, _CONTEXT_SIZE, _DIM_POSE])
    frames = tf.placeholder(tf.float32, [_BATCH_SIZE, _CONTEXT_SIZE, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG])

    mu_target, ep_gqn = gqn_draw(query_pose, target_frame, poses, frames, C, True)
    sigma_target = tf.constant(1., tf.float32, [_BATCH_SIZE, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG])
    mu_q, sigma_q, mu_pi, sigma_pi = [], [], [], []
    for i in range(_SEQ_LENGTH):
        mu_q.append(ep_gqn['mu_q_{}'.format(i)])
        sigma_q.append(ep_gqn['sigma_q_{}'.format(i)])
        mu_pi.append(ep_gqn['mu_pi_{}'.format(i)])
        sigma_pi.append(ep_gqn['sigma_pi_{}'.format(i)])
    elbo, ep_elbo = gqn_draw_elbo(mu_target, sigma_target, mu_q, sigma_q, mu_pi, sigma_pi, target_frame)
    optimizer = tf.train.AdamOptimizer()
    grad_vars = optimizer.compute_gradients(loss=elbo)
    updates = optimizer.apply_gradients(grads_and_vars=grad_vars)

    _query_pose = np.random.rand(_BATCH_SIZE, _DIM_POSE)
    _target_frame = np.random.rand(_BATCH_SIZE, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG)
    _poses = np.random.rand(_BATCH_SIZE, _CONTEXT_SIZE, _DIM_POSE)
    _frames = np.random.rand(_BATCH_SIZE, _CONTEXT_SIZE, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # loss = sess.run(elbo, feed_dict={query_pose: _query_pose, target_frame: _target_frame, poses: _poses, frames: _frames})
        # print(loss.shape)
        for step in range(MAX_TRAIN_STEP):
            _elbo, _grad_vars, _updates = sess.run([elbo, grad_vars, updates], feed_dict={query_pose: _query_pose, target_frame: _target_frame, poses: _poses, frames: _frames})
            if step == 0:
                for _grad, _var in _grad_vars:
                    print(_grad.shape, _var.shape)
            print(_elbo)
