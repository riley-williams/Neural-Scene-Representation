import tensorflow as tf
from graph import gqn_draw, gqn_vae
from objective import gqn_draw_elbo, gqn_vae_elbo
from config import CONFIG as C
from config import _CONFIG, Params
from utils import debug_canvas_image_mean


def _linear_noise_annealing(gqn_params):
    sigma_i = tf.constant(gqn_params.GENERATOR_SIGMA_ALPHA, dtype=tf.float32)
    sigma_f = tf.constant(gqn_params.GENERATOR_SIGMA_BETA, dtype=tf.float32)
    step = tf.cast(tf.train.get_global_step(), dtype=tf.float32)
    tau = tf.constant(gqn_params.ANNEAL_SIGMA_TAU, dtype=tf.float32)
    sigma_target = tf.maximum(sigma_f + (sigma_i - sigma_f)*(1. - step/tau), sigma_f)
    return sigma_target


def _linear_lr_annealing(gqn_params):
    eta_i = tf.constant(gqn_params.ADAM_LR_ALPHA, dtype=tf.float32)
    eta_f = tf.constant(gqn_params.ADAM_LR_BETA, dtype=tf.float32)
    step = tf.cast(tf.train.get_global_step(), dtype=tf.float32)
    tau = tf.constant(gqn_params.ANNEAL_LR_TAU, dtype=tf.float32)
    lr = tf.maximum(eta_f + (eta_i - eta_f)*(1. - step/tau), eta_f)
    return lr


def gqn_draw_model_fn(features, labels, mode, params):
    _CONTEXT_SIZE = params.CONTEXT_SIZE
    _SEQ_LENGTH = params.SEQ_LENGTH
    query_pose = features.query_camera
    target_frame = labels
    poses = features.context.cameras
    frames = features.context.frames

    mu_target, ep_gqn = gqn_draw(query_pose, target_frame, poses, frames, params, (mode != tf.estimator.ModeKeys.PREDICT))
    # target_normal = tf.distributions.Normal(loc=mu_target, scale=sigma_target)
    # target_sample = tf.identity(target_normal.sample(), name='target_sample')

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'prediction_mean': mu_target, 'frames': frames, 'poses':poses, 'query': query_pose}
        estimator_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    else:
        sigma_target = _linear_noise_annealing(params)
        l2_reconstruction = tf.identity(tf.metrics.mean_squared_error(labels=target_frame, predictions=mu_target))
        if params.DEBUG:
            for i in range(_CONTEXT_SIZE):
                tf.summary.image('context_frame_{}'.format(i+1), frames[:, i], max_outputs=1)
            tf.summary.image('target_images', labels, max_outputs=1)
            tf.summary.image('target_mean', mu_target, max_outputs=1)
            tf.summary.scalar('l2_reconstruction', l2_reconstruction[1])
            generator_sequence = debug_canvas_image_mean([ep_gqn['canvas_{}'.format(i)] for i in range(_SEQ_LENGTH)])
            tf.summary.image('generate_sequence_mean', generator_sequence, max_outputs=1)
        mu_q, sigma_q, mu_pi, sigma_pi = [], [], [], []
        for i in range(_SEQ_LENGTH):
            mu_q.append(ep_gqn['mu_q_{}'.format(i)])
            sigma_q.append(ep_gqn['sigma_q_{}'.format(i)])
            mu_pi.append(ep_gqn['mu_pi_{}'.format(i)])
            sigma_pi.append(ep_gqn['sigma_pi_{}'.format(i)])
        elbo, ep_elbo = gqn_draw_elbo(mu_target, sigma_target, mu_q, sigma_q, mu_pi, sigma_pi, target_frame)
        if params.DEBUG:
            tf.summary.scalar('target_llh', ep_elbo['target_llh'])
            tf.summary.scalar('kl_regularizer', ep_elbo['kl_regularizer'])

        if mode == tf.estimator.ModeKeys.TRAIN:
            lr = _linear_lr_annealing(params)
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            train_op = optimizer.minimize(elbo, tf.train.get_global_step())
            estimator_spec = tf.estimator.EstimatorSpec(mode=mode, loss=elbo, train_op=train_op)
        else:
            eval_metric_ops = {'l2_reconstruction': tf.metrics.mean_squared_error(target_frame, mu_target)}
            estimator_spec = tf.estimator.EstimatorSpec(mode=mode, loss=elbo, eval_metric_ops=eval_metric_ops)
    return estimator_spec


def gqn_draw_identity_model_fn(features, labels, mode, params=None):
    _CONFIG['CONTEXT_SIZE'] = 1
    params = Params(**_CONFIG)
    _SEQ_LENGTH = params.SEQ_LENGTH

    query_pose = features.query_camera
    target_frame = labels
    poses = features.query_camera
    frames = labels

    mu_target, ep_gqn = gqn_draw(query_pose, target_frame, poses, frames, params, mode != tf.estimator.ModeKeys.PREDICT)
    sigma_target = _linear_noise_annealing(params)
    target_normal = tf.distributions.Normal(loc=mu_target, scale=sigma_target)
    target_sample = tf.identity(target_normal.sample, name='target_sample')
    l2_reconstruction = tf.identity(tf.metrics.mean_squared_error(labels=target_frame, predictions=mu_target))
    if params.DEBUG:
        tf.summary.image('context_frame_1', frames, max_outputs=1)
        tf.summary.image('target_images', labels, max_outputs=1)
        tf.summary.image('target_means', mu_target, max_outputs=1)
        tf.summary.scalar('l2_reconstruction', l2_reconstruction[1])
        gs = debug_canvas_image_mean([ep_gqn['canvas_{}'.format(i)] for i in range(_SEQ_LENGTH)])
        tf.summary.image('generator_sequence_mean', gs, max_outputs=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'target_sample': target_sample}
        estimator_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    else:
        mu_q, sigma_q, mu_pi, sigma_pi = [], [], [], []
        for i in range(_SEQ_LENGTH):
            mu_q.append(ep_gqn['mu_q_{}'.format(i)])
            sigma_q.append(ep_gqn['sigma_q_{}'.format(i)])
            mu_pi.append(ep_gqn['mu_pi_{}'.format(i)])
            sigma_pi.append(ep_gqn['sigma_pi_{}'.format(i)])
        elbo, ep_elbo = gqn_draw_elbo(mu_target, sigma_target, mu_q, sigma_q, mu_pi, sigma_pi, target_frame)
        if params.DEBUG:
            tf.summary.scalar('target_llh', ep_elbo['target_llh'])
            tf.summary.scalar('kl_regularizer', ep_elbo['kl_regularizer'])

        if mode == tf.estimator.ModeKeys.TRAIN:
            lr = _linear_lr_annealing(params)
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            train_op = optimizer.minimize(elbo, tf.train.get_global_step())
            estimator_spec = tf.estimator.EstimatorSpec(mode=mode, loss=elbo, train_op=train_op)
        else:
            eval_metric_ops = {'l2_reconstruction': tf.metrics.mean_squared_error(target_frame, mu_target)}
            estimator_spec = tf.estimator.EstimatorSpec(mode=mode, loss=elbo, eval_metric_ops=eval_metric_ops)
    return estimator_spec


def gqn_vae_model_fn(features, labels, mode, params):
    query_pose = features.query_camera
    target_frame = labels
    poses = features.context.cameras
    frames = features.context.frames

    mu_target, ep_gqn_vae = gqn_vae(query_pose, target_frame, poses, frames, params)
    mu_q, sigma_q = ep_gqn_vae['mu_q'], ep_gqn_vae['sigma_q']
    sigma_target = _linear_noise_annealing(params)
    target_normal = tf.distributions.Normal(loc=mu_target, scale=sigma_target)
    target_sample = tf.identity(target_normal.sample, name='target_sample')
    if params.DEBUG:
        tf.summary.image('target_images', labels, max_outputs=1)
        tf.summary.image('target_means', mu_target, max_outputs=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'target_sample': target_sample}
        estimator_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    else:
        elbo, ep_elbo = gqn_vae_elbo(mu_target, sigma_target, mu_q, sigma_q, target_frame)

        if mode == tf.estimator.ModeKeys.TRAIN:
            lr = _linear_lr_annealing(params)
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            train_op = optimizer.minimize(elbo, tf.train.get_global_step())
            estimator_spec = tf.estimator.EstimatorSpec(mode=mode, loss=elbo, train_op=train_op)
        else:
            eval_metric_ops = {'l2_reconstruction': tf.metrics.mean_squared_error(target_frame, mu_target)}
            estimator_spec = tf.estimator.EstimatorSpec(mode=mode, loss=elbo, eval_metric_ops=eval_metric_ops)
    return estimator_spec
