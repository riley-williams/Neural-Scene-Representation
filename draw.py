import tensorflow as tf
from config import CONFIG as C
from gqn_lstm import _GeneratorCellInput, GeneratorLSTMCell, _InferenceCellInput, InferenceLSTMCell
from utils import broadcast_pose, eta_g, compute_eta_and_sample_z, sample_z


def generator_rnn(representations, query_poses, sequence_size=12, scope='GQN_RNN'):
    dim_r = representations.get_shape().as_list()
    batch = tf.shape(representations)[0]
    height, width = dim_r[1], dim_r[2]
    cell = GeneratorLSTMCell(input_shape=[height, width, C.GENERATOR_INPUT_CHANNELS], output_channels=C.LSTM_OUTPUT_CHANNELS, canvas_channels=C.LSTM_CANVAS_CHANNELS, kernel_size=C.LSTM_KERNEL_SIZE, name='GeneratorCell')
    outputs = []
    endpoints = {}
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as var_scope:
        if not tf.executing_eagerly():
            if var_scope.caching_device is None:
                var_scope.set_caching_device(lambda op: op.device)

        query_poses = broadcast_pose(query_poses, height, width)
        state = cell.zero_state(batch, tf.float32)

        for step in range(sequence_size):
            z = sample_z(state.lstm.h, scope='sample_eta_pi')
            inputs = _GeneratorCellInput(representations, query_poses, z)
            with tf.name_scope('Generator'):
                (output, state) = cell(inputs, state, 'LSTM_gen')
            ep_canvas = 'canvas_{}'.format(step)
            endpoints[ep_canvas] = output.canvas
            outputs.append(output)
        target_canvas = outputs[-1].canvas
    mu_target = eta_g(target_canvas, channels=C.IMG_CHANNELS, scope='eta_g')
    endpoints['mu_target'] = mu_target
    return mu_target, endpoints


def inference_rnn(representations, query_poses, target_frames, sequence_size=12, scope='GQN_RNN'):
    dim_r = representations.get_shape().as_list()
    batch = tf.shape(representations)[0]
    height, width = dim_r[1], dim_r[2]
    generator_cell = GeneratorLSTMCell(input_shape=[height, width, C.GENERATOR_INPUT_CHANNELS], output_channels=C.LSTM_OUTPUT_CHANNELS, canvas_channels=C.LSTM_CANVAS_CHANNELS, kernel_size=C.LSTM_KERNEL_SIZE, name='GeneratorCell')
    inference_cell = InferenceLSTMCell(input_shape=[height, width, C.INFERENCE_INPUT_CHANNELS], output_channels=C.LSTM_OUTPUT_CHANNELS, kernel_size=C.LSTM_KERNEL_SIZE, name='InferenceCell')
    outputs = []
    endpoints = {}
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as var_scope:
        if not tf.executing_eagerly():
            if var_scope.caching_device is None:
                var_scope.set_caching_device(lambda op: op.device)

        query_poses = broadcast_pose(query_poses, height, width)
        inf_state = inference_cell.zero_state(batch, tf.float32)
        gen_state = generator_cell.zero_state(batch, tf.float32)

        for step in range(sequence_size):
            inf_input = _InferenceCellInput(representations, query_poses, target_frames, gen_state.canvas, gen_state.lstm.h)
            with tf.name_scope('Inference'):
                (inf_output, inf_state) = inference_cell(inf_input, inf_state, 'LSTM_inf')
            mu_q, sigma_q, z_q = compute_eta_and_sample_z(inf_state.lstm.h, scope='sample_eta_q')

            gen_input = _GeneratorCellInput(representations, query_poses, z_q)
            with tf.name_scope('Generator'):
                (gen_output, gen_state) = generator_cell(gen_input, gen_state, 'LSTM_gen')
            mu_pi, sigma_pi, z_pi = compute_eta_and_sample_z(gen_state.lstm.h, scope='sample_eta_pi')

            outputs.append((inf_output, gen_output))

            endpoints['mu_q_{}'.format(step)] = mu_q
            endpoints['mu_pi_{}'.format(step)] = mu_pi
            endpoints['sigma_q_{}'.format(step)] = sigma_q
            endpoints['sigma_pi_{}'.format(step)] = sigma_pi
            endpoints['canvas_{}'.format(step)] = gen_output.canvas
        target_canvas = outputs[-1][1].canvas
    mu_target = eta_g(target_canvas, channels=C.IMG_CHANNELS, scope='eta_g')
    endpoints['mu_target'] = mu_target
    return mu_target, endpoints


if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only run on GPU 1
    import numpy as np
    _BATCH_SIZE = 3
    _CONTEXT_SIZE = 5
    _DIM_POSE = C.POSE_CHANNELS
    _DIM_H_IMG = C.IMG_HEIGHT
    _DIM_W_IMG = C.IMG_WIDTH
    _DIM_C_IMG = C.IMG_CHANNELS
    _DIM_R_H = C.ENC_HEIGHT
    _DIM_R_W = C.ENC_WIDTH
    _DIM_R_C = C.ENC_CHANNELS
    _SEQ_LENGTH = C.SEQ_LENGTH

    query_pose = tf.placeholder(tf.float32, [_BATCH_SIZE, _DIM_POSE])
    target_frame = tf.placeholder(tf.float32, [_BATCH_SIZE, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG])
    scene_representation = tf.placeholder(tf.float32, [_BATCH_SIZE, _DIM_R_H, _DIM_R_W, _DIM_R_C])

    mu_target_gen, ep_gen = generator_rnn(scene_representation, query_pose, 2)
    mu_target_inf, ep_inf = inference_rnn(scene_representation, query_pose, target_frame, _SEQ_LENGTH)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    mu_gen = sess.run(mu_target_gen, feed_dict={query_pose: np.random.rand(_BATCH_SIZE, _DIM_POSE), scene_representation: np.random.rand(_BATCH_SIZE, _DIM_R_H, _DIM_R_W, _DIM_R_C)})
    mu_inf = sess.run(mu_target_inf, feed_dict={query_pose: np.random.rand(_BATCH_SIZE, _DIM_POSE), target_frame: np.random.rand(_BATCH_SIZE, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG), scene_representation: np.random.rand(_BATCH_SIZE, _DIM_R_H, _DIM_R_W, _DIM_R_C)})

    print('Generator Test:')
    # print(mu_gen)
    print(mu_gen.shape)
    for ep, t in ep_gen.items():
        print(ep, t)

    print('\n\nInference Test:')
    # print(mu_inf)
    print(mu_inf.shape)
    for ep, t in ep_inf.items():
        print(ep, t)
