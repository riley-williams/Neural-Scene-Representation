from collections import namedtuple

_CONFIG = {
    'EPS_PARAM': 1e-8,
    'IMG_HEIGHT': 64,
    'IMG_WIDTH': 64,
    'IMG_CHANNELS': 3,
    'POSE_CHANNELS': 7,
    # input parameters
    'CONTEXT_SIZE': 5,
    # hyper-parameters: scene representation
    'ENC_TYPE': 'pool',  # encoding architecture used: pool | tower
    'ENC_HEIGHT': 16,
    'ENC_WIDTH': 16,
    'ENC_CHANNELS': 256,
    # hyper-parameters: generator LSTM
    'LSTM_OUTPUT_CHANNELS': 256,
    'LSTM_CANVAS_CHANNELS': 256,
    'LSTM_KERNEL_SIZE': 5,
    'Z_CHANNELS': 64,  # latent space size per image generation step
    'GENERATOR_INPUT_CHANNELS': 256+5+64,  # pose + representation + z
    'INFERENCE_INPUT_CHANNELS': 256+5,  # pose + representation
    'SEQ_LENGTH': 8,  # number image generation steps, orig.: 12
    # hyper-parameters: eta functions
    'ETA_INTERNAL_KERNEL_SIZE': 5,  # internal projection of states to means and variances
    'ETA_EXTERNAL_KERNEL_SIZE': 1,  # kernel size for final projection of canvas to mean image
    # hyper-parameters: ADAM optimization
    'ANNEAL_SIGMA_TAU': 200000,  # annealing interval for global noise
    'GENERATOR_SIGMA_ALPHA': 2.0,  # start value for global generation variance
    'GENERATOR_SIGMA_BETA': 0.7,  # final value for global generation variance
    'ANNEAL_LR_TAU': 1600000,  # annealing interval for learning rate
    'ADAM_LR_ALPHA': 5 * 10e-5,  # start learning rate of ADAM optimizer, orig.: 5 * 10e-4
    'ADAM_LR_BETA': 5 * 10e-6,  # final learning rate of ADAM optimizer, orig.: 5 * 10e-5

    'DEBUG': False,
    'BATCH_SIZE': 36,
    'TRAIN_EPOCHS': 40,
    'CKPT_STEPS': 10000,
    # 'DATA_DIR': '/media/gdao/Data/GQN',
    # 'DATASET': 'data',
    'MODEL_DIR': '/media/gdao/Data/GQN/models/rooms_ring_camera',
    'DATA_DIR': '/media/gdao/Data/GQN',
    'DATASET': 'data',
    'QUEUE_THREAD': 4,
    'QUEUE_BUFFER': 32,
    'LOG_STEPS': 100,
    'INITAIL_EVAL': False,
    'N': 9
}

Params = namedtuple('GQNParam', list(_CONFIG.keys()))
CONFIG = Params(**_CONFIG)
