import tensorflow as tf
from config import CONFIG as C


def add_scope(fn, scope):
    def _wrapper(*args, **kwargs):
        if scope is not None:
            with tf.variable_scope(scope):
                return fn(*args, **kwargs)
        else:
            return fn(*args, **kwargs)
    return _wrapper


def optional_scope_default(default_scope=None):
    def _optional_scope(fn):
        def extract_and_add_scope(*args, **kwargs):
            scope = kwargs.pop('scope', default_scope)
            return add_scope(fn, scope)(*args, **kwargs)
        return extract_and_add_scope
    return _optional_scope


optional_scope = optional_scope_default(None)


def create_sub_scope(scope, name):
    if scope is None:
        var_scope = tf.get_variable_scope()
    else:
        var_scope = tf.variable_scope(scope)
    with var_scope, tf.variable_scope(name) as sub_var_scope:
        return sub_var_scope


def broadcast_pose(pose_vec, height, width):
    ret = tf.reshape(pose_vec, [-1, 1, 1, C.POSE_CHANNELS])
    ret = tf.tile(ret, [1, height, width, 1])
    return ret


def broadcast_encoding(encoding_vec, height, width):
    ret = tf.reshape(encoding_vec, [-1, 1, 1, C.ENC_CHANNELS])
    ret = tf.tile(ret, [1, height, width, 1])
    return ret


def eta_g(canvas, kernel_size=C.ETA_EXTERNAL_KERNEL_SIZE, channels=C.IMG_CHANNELS, scope='eta_g'):
    return tf.layers.conv2d(canvas, filters=channels, kernel_size=kernel_size, padding='SAME', name=scope)


@optional_scope
def eta(h, kernel_size=C.LSTM_KERNEL_SIZE, channels=C.Z_CHANNELS):
    eta = tf.layers.conv2d(h, filters=2*channels, kernel_size=kernel_size, padding='SAME')
    mu, sigma = tf.split(eta, num_or_size_splits=2, axis=-1)
    sigma = tf.nn.softplus(sigma + 0.5) + C.EPS_PARAM
    return mu, sigma


@optional_scope
def compute_eta_and_sample_z(h, kernel_size=C.ETA_INTERNAL_KERNEL_SIZE, channels=C.Z_CHANNELS):
    mu, sigma = eta(h, kernel_size, channels, scope='eta')
    with tf.variable_scope('sampling'):
        z_shape = tf.concat([tf.shape(h)[:-1], [channels]], axis=0, name='create_z_shape')
        z = mu + tf.multiply(sigma, tf.random_normal(shape=z_shape))
    return mu, sigma, z


@optional_scope
def sample_z(h, kernel_size=C.ETA_INTERNAL_KERNEL_SIZE, channels=C.Z_CHANNELS):
    _, _, z = compute_eta_and_sample_z(h, kernel_size, channels)
    return z


_BAR_WIDTH = 2


def debug_canvas_image_mean(canvases, eta_g_scope='GQN'):
    with tf.variable_scope(eta_g_scope, reuse=True, auxiliary_name_scope=False), tf.name_scope('debug'):
        mean_images = []
        with tf.name_scope('make_white_bar'):
            cs = tf.shape(canvases[0])
            batch, height, channels = cs[0], cs[1], 3
            white_vertical_bar = tf.ones(shape=(batch, height, _BAR_WIDTH, channels), dtype=tf.float32, name='white_bar')

        for canvas in canvases:
            mean_images.append(eta_g(canvas))
            mean_images.append(white_vertical_bar)
        return tf.concat(mean_images[:-1], axis=-2, name='make_grid')
