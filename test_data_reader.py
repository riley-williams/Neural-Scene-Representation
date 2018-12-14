import collections
import glob
import os
import cv2
import numpy as np
import tensorflow as tf

nest = tf.contrib.framework.nest

DatasetInfo = collections.namedtuple('DatasetInfo', ['basepath', 'train_size', 'test_size', 'frame_size', 'sequence_size'])
Context = collections.namedtuple('Context', ['frames', 'cameras'])
Query = collections.namedtuple('Query', ['context', 'query_camera'])
TaskData = collections.namedtuple('TaskData', ['query', 'target'])
_DATASETS = dict(data=DatasetInfo(basepath='data', train_size=1,
                                  test_size=1, frame_size=64, sequence_size=4000))
_NUM_CHANNELS = 3
_NUM_RAW_CAMERA_PARAMS = 5
_MODES = ('train', 'test')


def _read_numpy_data(root, dataset, mode='train'):
    '''
    Return a list of (image, pose) pairs
    '''
    _imgs = []
    _poses = []
    img_files = glob.glob(root + '/' + dataset + '/' + 'test' + '/' + '*.png')
    poses = np.load(root + '/' + dataset + '/' + 'test.npy')[()]
    for f in img_files:
        coded = f.split('/')[-1]
        img = cv2.imread(f)
        img = img[:, :, ::-1]  # BGR to RGB
        pose = poses[coded]
        _p = [pose[0]/10., pose[1]/10., 0.]
        _p += [np.sin(pose[3]), np.cos(pose[3])]
        # _p += [np.sin(pose[4]), np.cos(pose[4])]
        _p += [np.sin(0.), np.cos(0.)]
        _imgs.append(img)
        _poses.append(_p)
    return (np.asarray(_imgs), np.asarray(_poses))


def _generator(imgs, poses, example_size):
    while True:
        idx = np.random.randint(imgs.shape[0], size=example_size)
        yield (imgs[idx], poses[idx])


class DataReader(tf.data.Dataset):
    def __init__(self, dataset, context_size, root, mode='train', custom_frame_size=None, num_threads=4, buffer_size=256, parse_batch_size=32):
        assert dataset in _DATASETS, 'dataset not in _DATASET'
        assert mode in _MODES, 'mode not in _MODES'
        self._dataset_info = _DATASETS[dataset]
        assert context_size < self._dataset_info.sequence_size, 'context_size >= sequence_size'
        self._context_size = context_size
        self._example_size = context_size + 1
        self._custom_frame_size = custom_frame_size
        imgs, poses = _read_numpy_data(root, dataset, mode)
        self._dataset = tf.data.Dataset.from_generator(_generator, (tf.float32, tf.float32), (tf.TensorShape(
            [self._example_size, 64, 64, 3]), tf.TensorShape([self._example_size, 7])), args=(imgs, poses, self._example_size))
        self._dataset = self._dataset.prefetch(buffer_size)
        self._dataset = self._dataset.batch(parse_batch_size)
        self._dataset = self._dataset.apply(tf.contrib.data.unbatch())

    def _as_variant_tensor(self):
        return self._dataset._as_variant_tensor()

    @property
    def output_classes(self):
        return self._dataset.output_classes

    @property
    def output_shapes(self):
        return self._dataset.output_shapes

    @property
    def output_types(self):
        return self._dataset.output_types


def input_fn(dataset, context_size, root, mode, batch_size=1, num_epochs=1, custom_frame_size=None, num_threads=4, buffer_size=256, seed=None):
    if mode == tf.estimator.ModeKeys.TRAIN:
        str_mode = 'train'
    else:
        str_mode = 'test'

    dataset = DataReader(dataset, context_size, root,
                         str_mode, custom_frame_size, num_threads, buffer_size)
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(buffer_size=(
            buffer_size*batch_size), seed=seed)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size*batch_size)
    it = dataset.make_one_shot_iterator()

    frames, cameras = it.get_next()
    context_frames = frames[:, :-1]/255.
    context_cameras = cameras[:, :-1]
    target = frames[:, -1]/255.
    query_camera = cameras[:, -1]
    context = Context(frames=context_frames, cameras=context_cameras)
    query = Query(context=context, query_camera=query_camera)
    return query, target
