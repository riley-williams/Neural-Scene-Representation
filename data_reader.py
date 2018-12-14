import collections
import os
import tensorflow as tf
from config import CONFIG as C
nest = tf.contrib.framework.nest  # pylint: disable=E1101

DatasetInfo = collections.namedtuple('DatasetInfo', ['basepath', 'train_size', 'test_size', 'frame_size', 'sequence_size'])
Context = collections.namedtuple('Context', ['frames', 'cameras'])
Query = collections.namedtuple('Query', ['context', 'query_camera'])
TaskData = collections.namedtuple('TaskData', ['query', 'target'])


_DATASETS = dict(
    jaco=DatasetInfo(
        basepath='jaco',
        train_size=3600,
        test_size=400,
        frame_size=64,
        sequence_size=11),

    data=DatasetInfo(
        basepath='data',
        train_size=1,
        test_size=1,
        frame_size=64,
        sequence_size=10),

    mazes=DatasetInfo(
        basepath='mazes',
        train_size=1080,
        test_size=120,
        frame_size=84,
        sequence_size=300),

    rooms_free_camera_with_object_rotations=DatasetInfo(
        basepath='rooms_free_camera_with_object_rotations',
        train_size=2034,
        test_size=226,
        frame_size=128,
        sequence_size=10),

    rooms_ring_camera=DatasetInfo(
        basepath='rooms_ring_camera',
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10),

    rooms_free_camera_no_object_rotations=DatasetInfo(
        basepath='rooms_free_camera_no_object_rotations',
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10),

    shepard_metzler_5_parts=DatasetInfo(
        basepath='shepard_metzler_5_parts',
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15),

    shepard_metzler_7_parts=DatasetInfo(
        basepath='shepard_metzler_7_parts',
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15)
)
_NUM_CHANNELS = 3
_NUM_RAW_CAMERA_PARAMS = 5
_MODES = ('train', 'test')


def _get_dataset_files(dateset_info, mode, root):
    """Generates lists of files for a given dataset version."""
    basepath = dateset_info.basepath
    base = os.path.join(root, basepath, mode)
    if mode == 'train':
        num_files = dateset_info.train_size
    else:
        num_files = dateset_info.test_size

    length = len(str(num_files))
    template = '{:0%d}-of-{:0%d}.tfrecord' % (length, length)
    return [os.path.join(base, template.format(i, num_files)) for i in range(1, num_files+1)]


def _convert_frame_data(jpeg_data):
    decoded_frames = tf.image.decode_jpeg(jpeg_data)
    return tf.image.convert_image_dtype(decoded_frames, dtype=tf.float32)


class GQNTFRecordDataset(tf.data.Dataset):
    def __init__(self, dataset, context_size, root, mode='train', custom_frame_size=None, num_threads=4, buffer_size=256, parse_batch_size=32):
        if dataset not in _DATASETS:
            raise ValueError('Cannot recognize dataset {}'.format(dataset))
        if mode not in _MODES:
            raise ValueError('Cannot recognize mode {}'.format(mode))
        self._dataset_info = _DATASETS[dataset]
        if context_size > self._dataset_info.sequence_size:
            raise ValueError('{} > {}'.format(context_size, self._dataset_info.sequence_size))
        self._context_size = context_size
        self._example_size = context_size + 1
        self._custom_frame_size = custom_frame_size

        self._feature_map = {'frames': tf.FixedLenFeature(shape=self._dataset_info.sequence_size, dtype=tf.string), 'cameras': tf.FixedLenFeature(shape=[self._dataset_info.sequence_size*_NUM_RAW_CAMERA_PARAMS], dtype=tf.float32)}
        file_names = _get_dataset_files(self._dataset_info, mode, root)
        self._dataset = tf.data.TFRecordDataset(file_names, num_parallel_reads=num_threads)
        self._dataset = self._dataset.prefetch(buffer_size)
        self._dataset = self._dataset.batch(parse_batch_size)
        self._dataset = self._dataset.map(self._parse_record, num_parallel_calls=num_threads)
        self._dataset = self._dataset.apply(tf.contrib.data.unbatch())  # pylint: disable=E1101


    def _parse_record(self, raw_data):
        example = tf.parse_example(raw_data, self._feature_map)
        indices = self._get_randomized_indices()
        frames = self._preprocess_frames(example, indices)
        cameras = self._preprocess_camera(example, indices)
        return frames, cameras

    def _get_randomized_indices(self):
        indices = tf.range(0, self._dataset_info.sequence_size)
        # indices = tf.random_shuffle(indices)
        indices = tf.slice(indices, begin=[0], size=[self._example_size])
        return indices

    def _preprocess_frames(self, example, indices):
        frames = tf.concat(example['frames'], axis=0)
        frames = tf.gather(frames, indices, axis=1)
        frames = tf.map_fn(_convert_frame_data, tf.reshape(frames, [-1]), dtype=tf.float32, back_prop=False)
        dataset_image_dimensions = tuple([self._dataset_info.frame_size]*2 + [_NUM_CHANNELS])
        frames = tf.reshape(frames, (-1, self._example_size) + dataset_image_dimensions)
        if self._custom_frame_size and self._custom_frame_size != self._dataset_info.frame_size:
            frames = tf.reshape(frames, (-1, ) + dataset_image_dimensions)
            new_frame_dimensions = (self._custom_frame_size, )*2 + (_NUM_CHANNELS, )
            frames = tf.image.resize_bilinear(frames, new_frame_dimensions[:2], align_corners=True)
            frames = tf.reshape(frames, (-1, self._example_size) + new_frame_dimensions)
        return frames

    def _preprocess_camera(self, example, indices):
        raw_pose_params = example['cameras']
        raw_pose_params = tf.reshape(raw_pose_params, [-1, self._dataset_info.sequence_size, _NUM_RAW_CAMERA_PARAMS])
        raw_pose_params = tf.gather(raw_pose_params, indices, axis=1)
        pos = raw_pose_params[:, :, 0:3]
        yaw = raw_pose_params[:, :, 3:4]
        pitch = raw_pose_params[:, :, 4:5]
        cameras = tf.concat([pos, tf.sin(yaw), tf.cos(yaw), tf.sin(pitch), tf.cos(pitch)], axis=2)
        return cameras

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


def gqn_input_fn(dataset, context_size, root, mode, batch_size=1, num_epochs=1, custom_frame_size=None, num_threads=4, buffer_size=256, seed=None):
    if mode == tf.estimator.ModeKeys.TRAIN:
        str_mode = 'train'
    else:
        str_mode = 'test'

    dataset = GQNTFRecordDataset(dataset, context_size, root, str_mode, custom_frame_size, num_threads, buffer_size)
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(buffer_size=(buffer_size*batch_size), seed=seed)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size*batch_size)
    it = dataset.make_one_shot_iterator()

    frames, cameras = it.get_next()
    context_frames = frames[:, :-1]
    context_cameras = cameras[:, :-1]
    target = frames[:, -1]
    query_camera = cameras[:, -1]
    context = Context(frames=context_frames, cameras=context_cameras)
    query = Query(context=context, query_camera=query_camera)
    return query, target


def gqn_input_fn2(dataset, context_size, root, mode, batch_size=1, num_epochs=1, custom_frame_size=None, num_threads=4, buffer_size=256, seed=None):
    if mode == tf.estimator.ModeKeys.TRAIN:
        str_mode = 'train'
    else:
        str_mode = 'test'

    dataset = GQNTFRecordDataset(dataset, context_size, root, str_mode, custom_frame_size, num_threads, buffer_size)
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(buffer_size=(buffer_size*batch_size), seed=seed)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size*batch_size)
    it = dataset.make_one_shot_iterator()

    frames, cameras = it.get_next()
    context_frames = frames[:, 1:6]
    context_cameras = cameras[:, 1:6]
    target = frames[:, -1]
    rate = (C.N - 1)/10
    xyz = (1.-rate)*cameras[0:3, 1] + rate*cameras[0:3, 3]
    wr = cameras[3:7, 1]
    query_camera = tf.concat([xyz, wr], axis=0)
    context = Context(frames=context_frames, cameras=context_cameras)
    query = Query(context=context, query_camera=query_camera)
    return query, target


class DataReader(object):  # From DeepMind
  def __init__(self,
               dataset,
               context_size,
               root,
               mode='train',
               # Optionally reshape frames
               custom_frame_size=None,
               # Queue params
               num_threads=4,
               capacity=256,
               min_after_dequeue=128,
               seed=None):
    if dataset not in _DATASETS:
      raise ValueError('Unrecognized dataset {} requested. Available datasets '
                       'are {}'.format(dataset, _DATASETS.keys()))

    if mode not in _MODES:
      raise ValueError('Unsupported mode {} requested. Supported modes '
                       'are {}'.format(mode, _MODES))

    self._dataset_info = _DATASETS[dataset]

    if context_size >= self._dataset_info.sequence_size:
      raise ValueError(
          'Maximum support context size for dataset {} is {}, but '
          'was {}.'.format(
              dataset, self._dataset_info.sequence_size-1, context_size))

    self._context_size = context_size
    # Number of views in the context + target view
    self._example_size = context_size + 1
    self._custom_frame_size = custom_frame_size

    with tf.device('/cpu'):
      file_names = _get_dataset_files(self._dataset_info, mode, root)
      filename_queue = tf.train.string_input_producer(file_names, seed=seed)
      reader = tf.TFRecordReader()

      read_ops = [self._make_read_op(reader, filename_queue)
                  for _ in range(num_threads)]

      dtypes = nest.map_structure(lambda x: x.dtype, read_ops[0])
      shapes = nest.map_structure(lambda x: x.shape[1:], read_ops[0])

      self._queue = tf.RandomShuffleQueue(
          capacity=capacity,
          min_after_dequeue=min_after_dequeue,
          dtypes=dtypes,
          shapes=shapes,
          seed=seed)

      enqueue_ops = [self._queue.enqueue_many(op) for op in read_ops]
      tf.train.add_queue_runner(tf.train.QueueRunner(self._queue, enqueue_ops))

  def read(self, batch_size):
    """Reads batch_size (query, target) pairs."""
    frames, cameras = self._queue.dequeue_many(batch_size)
    context_frames = frames[:, :-1]
    context_cameras = cameras[:, :-1]
    target = frames[:, -1]
    query_camera = cameras[:, -1]
    context = Context(cameras=context_cameras, frames=context_frames)
    query = Query(context=context, query_camera=query_camera)
    return TaskData(query=query, target=target)

  def _make_read_op(self, reader, filename_queue):
    """Instantiates the ops used to read and parse the data into tensors."""
    _, raw_data = reader.read_up_to(filename_queue, num_records=16)
    feature_map = {
        'frames': tf.FixedLenFeature(
            shape=self._dataset_info.sequence_size, dtype=tf.string),
        'cameras': tf.FixedLenFeature(
            shape=[self._dataset_info.sequence_size * _NUM_RAW_CAMERA_PARAMS],
            dtype=tf.float32)
    }
    example = tf.parse_example(raw_data, feature_map)
    indices = self._get_randomized_indices()
    frames = self._preprocess_frames(example, indices)
    cameras = self._preprocess_cameras(example, indices)
    return frames, cameras

  def _get_randomized_indices(self):
    """Generates randomized indices into a sequence of a specific length."""
    indices = tf.range(0, self._dataset_info.sequence_size)
    indices = tf.random_shuffle(indices)
    indices = tf.slice(indices, begin=[0], size=[self._example_size])
    return indices

  def _preprocess_frames(self, example, indices):
    """Instantiates the ops used to preprocess the frames data."""
    frames = tf.concat(example['frames'], axis=0)
    frames = tf.gather(frames, indices, axis=1)
    frames = tf.map_fn(
        _convert_frame_data, tf.reshape(frames, [-1]),
        dtype=tf.float32, back_prop=False)
    dataset_image_dimensions = tuple(
        [self._dataset_info.frame_size] * 2 + [_NUM_CHANNELS])
    frames = tf.reshape(
        frames, (-1, self._example_size) + dataset_image_dimensions)
    if (self._custom_frame_size and self._custom_frame_size != self._dataset_info.frame_size):
      frames = tf.reshape(frames, (-1,) + dataset_image_dimensions)
      new_frame_dimensions = (self._custom_frame_size,) * 2 + (_NUM_CHANNELS,)
      frames = tf.image.resize_bilinear(
          frames, new_frame_dimensions[:2], align_corners=True)
      frames = tf.reshape(
          frames, (-1, self._example_size) + new_frame_dimensions)
    return frames

  def _preprocess_cameras(self, example, indices):
    """Instantiates the ops used to preprocess the cameras data."""
    raw_pose_params = example['cameras']
    raw_pose_params = tf.reshape(
        raw_pose_params,
        [-1, self._dataset_info.sequence_size, _NUM_RAW_CAMERA_PARAMS])
    raw_pose_params = tf.gather(raw_pose_params, indices, axis=1)
    pos = raw_pose_params[:, :, 0:3]
    yaw = raw_pose_params[:, :, 3:4]
    pitch = raw_pose_params[:, :, 4:5]
    cameras = tf.concat(
        [pos, tf.sin(yaw), tf.cos(yaw), tf.sin(pitch), tf.cos(pitch)], axis=2)
    return cameras


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only run on GPU 1
    ROOT_PATH = '/Users/giangdao/Desktop'
    DATASET_NAME = 'rooms_ring_camera'
    CTX_SIZE = 5
    BATCH_SIZE = 1
    # data_reader = DataReader(dataset=DATASET_NAME, context_size=CTX_SIZE, root=ROOT_PATH)
    # data = data_reader.read(batch_size=BATCH_SIZE)
    # sess = tf.train.SingularMonitoredSession()
    # d = sess.run(data)
    # print("Shapes of fetched tensors:")
    # print("Query camera poses: %s" % str(d.query.query_camera.shape))  # (36, 7)
    # print("Target images: %s" % str(d.target.shape))  # (36, 64, 64, 3)
    # print("Context camera poses: %s" % str(d.query.context.cameras.shape))  # (36, 5, 7)
    # print("Context frames: %s" % str(d.query.context.frames.shape))  # (36, 5, 64, 64, 3)
    Qu, Ta = gqn_input_fn(DATASET_NAME, CTX_SIZE, ROOT_PATH, tf.estimator.ModeKeys.PREDICT, BATCH_SIZE)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    q, t = sess.run([Qu, Ta])
    print('q.query_camera: {}, q.context.cameras: {}, q.context.frames: {}, t: {}'.format(q.query_camera.shape, q.context.cameras.shape, q.context.frames.shape, t.shape))
