import os
import matplotlib.pyplot as plt
import tensorflow as tf
from config import CONFIG as C
# from data_reader import gqn_input_fn
# from test_data_reader import input_fn as gqn_input_fn
from data_reader import gqn_input_fn2 as gqn_input_fn
from model import gqn_draw_model_fn

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only run on GPU 0


gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.9)
sess_config = tf.ConfigProto(gpu_options=gpu_options)
run_config = tf.estimator.RunConfig(session_config=sess_config, save_checkpoints_steps=C.CKPT_STEPS)
classifier = tf.estimator.Estimator(model_fn=gqn_draw_model_fn, model_dir=C.MODEL_DIR, config=run_config, params=C)

# input_fn = lambda: gqn_input_fn(dataset=C.DATASET, context_size=C.CONTEXT_SIZE, root=C.DATA_DIR, mode=tf.estimator.ModeKeys.PREDICT)
input_fn = lambda: gqn_input_fn(dataset=C.DATASET, context_size=9, root=C.DATA_DIR, mode=tf.estimator.ModeKeys.PREDICT)

pred = classifier.predict(input_fn=input_fn)

next_pred = next(pred)
pred_img = next_pred['prediction_mean']
frames = next_pred['frames']
poses = next_pred['poses']
query = next_pred['query']
print(poses)
print(query)
fig = plt.figure()
for i in range(1, 6):
    fig.add_subplot(2, 5, i)
    plt.imshow(frames[i-1])
    plt.title('Input_{}'.format(i))
    plt.axis('off')
fig.add_subplot(2, 2, 4)
plt.imshow((pred_img - pred_img.min())/(pred_img.max() - pred_img.min()))
plt.title('Predicted')
plt.axis('off')
plt.savefig('roaming/{}.png'.format(C.N), dpi=200)
# plt.savefig('roaming/1.png', dpi=200)
# 1, 3, 4, 0, 2
