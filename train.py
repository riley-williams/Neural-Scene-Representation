import os
import tensorflow as tf
from config import CONFIG as C
from data_reader import gqn_input_fn
from model import gqn_draw_model_fn

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only run on GPU 1


def main():
    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.9)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    run_config = tf.estimator.RunConfig(session_config=sess_config, save_checkpoints_steps=C.CKPT_STEPS)
    classifier = tf.estimator.Estimator(model_fn=gqn_draw_model_fn, model_dir=C.MODEL_DIR, config=run_config, params=C)

    tensors_to_log = {'l2_reconstruction': 'l2_reconstruction'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=C.LOG_STEPS)

    if C.INITAIL_EVAL:
        eval_input = lambda: gqn_input_fn(dataset=C.DATASET, context_size=C.CONTEXT_SIZE, root=C.DATA_DIR, mode=tf.estimator.ModeKeys.EVAL, batch_size=C.BATCH_SIZE, num_threads=C.QUEUE_THREAD, buffer_size=C.QUEUE_BUFFER)
        eval_results = classifier.evaluate(input_fn=eval_input, hooks=[logging_hook])

    for _ in range(C.TRAIN_EPOCHS):
        train_input = lambda: gqn_input_fn(dataset=C.DATASET, context_size=C.CONTEXT_SIZE, root=C.DATA_DIR, mode=tf.estimator.ModeKeys.TRAIN, batch_size=C.BATCH_SIZE, num_threads=C.QUEUE_THREAD, buffer_size=C.QUEUE_BUFFER)
        classifier.train(input_fn=train_input, hooks=[logging_hook])
        eval_input = lambda: gqn_input_fn(dataset=C.DATASET, context_size=C.CONTEXT_SIZE, root=C.DATA_DIR, mode=tf.estimator.ModeKeys.EVAL, batch_size=C.BATCH_SIZE, num_threads=C.QUEUE_THREAD, buffer_size=C.QUEUE_BUFFER)
        eval_results = classifier.evaluate(input_fn=eval_input, hooks=[logging_hook])


if __name__ == '__main__':
    main()
