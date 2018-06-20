import time
import os
import re
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from collections import namedtuple

# -------------------------------------------------------- #
# Definition of the parameter matrix
EvaluatorParams = namedtuple(
    'EvaluatorParameters',
    ['checkpoint_path',  # directory must have '/' at the end
     'eval_dir',  # directory to save evaluation results
     'use_finetune',  # whether use checkpoints under 'finetune/' folder
     'is_training',  # whether evaluate while training is ongoing
     'eval_train_dataset',  # whether evaluate against training dataset
     'loop',  # whether evaluate in loops
     'which_checkpoint'  # specify a checkpoint to evaluate
     ])
# -------------------------------------------------------- #
# example evaluation parameters
eval_only_last_ckpt = EvaluatorParams(
    checkpoint_path='./logs/',
    eval_dir='./logs/eval',
    use_finetune=False,
    is_training=False,
    eval_train_dataset=False,
    loop=False,
    which_checkpoint='last'
)

def flatten(x):
    result = []
    for el in x:
        if isinstance(el, tuple):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

# -------------------------------------------------------- #
# Definition of class Evaluator
class Evaluator:
    def __init__(self, ssd_model, data_preparer, data_postprocessor, params):
        self.checkpoint_path = params.checkpoint_path
        if params.use_finetune:
            self.checkpoint_path = self.checkpoint_path + 'finetune/'
        self.eval_dir = params.eval_dir
        self.is_training = params.is_training
        self.eval_train_dataset = params.eval_train_dataset
        self.loop = params.loop
        self.which_checkpoint = params.which_checkpoint

        self.g_ssd = ssd_model
        self.g_prepare = data_preparer
        self.g_post = data_postprocessor

    def start_evaluation(self):
        if self.is_training:
            with tf.device('/device:CPU:0'):
                self._setup_evaluation()
        else:
            with tf.device('/device:CPU:0'):
                self._setup_evaluation()

    def eval_all_checkpoints(self, min_step, step):
        selected_checkpoints = self._get_all_checkpoints(min_step, step)
        for ckpt in selected_checkpoints:
            ckpt_file = self.checkpoint_path + 'model.ckpt' + str(ckpt)
            if self.eval_train_dataset:
                data = 'train'
            else:
                data = 'test'
            print('checkpoint {}, {} data'.format(ckpt_file, data))
            self.start_evaluation()

    def _setup_evaluation(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        _ = slim.get_or_create_global_step()

        if self.is_training:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01)
        else:
            #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
            gpu_options = tf.GPUOptions(allow_growth=True)

        if self.eval_train_dataset:
            image, _, glabels, gbboxes, gdifficults, _, _, _ = \
                self.g_prepare.get_voc_2007_2012_train_data(is_training_data=False)
        else:
            image, _, glabels, gbboxes, gdifficults, _, _, _ = self.g_prepare.get_voc_2007_test_data()

        # get model outputs
        predictions, localisations, logits, end_points = self.g_ssd.get_model(image)
        names_to_updates = \
            self.g_post.get_mAP_tf_accumulative(predictions, localisations, glabels, gbboxes, gdifficults)
        variables_to_restore = slim.get_variables_to_restore()
        num_batches = math.ceil(self.g_prepare.dataset.num_samples / float(self.g_prepare.batch_size))
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)

        if not self.loop:
            # standard evaluation procedure
            print('one time evaluation...')
            if tf.gfile.IsDirectory(self.checkpoint_path):
                checkpoint_file = tf.train.latest_checkpoint(self.checkpoint_path)
            else:
                checkpoint_file = self.checkpoint_path
            # tf.logging.INFO('Evaluating %s' % checkpoint_file)
            start = time.time()
            slim.evaluation.evaluate_once(master='',
                                          checkpoint_path=checkpoint_file,
                                          logdir=self.eval_dir,
                                          num_evals=num_batches,
                                          eval_op=flatten(list(names_to_updates.values())),
                                          session_config=config,
                                          variables_to_restore=variables_to_restore)
	        # log time spent
            end = time.time()
            elapsed = end - start
            print('Time spent : %.3f seconds' % elapsed)
            print('Time spent per batch : %.3f seconds' % (elapsed / num_batches))
        else:
            print('Evaluate during training...')
            # waiting loop
            slim.evaluation.evaluation_loop(master='',
                                            checkpoint_dir=self.checkpoint_path,
                                            logdir=self.eval_dir,
                                            num_evals=50, # num_baches
                                            eval_op=list(names_to_updates.values()),
                                            variables_to_restore=variables_to_restore,
                                            eval_interval_secs=60,
                                            session_config=config,
                                            max_number_of_evaluations=np.inf,
                                            timeout=None)

    def _get_all_checkpoints(self, min_step, step):
        with open(self.checkpoint_path + 'checkpoint') as f:
            content = f.readline()
        content = [x.strip() for x in content]
        checkpoints = []
        for line in content:
            m = re.search('all_model_checkpoint_paths: "model.ckpt-(.*)"', line)
            if m:
                num = m.group(1)
                checkpoints.append(num)
        last_step = min_step
        selected_checkpoints = []
        for ckpt in checkpoints:
            ckpt = int(ckpt)
            if ckpt < min_step:
                continue
            if ckpt == int(checkpoints[-1]):
                # the last checkpoint always get selected
                selected_checkpoints.append(ckpt)
                continue
            if ckpt >= last_step:
                selected_checkpoints.append(ckpt)
                last_step = last_step + step
        if self.which_checkpoint == 'last':
            selected_checkpoints = [selected_checkpoints[-1]]
        return selected_checkpoints
