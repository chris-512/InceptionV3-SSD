from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import timeline
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as tf_saver
from collections import namedtuple

from ssd import ssdmodel

# -------------------------------------------------------- #
# Definition of the parameter matrix
TrainerParams = namedtuple(
    'TrainerParameters',
    ['feature_extractor',
     'model_name',
     'fine_tune_fe',  # (True / False) whether feature extractor should be fine tuned
     'train_dir',  # directory to save model weights
     'checkpoint_path',  # directory of the initial / pre trained weights
     'ignore_missing_vars',  # (True / False) whether to ignore layers in the pre trained weights
     'learning_rate',  # initial learning rate
     'learning_rate_decay_type',
     'learning_rate_decay_factor',
     'num_epochs_per_decay',
     'end_learning_rate',
     'max_number_of_steps',  # maximal number of training steps
     'optimizer',
     'weight_decay',
     'batch_size',
     'log_every_n_steps',
     'save_interval_secs',
     'save_summaries_secs',
     'labels_offset',
     'matched_thresholds'
     ])
# -------------------------------------------------------- #
# tmp, serves as example
tmp_params = TrainerParams(
    feature_extractor='vgg_16',
    model_name='ssd512',
    fine_tune_fe=False,
    train_dir='../tmp/logs',
    checkpoint_path='../checkpoints/vgg_16.ckpt',
    ignore_missing_vars=False,
    learning_rate=0.1,
    learning_rate_decay_type='fixed',
    learning_rate_decay_factor=1,
    num_epochs_per_decay=1,
    end_learning_rate=0.1,
    max_number_of_steps=30000,
    optimizer='adam',
    weight_decay=0.0005,
    batch_size=20,
    log_every_n_steps=100,
    save_interval_secs=60*60,
    save_summaries_secs=30,
    labels_offset=0,
    matched_thresholds=0.5
    )

# -------------------------------------------------------- #
# Definition of class Trainer
class Trainer:

    # ============================= PUBLIC METHODS ============================== #
    def __init__(self, ssd_model, data_preparer, data_postprocessor, params):
        self.adadelta_rho = 0.95
        self.opt_epsilon = 1.0
        self.adagrad_initial_accumulator_value = 0.1
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.ftrl_learning_rate_power = -0.5
        self.ftrl_initial_accumulator_value = 0.1
        self.ftrl_l1 = 0.0
        self.ftrl_l2 = 0.0
        self.momentum = 0.9
        self.rmsprop_decay = 0.9
        self.rmsprop_momentum = 0.9
        self.label_smoothing = 0

        self.g_prepare = data_preparer
        self.g_post = data_postprocessor
        self.g_ssd = ssd_model

        self.fine_tune_fe = params.fine_tune_fe
        self.train_dir = params.train_dir
        self.checkpoint_path = params.checkpoint_path
        self.ignore_missing_vars = params.ignore_missing_vars
        self.learning_rate = params.learning_rate
        self.learning_rate_decay_type = params.learning_rate_decay_type
        self.learning_rate_decay_factor = params.learning_rate_decay_factor
        self.num_epochs_per_decay = params.num_epochs_per_decay
        self.end_learning_rate = params.end_learning_rate
        self.max_number_of_steps = params.max_number_of_steps
        self.optimizer = params.optimizer
        self.weight_decay = params.weight_decay
        self.batch_size = params.batch_size
        self.log_every_n_steps = params.log_every_n_steps
        self.save_interval_secs = params.save_interval_secs
        self.save_summaries_secs = params.save_summaries_secs

        if self.fine_tune_fe is False:
            self.checkpoint_exclude_scopes = '{},{}'.format(self.g_ssd.params.model_name, 'bbox_layers')
            self.trainable_scopes = self.checkpoint_exclude_scopes
        elif self.fine_tune_fe is True:
            self.checkpoint_exclude_scopes = None
            self.trainable_scopes = '{},{}'.format(self.g_ssd.params.model_name, 'bbox_layers')
        else:
            raise ValueError('Wrong definition of fine_tune_fe!')

    def start_training(self):
        tf.logging.set_verbosity(tf.logging.DEBUG)

        # Get batched training data
        image, filename, glabels, gbboxes, gdifficulties, gclasses, localizations, gscores = \
            self.g_prepare.get_voc_2007_2012_train_data()
        # Get model outputs
        predictions, localisations, logits, end_points = self.g_ssd.get_model(image)
        # Get model training loss
        total_loss = ssdmodel.get_losses(logits, localisations, gclasses, localizations, gscores)

        global_step = slim.get_or_create_global_step()
        variables_to_train = self._get_variables_to_train()
        print(variables_to_train)
        learning_rate = self._configure_learning_rate(self.g_prepare.dataset.num_samples, global_step)
        optimizer = self._configure_optimizer(learning_rate)

		# Create the train_op and clip the gradient norms:
        train_op = slim.learning.create_train_op(total_loss, optimizer, variables_to_train=variables_to_train, clip_gradient_norm=4)
        self._add_summaries(end_points, total_loss)
        tf.summary.scalar('learning_rate', learning_rate)
        self._setup_debugging(predictions, localizations, glabels, gbboxes, gdifficulties)

		
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)

        slim.learning.train(
            train_op,
            self.train_dir,
            train_step_fn=self._train_step,
            saver=tf_saver.Saver(max_to_keep=500),
            init_fn=self._get_init_fn(),
            number_of_steps=self.max_number_of_steps,
            log_every_n_steps=self.log_every_n_steps,
            save_summaries_secs=self.save_summaries_secs,
            save_interval_secs=self.save_interval_secs,
            session_config=config
        )

    # =========================== PRIVATE METHODS ============================ #
    def _train_step(self, sess, train_op, global_step, train_step_kwargs):
        """Function that takes a gradient step and specifies whether to stop.

        Args:
            sess: The current session.
            train_op: An `Operation` that evaluates the gradients and returns the
                total loss.
            global_step: A `Tensor` representing the global training step.
            train_step_kwargs: A dictionary of keyword arguments.

        Returns:
            The total loss and a boolean indicating whether or not to stop training.

        Raises:
            ValueError: if 'should_trace' is in `train_step_kwargs` but `logdir` is not.
        """
        start_time = time.time()
        trace_run_options = None
        run_metadata = None
        if 'should_trace' in train_step_kwargs:
            if 'logdir' not in train_step_kwargs:
                raise ValueError('logdir must be present in train_step_kwargs when ''should_trace is present')
            if sess.run(train_step_kwargs['should_trace']):
                trace_run_options = config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE)
                run_metadata = config_pb2.RunMetadata()
        total_loss, np_global_step = sess.run([train_op, global_step],
                                              options=trace_run_options,
                                              run_metadata=run_metadata)
        time_elapsed = time.time() - start_time
        if run_metadata is not None:
            tl = timeline.Timeline(run_metadata.step_stats)
            trace = tl.generate_chrome_trace_format()
            trace_filename = os.path.join(train_step_kwargs['logdir'], 'tf_trace-%d.json' % np_global_step)
            logging.info('Writing trace to %s', trace_filename)
            file_io.write_string_to_file(trace_filename, trace)
            if 'summary_writer' in train_step_kwargs:
                train_step_kwargs['summary_writer'].add_run_metadata(run_metadata, 'run_metadata-%d' % np_global_step)
        if 'should_log' in train_step_kwargs:
            if sess.run(train_step_kwargs['should_log']):
                logging.info('global step %d: loss = %.4f (%.2f sec/step)', np_global_step, total_loss, time_elapsed)
        if 'should_stop' in train_step_kwargs:
            should_stop = sess.run(train_step_kwargs['should_stop'])
        else:
            should_stop = False
        return total_loss, should_stop

    def _get_variables_to_train(self):
        """Returns a list of variables to train.

        Returns:
            A list of variables to train by the optimizer.
        """
        if self.trainable_scopes is None:
            return tf.trainable_variables()
        else:
            scopes = [scope.strip() for scope in self.trainable_scopes.split(',')]
		
        # added trainable scopes
        #additional_scopes = ['Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'MaxPool_5a_3x3', 'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d']
        additional_scopes = ['Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d']
        additional_scopes = ['InceptionV3/' + scope for scope in additional_scopes]
        scopes.extend(additional_scopes)

		# exclude 

        variables_to_train = []
        for scope in scopes:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            variables_to_train.extend(variables)
        return variables_to_train

    def _configure_learning_rate(self, num_samples_per_epoch, global_step):
        """Configures the learning rate.

        Args:
            num_samples_per_epoch: The number of samples in each epoch of training.
            global_step: The global_step tensor.

        Returns:
            A `Tensor` representing the learning rate.

        Raises:
            ValueError: if
        """
        decay_steps = int(num_samples_per_epoch / self.batch_size *
                          self.num_epochs_per_decay)
        if self.learning_rate_decay_type == 'exponential':
            return tf.train.exponential_decay(self.learning_rate,
                                              global_step,
                                              decay_steps,
                                              self.learning_rate_decay_factor,
                                              staircase=True,
                                              name='exponential_decay_learning_rate')
        elif self.learning_rate_decay_type == 'fixed':
            return tf.constant(self.learning_rate, name='fixed_learning_rate')
        elif self.learning_rate_decay_type == 'polynomial':
            return tf.train.polynomial_decay(self.learning_rate,
                                             global_step,
                                             decay_steps,
                                             self.end_learning_rate,
                                             power=1.0,
                                             cycle=False,
                                             name='polynomial_decay_learning_rate')
        else:
            raise ValueError('learning_rate_decay_type [%s] was not recognized',
                             self.learning_rate_decay_type)

    def _configure_optimizer(self, learning_rate):
        """Configures the optimizer used for training.

        Args:
            learning_rate: A scalar or `Tensor` learning rate.

        Returns:
            An instance of an optimizer.

        Raises:
            ValueError: if FLAGS.optimizer is not recognized.
        """
        if self.optimizer == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(
                learning_rate,
                rho=self.adadelta_rho,
                epsilon=self.opt_epsilon)
        elif self.optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(
                learning_rate,
                initial_accumulator_value=self.adagrad_initial_accumulator_value)
        elif self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(
                learning_rate,
                beta1=self.adam_beta1,
                beta2=self.adam_beta2,
                epsilon=self.opt_epsilon)
        elif self.optimizer == 'ftrl':
            optimizer = tf.train.FtrlOptimizer(
                learning_rate,
                learning_rate_power=self.ftrl_learning_rate_power,
                initial_accumulator_value=self.ftrl_initial_accumulator_value,
                l1_regularization_strength=self.ftrl_l1,
                l2_regularization_strength=self.ftrl_l2)
        elif self.optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(
                learning_rate,
                momentum=self.momentum,
                name='Momentum')
        elif self.optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(
                learning_rate,
                decay=self.rmsprop_decay,
                momentum=self.rmsprop_momentum,
                epsilon=self.opt_epsilon)
        elif self.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        else:
            raise ValueError('Optimizer [%s] was not recognized', self.optimizer)
        return optimizer

    def _add_summaries(self, end_points, total_loss):
        # Add summaries for end_points (activations).
        for end_point in end_points:
            x = end_points[end_point]
            tf.summary.histogram('activations/' + end_point, x)
            tf.summary.scalar('sparsity/' + end_point, tf.nn.zero_fraction(x))
        # Add summaries for losses and extra losses.
        tf.summary.scalar('total_loss', total_loss)
        for loss in tf.get_collection('EXTRA_LOSSES'):
            tf.summary.scalar(loss.op.name, loss)
        # Add summaries for variables.
        for variable in slim.get_model_variables():
            tf.summary.histogram(variable.op.name, variable)

    def _setup_debugging(self, predictions, localizations, glabels, gbboxes, gdifficults):
        _, self.mAP_12_op_train = \
            self.g_post.get_mAP_tf_current_batch(predictions, localizations, glabels, gbboxes, gdifficults)
        return None

    def _get_init_fn(self):
        """Returns a function run by the chief worker to warm-start the training.

        Note that the init_fn is only run when initializing the model during the very
        first global step.

        Returns:
            An init function run by the supervisor.
        """
        if self.checkpoint_path is None:
            return None
        # Warn the user if a checkpoint exists in the train_dir. Then we'll be ignoring the checkpoint anyway.
        if tf.train.latest_checkpoint(self.train_dir):
            tf.logging.info(
                'Ignoring --checkpoint_path because a checkpoint already exists in %s'
                % self.train_dir)
            return None

        exclusions = []
        if self.checkpoint_exclude_scopes:
            exclusions = [scope.strip()
                          for scope in self.checkpoint_exclude_scopes.split(',')]

        variables_to_restore = []
        all_variables = slim.get_model_variables()
        if self.fine_tune_fe:
            global_step = slim.get_or_create_global_step()
            all_variables.append(global_step)
        for var in all_variables:
            excluded = False

            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)

        if tf.gfile.IsDirectory(self.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_path)
        else:
            checkpoint_path = self.checkpoint_path

        tf.logging.info('Fine-tuning from %s' % checkpoint_path)

        return slim.assign_from_checkpoint_fn(
            checkpoint_path,
            variables_to_restore,
            ignore_missing_vars=self.ignore_missing_vars)
