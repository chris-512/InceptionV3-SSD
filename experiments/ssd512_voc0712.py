# Train VGG_16 SSD512 on VOC data, 29.11.2017
# 1. Step: train on voc0712 trainval, without vertical flipping and rotation
# 2. Step: validate on voc07 test
# 3. Step: fine tune on voc07 person + HDA + PIROPO with vertical flipping and rotation
# 4. Step: validate on HDA + PIROPO

from trainer.trainer import TrainerParams
from evaluator.evaluator import EvaluatorParams

# -------------------------------------------------------- #
# Train VGG16-SSD512 on VOC0712 Trainval
step1_1 = TrainerParams(
    feature_extractor='vgg_16',
    model_name='ssd512',
    fine_tune_fe=False,
    train_dir='../experiments/ssd512_voc0712_29-11-2017/logs',
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
# At 18719 steps, total_loss=~4.3
# After 18719 steps, reduce batch_size to 16 because global_step fluctuates. Also, make room for evaluation.
# Original batch_size by @LevinJ is 32, so raise max_number_of_steps to 55k.
# Also reduce per_process_gpu_memory_fraction to 0.8 to allow some room for other applications and reduce OOM problems.
step1_2 = TrainerParams(
    feature_extractor='vgg_16',
    model_name='ssd512',
    fine_tune_fe=False,
    train_dir='../experiments/ssd512_voc0712_29-11-2017/logs',
    checkpoint_path='../checkpoints/vgg_16.ckpt',
    ignore_missing_vars=False,
    learning_rate=0.1,
    learning_rate_decay_type='fixed',
    learning_rate_decay_factor=1,
    num_epochs_per_decay=1,
    end_learning_rate=0.1,
    max_number_of_steps=55000,
    optimizer='adam',
    weight_decay=0.0005,
    batch_size=16,
    log_every_n_steps=100,
    save_interval_secs=60*60,
    save_summaries_secs=30,
    labels_offset=0,
    matched_thresholds=0.5
    )
# At 25406 steps, total_loss=~4.9, mAP has deteriorated too
# Change back to original batch size of 20 for training. And restart from 18719.
step1_3 = TrainerParams(
    feature_extractor='vgg_16',
    model_name='ssd512',
    fine_tune_fe=False,
    train_dir='../experiments/ssd512_voc0712_29-11-2017/logs',
    checkpoint_path='../checkpoints/vgg_16.ckpt',
    ignore_missing_vars=False,
    learning_rate=0.1,
    learning_rate_decay_type='fixed',
    learning_rate_decay_factor=1,
    num_epochs_per_decay=1,
    end_learning_rate=0.1,
    max_number_of_steps=48000,
    optimizer='adam',
    weight_decay=0.0005,
    batch_size=20,
    log_every_n_steps=100,
    save_interval_secs=60*60,
    save_summaries_secs=60,
    labels_offset=0,
    matched_thresholds=0.5
    )
# At 48k steps, total_loss=~4.2
# Fine tune from 44702 steps.
step2_1 = TrainerParams(
    feature_extractor='vgg_16',
    model_name='ssd512',
    fine_tune_fe=True,
    train_dir='../experiments/ssd512_voc0712_29-11-2017/logs/finetune',
    checkpoint_path='../experiments/ssd512_voc0712_29-11-2017/logs',
    ignore_missing_vars=False,
    learning_rate=0.01,
    learning_rate_decay_type='fixed',
    learning_rate_decay_factor=1,
    num_epochs_per_decay=1,
    end_learning_rate=0.01,
    max_number_of_steps=144000,
    optimizer='adam',
    weight_decay=0.0005,
    batch_size=10,
    log_every_n_steps=100,
    save_interval_secs=60*60,
    save_summaries_secs=60,
    labels_offset=0,
    matched_thresholds=0.5
    )
# Finished first step of fine tuning on 1:18 Sunday 03.12. Total_loss=~3.7
step2_2 = TrainerParams(
    feature_extractor='vgg_16',
    model_name='ssd512',
    fine_tune_fe=True,
    train_dir='../experiments/ssd512_voc0712_29-11-2017/logs/finetune',
    checkpoint_path='../experiments/ssd512_voc0712_29-11-2017/logs',
    ignore_missing_vars=False,
    learning_rate=0.01,
    learning_rate_decay_type='fixed',
    learning_rate_decay_factor=1,
    num_epochs_per_decay=1,
    end_learning_rate=0.001,
    max_number_of_steps=208000,
    optimizer='adam',
    weight_decay=0.0005,
    batch_size=10,
    log_every_n_steps=100,
    save_interval_secs=60*60,
    save_summaries_secs=60,
    labels_offset=0,
    matched_thresholds=0.5
    )
# Finished second step of fine tuning on 04:40 Wednesday 06.12. Total_loss=~3.7
step2_3 = TrainerParams(
    feature_extractor='vgg_16',
    model_name='ssd512',
    fine_tune_fe=True,
    train_dir='../experiments/ssd512_voc0712_29-11-2017/logs/finetune',
    checkpoint_path='../experiments/ssd512_voc0712_29-11-2017/logs',
    ignore_missing_vars=False,
    learning_rate=0.01,
    learning_rate_decay_type='fixed',
    learning_rate_decay_factor=1,
    num_epochs_per_decay=1,
    end_learning_rate=0.0005,
    max_number_of_steps=216000,
    optimizer='adam',
    weight_decay=0.0005,
    batch_size=10,
    log_every_n_steps=100,
    save_interval_secs=60*60,
    save_summaries_secs=60,
    labels_offset=0,
    matched_thresholds=0.5
    )
# start from 184587 steps for the last fine tune stage. Finished at 22:00 on 6.12 Wednesday. Total_loss=~3.5
# -------------------------------------------------------- #
# Evaluate when not training
eval1 = EvaluatorParams(
    checkpoint_path='/home/yjin/SSD/experiments/ssd512_voc0712_29-11-2017/logs/',
    use_finetune=False,
    is_training=False,
    eval_train_dataset=False,
    loop=False,
    which_checkpoint=None
    )
# Evaluate while training
eval2 = EvaluatorParams(
    checkpoint_path='/home/yjin/SSD/experiments/ssd512_voc0712_29-11-2017/logs/',
    use_finetune=False,
    is_training=True,
    eval_train_dataset=False,
    loop=True,
    which_checkpoint=None
    )
# Evluate when not training, after fine tune
eval3 = EvaluatorParams(
    checkpoint_path='/home/yjin/SSD/experiments/ssd512_voc0712_29-11-2017/logs/',
    use_finetune=True,
    is_training=False,
    eval_train_dataset=False,
    loop=False,
    which_checkpoint=None
    )
# Evluate when not training, after fine tune, on train set
eval4 = EvaluatorParams(
    checkpoint_path='/home/yjin/SSD/experiments/ssd512_voc0712_29-11-2017/logs/',
    use_finetune=True,
    is_training=False,
    eval_train_dataset=True,
    loop=False,
    which_checkpoint=None
    )
# At 18719 steps, mAP = 0.464 on VOC07test.
# At 22136 steps, mAP = 0.457.
# At 25406 steps, mAP = 0.458.
# At 44702 steps, mAP = 0.519.
# At 48000 steps, mAP = 0.518.
# At 50050 steps, mAP = 0.449. Fine tuned from step 44702.
# At 124k steps, mAP = 0.564. Although loss looks lower, mAP is not as good as 144k steps.
# At 144k steps, mAP = 0.571. Train mAP = 0.629
# At 176k steps, mAP = 0.574. Train mAP = 0.637
# At 184k steps, mAP = 0.579. Train mAP = 0.649
# At 198k steps, mAP = 0.564
# At 208k steps, mAP = 0.574.
# At 216k steps (starting from 184k steps), mAP = 0.570. Train mAP = 0.646
