from evaluator.evaluator import EvaluatorParams
from trainer.trainer import TrainerParams

# -------------------------------------------------------- #
# Training parameters for MobileNet-SSD512
train1_1 = TrainerParams(
    feature_extractor='mobilenet_v1',
    model_name='ssd512',
    fine_tune_fe=False,
    train_dir='../experiments/mobilenet_11-12-2017/logs',
    checkpoint_path='../checkpoints/mobilenet/mobilenet_v1_1.0_224.ckpt',
    ignore_missing_vars=False,
    learning_rate=0.1,
    learning_rate_decay_type='fixed',
    learning_rate_decay_factor=1,
    num_epochs_per_decay=1,
    end_learning_rate=0.1,
    max_number_of_steps=60000,
    optimizer='adam',
    weight_decay=0.0005,
    batch_size=16,
    log_every_n_steps=20,
    save_interval_secs=60*60,
    save_summaries_secs=60,
    labels_offset=0,
    matched_thresholds=0.5
    )
# At 30k steps, loss ~= 6.0, train mAP = 0.347, test mAP = 0.353
# At 60k steps, loss ~= 5.7, train mAP = 0.422, test mAP = 0.413
train1_2 = TrainerParams(
    feature_extractor='mobilenet_v1',
    model_name='ssd512',
    fine_tune_fe=False,
    train_dir='../experiments/mobilenet_11-12-2017/logs',
    checkpoint_path='../experiments/mobilenet_11-12-2017/logs',
    ignore_missing_vars=False,
    learning_rate=0.01,
    learning_rate_decay_type='fixed',
    learning_rate_decay_factor=1,
    num_epochs_per_decay=1,
    end_learning_rate=0.01,
    max_number_of_steps=90000,
    optimizer='adam',
    weight_decay=0.0005,
    batch_size=16,
    log_every_n_steps=20,
    save_interval_secs=60*60,
    save_summaries_secs=60,
    labels_offset=0,
    matched_thresholds=0.5
    )
# At 79.5k steps, loss ~= 5.4, train mAP = 0.456, test mAP = 0.436
# At 90k steps, loss ~= 5.5, train mAP = 0.455, test mAP = 0.438
# It seems this has reached the stop point. I will turn on finetuning for the next steps.
train2_1 = TrainerParams(
    feature_extractor='mobilenet_v1',
    model_name='ssd512',
    fine_tune_fe=True,
    train_dir='../experiments/mobilenet_11-12-2017/logs/finetune',
    checkpoint_path='../experiments/mobilenet_11-12-2017/logs',
    ignore_missing_vars=False,
    learning_rate=0.01,
    learning_rate_decay_type='fixed',
    learning_rate_decay_factor=0.75,
    num_epochs_per_decay=2,
    end_learning_rate=0.0005,
    max_number_of_steps=95338,
    optimizer='adam',
    weight_decay=0.0005,
    batch_size=8,
    log_every_n_steps=20,
    save_interval_secs=60*60,
    save_summaries_secs=60,
    labels_offset=0,
    matched_thresholds=0.5
    )
# At 95339 steps, test mAP = 0.436.
train2_2 = TrainerParams(
    feature_extractor='mobilenet_v1',
    model_name='ssd512',
    fine_tune_fe=True,
    train_dir='../experiments/mobilenet_11-12-2017/logs/finetune',
    checkpoint_path='../experiments/mobilenet_11-12-2017/logs',
    ignore_missing_vars=False,
    learning_rate=0.002,
    learning_rate_decay_type='fixed',
    learning_rate_decay_factor=0.75,
    num_epochs_per_decay=2,
    end_learning_rate=0.0005,
    max_number_of_steps=106000,
    optimizer='adam',
    weight_decay=0.0005,
    batch_size=8,
    log_every_n_steps=20,
    save_interval_secs=60*60,
    save_summaries_secs=60,
    labels_offset=0,
    matched_thresholds=0.5
    )
# Tried a few times, loss explodes at about 100k steps. No idea what happend. I set save interval to 2 hours. So no
# weights could be saved from this period.
# At 106k steps, test mAP = 0.440. Training broke at 107k steps again.
train2_3 = TrainerParams(
    feature_extractor='mobilenet_v1',
    model_name='ssd512',
    fine_tune_fe=True,
    train_dir='../experiments/mobilenet_11-12-2017/logs/finetune',
    checkpoint_path='../experiments/mobilenet_11-12-2017/logs',
    ignore_missing_vars=False,
    learning_rate=0.001,
    learning_rate_decay_type='fixed',
    learning_rate_decay_factor=0.75,
    num_epochs_per_decay=2,
    end_learning_rate=0.0005,
    max_number_of_steps=120000,
    optimizer='sgd',
    weight_decay=0.0005,
    batch_size=8,
    log_every_n_steps=20,
    save_interval_secs=60*60,
    save_summaries_secs=60,
    labels_offset=0,
    matched_thresholds=0.5
    )
# Finally got training run beyond 107k steps with this setting. According to papers sgd with small lr is better than adam.
# At 120k steps, test mAP = 0.441.
train2_4 = TrainerParams(
    feature_extractor='mobilenet_v1',
    model_name='ssd512',
    fine_tune_fe=True,
    train_dir='../experiments/mobilenet_11-12-2017/logs/finetune',
    checkpoint_path='../experiments/mobilenet_11-12-2017/logs',
    ignore_missing_vars=False,
    learning_rate=0.0002,
    learning_rate_decay_type='fixed',
    learning_rate_decay_factor=0.75,
    num_epochs_per_decay=2,
    end_learning_rate=0.0005,
    max_number_of_steps=150000,
    optimizer='sgd',
    weight_decay=0.0005,
    batch_size=8,
    log_every_n_steps=20,
    save_interval_secs=60*60,
    save_summaries_secs=60,
    labels_offset=0,
    matched_thresholds=0.5
    )
# Last setting still breaks up. Use this one instead.
# At 125k steps, test mAP = 0.440
# At this point I am positively sure that this model cannot be trained to a better performance.

# -------------------------------------------------------- #
# Parameters for evaluating model
eval_train = EvaluatorParams(
    checkpoint_path='../experiments/mobilenet_11-12-2017/logs/',
    eval_dir='../experiments/mobilenet_11-12-2017/eval_train',
    use_finetune=False,
    is_training=False,
    eval_train_dataset=True,
    loop=False,
    which_checkpoint='last'
)

eval_test = EvaluatorParams(
    checkpoint_path='../experiments/mobilenet_11-12-2017/logs/',
    eval_dir='../experiments/mobilenet_11-12-2017/eval_test',
    use_finetune=False,
    is_training=False,
    eval_train_dataset=False,
    loop=False,
    which_checkpoint='last'
)

eval_test_f = EvaluatorParams(
    checkpoint_path='../experiments/mobilenet_11-12-2017/logs/',
    eval_dir='../experiments/mobilenet_11-12-2017/eval_test',
    use_finetune=True,
    is_training=False,
    eval_train_dataset=False,
    loop=False,
    which_checkpoint='last'
)
