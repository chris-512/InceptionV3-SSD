from evaluator.evaluator import EvaluatorParams
from trainer.trainer import TrainerParams

# Training parameters for InceptionV3-SSD512
train1_1 = TrainerParams(
    feature_extractor='inception_v3',
    model_name='ssd512',
    fine_tune_fe=False,
    train_dir='./experiments/inception_v3/logs/inception_freezed_4_layers',
    checkpoint_path='./checkpoints/inception_v3/inception_v3.ckpt',
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
    log_every_n_steps=30,
    save_interval_secs=1800,
    save_summaries_secs=120,
    labels_offset=0,
    matched_thresholds=0.5
    )

train1_2 = TrainerParams(
    feature_extractor='inception_v3',
    model_name='ssd512',
    fine_tune_fe=True,
    train_dir='./experiments/inception_v3/logs/finetune_14008',
    checkpoint_path='./experiments/inception_v3/logs/model.ckpt-14008',
    ignore_missing_vars=False,
    learning_rate=0.01,
    learning_rate_decay_type='fixed',
    learning_rate_decay_factor=1,
    num_epochs_per_decay=1,
    end_learning_rate=0.01,
    max_number_of_steps=60000,
    optimizer='adam',
    weight_decay=0.0005,
    batch_size=16,
    log_every_n_steps=30,
    save_interval_secs=60,
    save_summaries_secs=60,
    labels_offset=0,
    matched_thresholds=0.5
    )
train1_3 = TrainerParams(
    feature_extractor='inception_v3',
    model_name='ssd512',
    fine_tune_fe=True,
    train_dir='./experiments/inception_v3/logs/finetune_14008',
    checkpoint_path='./experiments/inception_v3/logs/finetune_14008',
    ignore_missing_vars=False,
    learning_rate=0.01,
    learning_rate_decay_type='fixed',
    learning_rate_decay_factor=1,
    num_epochs_per_decay=1,
    end_learning_rate=0.01,
    max_number_of_steps=60000,
    optimizer='adam',
    weight_decay=0.0005,
    batch_size=16,
    log_every_n_steps=30,
    save_interval_secs=60*30,
    save_summaries_secs=60,
    labels_offset=0,
    matched_thresholds=0.5
    )
train1_4 = TrainerParams(
    feature_extractor='inception_v3',
    model_name='ssd512',
    fine_tune_fe=True,
    train_dir='./experiments/inception_v3/logs/finetune_14008',
    checkpoint_path='./experiments/inception_v3/logs/finetune_14008',
    ignore_missing_vars=False,
    learning_rate=0.001,
    learning_rate_decay_type='fixed',
    learning_rate_decay_factor=1,
    num_epochs_per_decay=1,
    end_learning_rate=0.001,
    max_number_of_steps=60000,
    optimizer='adam',
    weight_decay=0.0005,
    batch_size=16,
    log_every_n_steps=30,
    save_interval_secs=60*30,
    save_summaries_secs=60,
    labels_offset=0,
    matched_thresholds=0.5
    )
train1_4 = TrainerParams(
    feature_extractor='inception_v3',
    model_name='ssd512',
    fine_tune_fe=True,
    train_dir='./experiments/inception_v3/logs/finetune_14008',
    checkpoint_path='./experiments/inception_v3/logs/finetune_14008',
    ignore_missing_vars=False,
    learning_rate=0.001,
    learning_rate_decay_type='fixed',
    learning_rate_decay_factor=1,
    num_epochs_per_decay=1,
    end_learning_rate=0.001,
    max_number_of_steps=200000,
    optimizer='adam',
    weight_decay=0.0005,
    batch_size=32,
    log_every_n_steps=30,
    save_interval_secs=60*10,
    save_summaries_secs=60,
    labels_offset=0,
    matched_thresholds=0.5
    )


# Parameters for evaluating model
eval_train = EvaluatorParams(
    checkpoint_path='../experiments/mobilenet/logs/',
    eval_dir='../experiments/mobilenet/eval_train',
    use_finetune=False,
    is_training=False,
    eval_train_dataset=True,
    loop=False,
    which_checkpoint='last'
)

eval_test = EvaluatorParams(
    checkpoint_path='./experiments/inception_v3/logs/inception_freezed_4_layers',
    eval_dir='./experiments/inception_v3/eval_test_freezed_4_layers',
    use_finetune=False,
    is_training=False,
    eval_train_dataset=False,
    loop=True,
    which_checkpoint='last'
)

eval_test_f = EvaluatorParams(
    checkpoint_path='./experiments/inception_v3/logs/',
    eval_dir='./experiments/inception_v3/eval_test',
    use_finetune=True,
    is_training=False,
    eval_train_dataset=False,
    loop=False,
    which_checkpoint='last'
)
