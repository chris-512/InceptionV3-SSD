# Continue experiment step 3


from trainer.trainer import TrainerParams
from evaluator.evaluator import EvaluatorParams


# Fine tune from 184587 steps.
step3_1 = TrainerParams(
    feature_extractor='vgg_16',
    model_name='ssd512',
    fine_tune_fe=True,
    train_dir='../experiments/ssd512_voc0712_29-11-2017/logs/finetune_person',
    checkpoint_path='../experiments/ssd512_voc0712_29-11-2017/logs/finetune_person',
    ignore_missing_vars=False,
    learning_rate=0.001,
    learning_rate_decay_type='fixed',
    learning_rate_decay_factor=1,
    num_epochs_per_decay=1,
    end_learning_rate=0.01,
    max_number_of_steps=300000,
    optimizer='adam',
    weight_decay=0.0005,
    batch_size=10,
    log_every_n_steps=100,
    save_interval_secs=60*60,
    save_summaries_secs=60,
    labels_offset=0,
    matched_thresholds=0.5
    )
# Training stopped at 203192 steps because it ran out of space on my ssd.
# Believing that the evaluation number is somewhat off, I ran the demo and the result is actually already usable.
# In the data I have that is taken in the experiment apartment downstairs the network recognizes the person most of the
# time, despite some false positives.
# I looked into the evaluation code and found out that the standard used is somewhat too harsh for the result.
# The detections have confidence in the range 35%~100%, whereas the evaluation uses 50% as the lower limit.
# Also noticeable is that in the piropo test data person on the far end cannot be recognized.


# Evaluation on HDA and PIROPO.
eval_train = EvaluatorParams(
    checkpoint_path='/home/yjin/SSD/experiments/ssd512_voc0712_29-11-2017/logs/finetune_person',
    use_finetune=False,
    is_training=False,
    eval_train_dataset=True,
    loop=False,
    which_checkpoint=None
    )

# 195129 steps, mAP = 0.0364

