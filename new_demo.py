import os
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf

# slim = tf.contrib.slim

sys.path.append('../')

from pathlib import Path

from preprocessing import ssd_vgg_preprocessing
from utils import visualization
from ssd.ssdmodel import SSDModel
import utils.np_methods as np_methods

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

# Input placeholder.
net_shape = (512, 512)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
g_ssd_model = SSDModel('inception_v3', 'ssd512', weight_decay=0.0005)
predictions, localisations, _, _ = g_ssd_model.get_model(image_4d)

# Restore SSD model.
ckpt_filename = tf.train.latest_checkpoint('./experiments/inception_v3/logs/finetune_14008')
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = g_ssd_model.get_anchors_all_layers()


# Main image processing routine.
# def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(512, 512)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = \
        isess.run([image_4d, predictions, localisations, bbox_img], feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = \
        np_methods.ssd_bboxes_select(rpredictions, rlocalisations, ssd_anchors,
                                     select_threshold=select_threshold,
                                     img_shape=net_shape, num_classes=21, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes


# Test on some demo image and visualize output.
data_path = '/home/siyeong/test/modular_SSD_tensorflow/data/'

for image_path in os.listdir(data_path):
	img = mpimg.imread(data_path + image_path)
	rclasses, rscores, rbboxes = process_image(img)
	# visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
	visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
	plt.show()
