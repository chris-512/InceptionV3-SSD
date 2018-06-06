import tensorflow as tf
import tensorflow.contrib.slim as slim

from datasets import pascalvoc_datasets
from preprocessing.ssd_vgg_preprocessing import preprocess_for_eval
from preprocessing.ssd_vgg_preprocessing import preprocess_for_train
from utils import tf_utils

tfrecords_dir = '/database/pascalvoc/VOCdevkit/tfrecords/'

class PrepareData:
    # ============================= PUBLIC METHODS ============================== #
    def __init__(self, ssd_model, batch_size, labels_offset, matched_thresholds):
        self.batch_size = batch_size
        self.labels_offset = labels_offset
        self.matched_thresholds = matched_thresholds
        self.g_ssd = ssd_model

    def get_voc_2007_train_data(self, is_training_data=True):
        #  data_sources = "../data/voc/tfrecords/voc_train_2007*.tfrecord"
        data_sources = tfrecords_dir + 'voc_2007_train*.tfrecord'
        num_samples = pascalvoc_datasets.DATASET_SIZE['2007_train']
        return self._get_images_labels_bboxes(data_sources, num_samples, is_training_data)

    def get_voc_2012_train_data(self,is_training_data=True):
        data_sources = tfrecords_dir + 'voc_2012_train*.tfrecord'
        num_samples = pascalvoc_datasets.DATASET_SIZE['2012_train']
        return self._get_images_labels_bboxes(data_sources, num_samples, is_training_data)

    def get_voc_2007_2012_train_data(self,is_training_data=True):
        data_sources = tfrecords_dir + 'voc_*_train*.tfrecord'
        num_samples = pascalvoc_datasets.DATASET_SIZE['2007_train'] + pascalvoc_datasets.DATASET_SIZE['2012_train']
        return self._get_images_labels_bboxes(data_sources, num_samples, is_training_data)

    def get_voc_2007_test_data(self):
        data_sources = tfrecords_dir + 'voc_2007_test*.tfrecord'
        num_samples = pascalvoc_datasets.DATASET_SIZE['2007_test']
        return self._get_images_labels_bboxes(data_sources, num_samples, False)

    # =========================== PRIVATE METHODS ============================ #
    def _get_images_labels_bboxes(self, data_sources, num_samples, is_training_data):
        self.dataset = pascalvoc_datasets.get_dataset_info(data_sources, num_samples)
        self.is_training_data = is_training_data
        if self.is_training_data:
            shuffle = True
            # make sure most samples can be fetched in one epoch
            self.num_readers = 2
        else:
            # make sure data is fetched in sequence
            shuffle = False
            self.num_readers = 1

        provider = slim.dataset_data_provider.DatasetDataProvider(
            self.dataset,
            shuffle=shuffle,
            num_readers=self.num_readers,
            common_queue_capacity=20 * self.batch_size,
            common_queue_min=10 * self.batch_size)

        # Get for SSD network: image, labels, bboxes.
        [image, shape, format, filename, glabels, gbboxes, gdifficults] = provider.get(
            ['image', 'shape', 'format', 'filename',
             'object/label',
             'object/bbox',
             'object/difficult'])
        glabels -= self.labels_offset

        # Pre-processing image, labels and bboxes.
        image, glabels, gbboxes = self._preprocess_data(image, glabels, gbboxes)

        # Assign groundtruth information for all default/anchor boxes
        gclasses, glocalisations, gscores = self.g_ssd.match_achors(glabels, gbboxes,
                                                                    matching_threshold=self.matched_thresholds)

        return self._batching_data(image, glabels, format, filename,
                                   gbboxes, gdifficults, gclasses, glocalisations, gscores)

    def _preprocess_data(self, image, labels, bboxes):
        out_shape = self.g_ssd.params.img_shape
        if self.is_training_data:
            image, labels, bboxes = preprocess_for_train(image, labels, bboxes, out_shape=out_shape)
        else:
            image, labels, bboxes, _ = preprocess_for_eval(image, labels, bboxes, out_shape=out_shape)
        return image, labels, bboxes

    def _batching_data(self, image, glabels, format, filename,
                       gbboxes, gdifficults, gclasses, glocalisations, gscores):

        # we will want to batch original glabels and gbboxes
        # this information is still useful even if they are padded after dequeuing
        dynamic_pad = True
        batch_shape = [1, 1, 1, 1, 1] + [len(gclasses), len(glocalisations), len(gscores)]
        tensors = [image, filename, glabels, gbboxes, gdifficults, gclasses, glocalisations, gscores]
        # Batch the samples
        if self.is_training_data:
            self.num_preprocessing_threads = 1
        else:
            # to make sure data is fectched in sequence during evaluation
            self.num_preprocessing_threads = 1

        # tf.train.batch accepts only list of tensors, this batch shape can used to
        # flatten the list in list, and later on convet it back to list in list.
        batch = tf.train.batch(
            tf_utils.reshape_list(tensors),
            batch_size=self.batch_size,
            num_threads=self.num_preprocessing_threads,
            dynamic_pad=dynamic_pad,
            capacity=5 * self.batch_size)

        # convert it back to the list in list format which allows us to easily use later on
        batch = tf_utils.reshape_list(batch, batch_shape)
        return batch
