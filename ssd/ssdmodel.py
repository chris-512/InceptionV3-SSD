import tensorflow as tf
from tensorflow.python.ops import array_ops
import numpy as np
import math

import tf_extended as tfe
from utils import custom_layers
from ssd import ssd_utils
from ssd import ssd_blocks
from nets import nets_factory as nf

slim = tf.contrib.slim


class SSDModel:
    """
    Implementation of the SSD network.
    """

    # ============================= PUBLIC METHODS ============================== #
    def __init__(self, feature_extractor, model_name, weight_decay, is_training=True):
        """
        Initialize an instance of the SSDModel
        :param feature_extractor: name of the feature extractor (backbone)
        :param model_name: name of the SSD model to use: ssd300 or ssd500
        """
        if feature_extractor not in nf.base_networks_map:
            raise ValueError('Feature extractor %s unknown.' % feature_extractor)
        if model_name not in ['ssd300', 'ssd512']:
            raise ValueError('Model %s unknown. Choose either ssd300 or ssd512.' % model_name)

        if model_name == 'ssd300':
            self.params = ssd_blocks.ssd300_params
            self._ssd_blocks = ssd_blocks.ssd300
        elif model_name == 'ssd512':
            self.params = ssd_blocks.ssd512_params
            self._ssd_blocks = ssd_blocks.ssd512

        self.feature_extractor = feature_extractor
        self._feature_extractor = nf.get_base_network_fn(feature_extractor, weight_decay=weight_decay)
        self.params.feature_layers.insert(0, ssd_blocks.feature_layer[feature_extractor])
        self.is_training = is_training
        # all of the computed anchors for this model,
        # format: layer_number, numpy array format for x / y / w / h
        self.np_anchors = None
        # format: layer number, numpy format for ymin,xmin,ymax,xmax
        self.np_anchors_minmax = None

    def get_model(self, inputs):
        net, end_points = self._feature_extractor(inputs)
        keep_prob = 0.8
        with slim.arg_scope([slim.conv2d], activation_fn=None):
            with slim.arg_scope([slim.batch_norm], activation_fn=tf.nn.relu,
                                is_training=self.is_training, updates_collections=None):
                with slim.arg_scope([slim.dropout], is_training=self.is_training, keep_prob=keep_prob):
                    with tf.variable_scope(self.params.model_name):
                        net, end_points = self._ssd_blocks(net, end_points)

        # Prediction and localisations layers.
        # set breakpoint here to find out which layer in feature extractor to use as bbox layer
        predictions = []
        logits = []
        localisations = []
        with tf.variable_scope('bbox_layers'):
            for i, layer in enumerate(self.params.feature_layers):
                with tf.variable_scope(layer + '_box'):
                    p, l = ssd_utils.multibox_layer(end_points[layer],
                                                    self.params.num_classes,
                                                    self.params.anchor_sizes[i],
                                                    self.params.anchor_ratios[i],
                                                    self.params.normalizations[i])
                predictions.append(slim.softmax(p))
                logits.append(p)
                localisations.append(l)

        return predictions, localisations, logits, end_points

    def get_anchors_all_layers(self, dtype=np.float32):
        """Compute anchor boxes for all feature layers.
        """
        # ssd_anchors_all_layers()
        layers_anchors = []
        for i, s in enumerate(self.params.feature_shapes):
            anchor_bboxes = ssd_utils.anchor_one_layer(self.params.img_shape, s,
                                                       self.params.anchor_sizes[i],
                                                       self.params.anchor_ratios[i],
                                                       self.params.anchor_steps[i],
                                                       offset=self.params.anchor_offset,
                                                       dtype=dtype)
            layers_anchors.append(anchor_bboxes)
        return layers_anchors

    def get_all_anchors(self, minmaxformat=False):
        #         print("minmaxformat {}".format(minmaxformat))

        if self.np_anchors is None:
            anchors = self.get_anchors_all_layers()
            self.np_anchors = []
            self.np_anchors_minmax = []
            for _, anchors_layer in enumerate(anchors):
                yref, xref, href, wref = anchors_layer
                ymin = yref - href / 2.
                xmin = xref - wref / 2.
                ymax = yref + href / 2
                xmax = xref + wref / 2.

                temp_achors = np.concatenate(
                    [ymin[..., np.newaxis], xmin[..., np.newaxis],
                        ymax[..., np.newaxis], xmax[..., np.newaxis]],
                    axis=-1)
                self.np_anchors_minmax.append(temp_achors)

                # Transform to center / size.
                cy = (ymax + ymin) / 2.
                cx = (xmax + xmin) / 2.
                h = ymax - ymin
                w = xmax - xmin
                temp_achors = np.concatenate(
                    [cx[..., np.newaxis], cy[..., np.newaxis], w[..., np.newaxis], h[..., np.newaxis]], axis=-1)

                # append achors for this layer
                self.np_anchors.append(temp_achors)
        if minmaxformat:
            return self.np_anchors_minmax
        else:
            return self.np_anchors

    def match_achors(self, gt_labels, gt_bboxes, matching_threshold=0.5):
        anchors = self.get_all_anchors(minmaxformat=True)
        # flattent the anchors
        temp_anchors = []
        for i in range(len(anchors)):
            temp_anchors.append(tf.reshape(anchors[i], [-1, 4]))
        anchors = tf.concat(temp_anchors, axis=0)

        jaccard = ssd_utils.compute_jaccard(gt_bboxes, anchors)
        num_anchors = jaccard.shape[1]

        # initialize output
        gt_anchor_labels = tf.zeros(num_anchors, dtype=tf.int64)
        gt_anchor_scores = tf.zeros(num_anchors, dtype=tf.float32)
        gt_anchor_ymins = tf.zeros(num_anchors)
        gt_anchor_xmins = tf.zeros(num_anchors)
        gt_anchor_ymaxs = tf.ones(num_anchors)
        gt_anchor_xmaxs = tf.ones(num_anchors)
        gt_anchor_bboxes = tf.stack(
            [gt_anchor_ymins, gt_anchor_xmins, gt_anchor_ymaxs, gt_anchor_xmaxs], axis=-1)

        n__glabels = tf.size(gt_labels)
        gt_anchor_labels, gt_anchor_bboxes, gt_anchor_scores = tf.cond(tf.equal(n__glabels, 0),
                                                                       lambda: ssd_utils.match_no_labels(
                                                                           gt_anchor_labels,
                                                                           gt_anchor_bboxes,
                                                                           gt_anchor_scores,
                                                                           jaccard,
                                                                           matching_threshold,
                                                                           gt_labels,
                                                                           gt_bboxes,
                                                                           num_anchors),
                                                                       lambda: ssd_utils.match_with_labels(
                                                                           gt_anchor_labels, gt_anchor_bboxes,
                                                                           gt_anchor_scores, jaccard,
                                                                           matching_threshold, gt_labels, gt_bboxes,
                                                                           num_anchors))

        # Transform to center / size.
        feat_cx = (gt_anchor_bboxes[:, 3] + gt_anchor_bboxes[:, 1]) / 2.
        feat_cy = (gt_anchor_bboxes[:, 2] + gt_anchor_bboxes[:, 0]) / 2.
        feat_w = gt_anchor_bboxes[:, 3] - gt_anchor_bboxes[:, 1]
        feat_h = gt_anchor_bboxes[:, 2] - gt_anchor_bboxes[:, 0]

        xref = (anchors[:, 3] + anchors[:, 1]) / 2.
        yref = (anchors[:, 2] + anchors[:, 0]) / 2.
        wref = anchors[:, 3] - anchors[:, 1]
        href = anchors[:, 2] - anchors[:, 0]

        # Encode features, convert ground truth bboxes to  shape offset relative to default boxes
        feat_cx = (feat_cx - xref) / wref / self.params.prior_scaling[1]
        feat_cy = (feat_cy - yref) / href / self.params.prior_scaling[0]
        feat_w = tf.log(feat_w / wref) / self.params.prior_scaling[3]
        feat_h = tf.log(feat_h / href) / self.params.prior_scaling[2]

        gt_anchor_bboxes = tf.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)

        gt_anchor_labels, gt_anchor_bboxes, gt_anchor_scores = \
            self._convert2layers(gt_anchor_labels, gt_anchor_bboxes, gt_anchor_scores)

        return gt_anchor_labels, gt_anchor_bboxes, gt_anchor_scores

    def decode_bboxes_layer(self, feat_localizations, anchors):
        """convert ssd boxes from relative to input image anchors to relative to
        input width/height, for one single feature layer

        Return:
          numpy array Batches x H x W x 4: ymin, xmin, ymax, xmax
        """

        l_shape = feat_localizations.shape
        #         if feat_localizations.shape != anchors.shape:
        #             raise "feat_localizations and anchors should be of identical shape, and corresond to each other"

        # Reshape for easier broadcasting.
        feat_localizations = feat_localizations[np.newaxis, :]
        anchors = anchors[np.newaxis, :]

        xref = anchors[..., 0]
        yref = anchors[..., 1]
        wref = anchors[..., 2]
        href = anchors[..., 3]

        # Compute center, height and width
        cy = feat_localizations[..., 1] * href * self.params.prior_scaling[0] + yref
        cx = feat_localizations[..., 0] * wref * self.params.prior_scaling[1] + xref
        h = href * np.exp(feat_localizations[..., 3] * self.params.prior_scaling[2])
        w = wref * np.exp(feat_localizations[..., 2] * self.params.prior_scaling[3])

        # bboxes: ymin, xmin, xmax, ymax.
        bboxes = np.zeros_like(feat_localizations)
        bboxes[..., 0] = cy - h / 2.
        bboxes[..., 1] = cx - w / 2.
        bboxes[..., 2] = cy + h / 2.
        bboxes[..., 3] = cx + w / 2.
        bboxes = np.reshape(bboxes, l_shape)
        return bboxes

    def decode_bboxes_all_layers(self, localizations):
        """convert ssd boxes from relative to input image anchors to relative to
        input width/height

        Return:
          numpy array Batches x H x W x 4: ymin, xmin, ymax, xmax
        """
        decoded_bboxes = []
        all_anchors = self.get_all_anchors()
        for i in range(len(localizations)):
            decoded_bboxes.append(self.decode_bboxes_layer(localizations[i], all_anchors[i]))
        return decoded_bboxes

    def decode_bboxes_all_layers_tf(self, feat_localizations):
        """convert ssd boxes from relative to input image anchors to relative to input width/height

        Return:
          numpy array NlayersxNx4: ymin, xmin, ymax, xmax
        """
        anchors = self.get_anchors_all_layers()
        bboxes = ssd_utils.bboxes_decode(feat_localizations, anchors, prior_scaling=self.params.prior_scaling)
        return bboxes

    def bboxes_encode(self, labels, bboxes, dtype=tf.float32, scope='ssd_bboxes_encode'):
        # tf_ssd_bboxes_encode
        """Encode groundtruth information for all default boxes, for one input image

        Arguments:
          labels: 1D Tensor(int64) containing groundtruth labels;
          bboxes: Nx4 Tensor(float) with bboxes relative coordinates;

        Return:
          (target_labels, target_localizations, target_scores):
            Each element is a list of target Tensors.
            target_labels: target labels for all default boex,
            target_localizations: target localization offset for all default boxes
            target_scores: jaccard scores for all default boxes
            For default boxes that have no intersection with any of the ground truth boxes, target label and target score is 0,
            and target_localization is the whole input image
            If a default boxes intersect with multiple ground truth boxes, it will choose the one having the highest jaccard values
        """
        anchors = self.get_anchors_all_layers()
        with tf.name_scope(scope):
            target_labels = []
            target_localizations = []
            target_scores = []
            for i, anchors_layer in enumerate(anchors):
                with tf.name_scope('bboxes_encode_block_%i' % i):
                    t_labels, t_loc, t_scores = \
                        ssd_utils.bboxes_encode_layer(labels, bboxes, anchors_layer,
                                                      self.params.num_classes,
                                                      self.params.prior_scaling,
                                                      dtype)
                    target_labels.append(t_labels)
                    target_localizations.append(t_loc)
                    target_scores.append(t_scores)
            return target_labels, target_localizations, target_scores

    # =========================== PRIVATE METHODS ============================ #
    def _convert2layers(self, gclasses, glocalisations, gscores):
        gt_anchor_labels = []
        gt_anchor_bboxes = []
        gt_anchor_scores = []

        anchors = self.get_all_anchors(minmaxformat=False)

        start = 0
        end = 0

        for i in range(len(anchors)):
            anchor_shape = anchors[i].shape[:-1]
            anchor_shape = list(anchor_shape)
            anchor_num = np.array(anchor_shape).prod()
            start = end
            end = start + anchor_num

            gt_anchor_labels.append(tf.reshape(gclasses[start:end], anchor_shape))
            gt_anchor_scores.append(tf.reshape(gscores[start:end], anchor_shape))
            gt_anchor_bboxes.append(tf.reshape(glocalisations[start:end], anchor_shape + [4]))

        return gt_anchor_labels, gt_anchor_bboxes, gt_anchor_scores


# =========================== STATIC METHODS ============================ #
def get_losses(logits, localisations, gclasses, glocalisations, gscores,
               match_threshold=0, negative_ratio=2.5, alpha=1., scope=None):
    """Loss functions for training the SSD 300 VGG network.

    This function defines the different loss components of the SSD, and
    adds them to the TF loss collection.

    Arguments:
      logits: (list of) predictions logits Tensors;
      localisations: (list of) localisations Tensors;
      gclasses: (list of) groundtruth labels Tensors;
      glocalisations: (list of) groundtruth localisations Tensors;
      gscores: (list of) groundtruth score Tensors;
    """
    with tf.name_scope(scope, 'ssd_losses'):
        lshape = tfe.get_shape(logits[0], 5)
        num_classes = lshape[-1]
        # batch_size = lshape[0]

        # Flatten out all vectors!
        flogits = []
        fgclasses = []
        fgscores = []
        flocalisations = []
        fglocalisations = []
        for i in range(len(logits)):
            flogits.append(tf.reshape(logits[i], [-1, num_classes]))
            fgclasses.append(tf.reshape(gclasses[i], [-1]))
            fgscores.append(tf.reshape(gscores[i], [-1]))
            flocalisations.append(tf.reshape(localisations[i], [-1, 4]))
            fglocalisations.append(tf.reshape(glocalisations[i], [-1, 4]))
        # And concat the crap!
        logits = tf.concat(flogits, axis=0)
        gclasses = tf.concat(fgclasses, axis=0)
        gscores = tf.concat(fgscores, axis=0)
        localisations = tf.concat(flocalisations, axis=0)
        glocalisations = tf.concat(fglocalisations, axis=0)
        dtype = logits.dtype

        # Compute positive matching mask...
        pmask = gclasses > match_threshold
        fpmask = tf.cast(pmask, dtype)
        n_positives = tf.reduce_sum(fpmask)

        # Hard negative mining...
        # for no_classes, we only care that false positive's label is 0
        # this is why pmask sufice our needs
        no_classes = tf.cast(pmask, tf.int32)
        predictions = slim.softmax(logits)
        nmask = tf.logical_not(pmask)

        fnmask = tf.cast(nmask, dtype)
        nvalues = tf.where(nmask, predictions[:, 0], 1. - fnmask)
        nvalues_flat = tf.reshape(nvalues, [-1])
        # Number of negative entries to select.
        max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)

        n_neg = tf.cast(negative_ratio * n_positives, tf.int32)
        n_neg = tf.minimum(n_neg, max_neg_entries)
        # avoid n_neg is zero, and cause error when doing top_k later on
        n_neg = tf.maximum(n_neg, 1)

        val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
        max_hard_pred = -val[-1]
        # Final negative mask, hard negative mining
        nmask = tf.logical_and(nmask, nvalues <= max_hard_pred)
        fnmask = tf.cast(nmask, dtype)

        # Add cross-entropy loss.
        with tf.name_scope('cross_entropy_pos'):
            total_cross_pos = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=gclasses)
            total_cross_pos = tf.reduce_sum(total_cross_pos * fpmask, name="cross_entropy_pos")
            tf.losses.add_loss(total_cross_pos)

        with tf.name_scope('cross_entropy_neg'):
            total_cross_neg = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=no_classes)
            total_cross_neg = tf.reduce_sum(total_cross_neg * fnmask, name="cross_entropy_neg")
            tf.losses.add_loss(total_cross_neg)

        # Add localization loss: smooth L1, L2, ...
        with tf.name_scope('localization'):
            # Weights Tensor: positive mask + random negative.
            weights = tf.expand_dims(alpha * fpmask, axis=-1)
            total_loc = custom_layers.abs_smooth_2(localisations - glocalisations)
            total_loc = tf.reduce_sum(total_loc * weights, name="localization")
            tf.losses.add_loss(total_loc)

        total_cross = tf.add(total_cross_pos, total_cross_neg, 'cross_entropy')

        # Add to EXTRA LOSSES TF.collection
        tf.add_to_collection('EXTRA_LOSSES', total_cross_pos)
        tf.add_to_collection('EXTRA_LOSSES', total_cross_neg)
        tf.add_to_collection('EXTRA_LOSSES', total_cross)
        tf.add_to_collection('EXTRA_LOSSES', total_loc)

        # stick with the original paper in terms of definig model loss
        model_loss = tf.get_collection(tf.GraphKeys.LOSSES)
        model_loss = tf.add_n(model_loss)
        model_loss = array_ops.where(tf.equal(n_positives, 0),
                                     array_ops.zeros_like(model_loss),
                                     tf.div(1.0, n_positives) * model_loss)
        # Add regularization loss
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        regularization_loss = tf.add_n(regularization_losses, name='regularization_loss')

        # if model loss is zero, no need to do gradient update on this batch
        total_loss = array_ops.where(tf.equal(n_positives, 0),
                                     array_ops.zeros_like(model_loss),
                                     tf.add(model_loss, regularization_loss))

        # debugging info
        tf.summary.scalar("postive_num", n_positives)
        tf.summary.scalar("negative_num", n_neg)
        tf.summary.scalar("regularization_loss", regularization_loss)
        #             with tf.name_scope('variables_loc'):
        #                 selected_p = tf.boolean_mask(glocalisations, pmask)
        #                 p_mean, p_variance = tf.nn.moments(selected_p, [0])
        #                 tf.summary.scalar("mean_cx", p_mean[0])
        #                 tf.summary.scalar("mean_cy", p_mean[1])
        #                 tf.summary.scalar("mean_w", p_mean[2])
        #                 tf.summary.scalar("mean_h", p_mean[3])
        #
        #                 tf.summary.scalar("var_cx", p_variance[0])
        #                 tf.summary.scalar("var_cy", p_variance[1])
        #                 tf.summary.scalar("var_w", p_variance[2])
        #                 tf.summary.scalar("var_h", p_variance[3])

        return total_loss


def detected_bboxes(predictions, localisations, num_classes,
                    select_threshold=0.01, nms_threshold=0.45,
                    clipping_bbox=None, top_k=400, keep_top_k=200):
    """Get the detected bounding boxes from the SSD network output.
    """
    # Select top_k bboxes from predictions, and clip
    rscores, rbboxes = ssd_utils.bboxes_select(predictions, localisations,
                                               select_threshold=select_threshold,
                                               num_classes=num_classes)
    rscores, rbboxes = tfe.bboxes_sort(rscores, rbboxes, top_k=top_k)
    # Apply NMS algorithm.
    rscores, rbboxes = tfe.bboxes_nms_batch(rscores, rbboxes,
                                            nms_threshold=nms_threshold,
                                            keep_top_k=keep_top_k)
    if clipping_bbox is not None:
        rbboxes = tfe.bboxes_clip(clipping_bbox, rbboxes)
    return rscores, rbboxes
