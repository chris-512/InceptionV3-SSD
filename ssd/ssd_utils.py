"""
This file contains methods and functions used by SSDModel.
These are not directly needed for training and evaluation.
"""

import tensorflow as tf
import numpy as np
import math
from collections import namedtuple
from utils import custom_layers
import tf_extended as tfe
import ssd

slim = tf.contrib.slim


# =========================================================================== #
# Definition of the parameter structure
# =========================================================================== #
SSDParams = namedtuple('SSDParameters', ['model_name',
                                         'img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feature_layers',
                                         'feature_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'])


def tensor_shape(x, rank=3):
    """Returns the dimensions of a tensor.
    Args:
      image: A N-D Tensor of shape.
    Returns:
      A list of dimensions. Dimensions that are statically known are python
        integers,otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d for s, d in zip(static_shape, dynamic_shape)]


def multibox_layer(inputs, num_classes, anchor_sizes, anchor_ratios, normalization):
    """
    Construct a multibox layer, return a class and localization predictions.
    """
    net = inputs
    if normalization > 0:
        net = custom_layers.l2_normalization(net, scaling=True)

    # Number of anchors.
    num_anchors = len(anchor_sizes) + len(anchor_ratios)

    # Location.
    num_loc_pred = num_anchors * 4
    loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], activation_fn=None, scope='conv_loc')
    loc_pred = custom_layers.channel_to_last(loc_pred)
    loc_pred = tf.reshape(loc_pred, ssd.ssd_utils.tensor_shape(loc_pred, 4)[:-1] + [num_anchors, 4])

    # Class prediction.
    num_cls_pred = num_anchors * num_classes
    cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], activation_fn=None, scope='conv_cls')
    cls_pred = custom_layers.channel_to_last(cls_pred)
    cls_pred = tf.reshape(cls_pred, ssd.ssd_utils.tensor_shape(cls_pred, 4)[:-1] + [num_anchors, num_classes])

    return cls_pred, loc_pred


def compute_jaccard(gt_bboxes, anchors):

    gt_bboxes = tf.reshape(gt_bboxes, (-1, 1, 4))
    anchors = tf.reshape(anchors, (1, -1, 4))

    inter_ymin = tf.maximum(gt_bboxes[:, :, 0], anchors[:, :, 0])
    inter_xmin = tf.maximum(gt_bboxes[:, :, 1], anchors[:, :, 1])
    inter_ymax = tf.minimum(gt_bboxes[:, :, 2], anchors[:, :, 2])
    inter_xmax = tf.minimum(gt_bboxes[:, :, 3], anchors[:, :, 3])

    h = tf.maximum(inter_ymax - inter_ymin, 0.)
    w = tf.maximum(inter_xmax - inter_xmin, 0.)

    inter_area = h * w
    anchors_area = (anchors[:, :, 3] - anchors[:, :, 1]) * (anchors[:, :, 2] - anchors[:, :, 0])
    gt_bboxes_area = (gt_bboxes[:, :, 3] - gt_bboxes[:, :, 1]) * (gt_bboxes[:, :, 2] - gt_bboxes[:, :, 0])
    union_area = anchors_area - inter_area + gt_bboxes_area
    jaccard = inter_area / union_area

    return jaccard


def anchor_one_layer(img_shape, feat_shape, sizes, ratios, step, offset=0.5, dtype=np.float32):
    """Computer SSD default anchor boxes for one feature layer.

    Determine the relative position grid of the centers, and the relative
    width and height.

    Arguments:
      feat_shape: Feature shape, used for computing relative position grids;
      size: Absolute reference sizes;
      ratios: Ratios to use on these features;
      img_shape: Image shape, used for computing height, width relatively to the
        former;
      offset: Grid offset.

    Return:
      y, x, h, w: Relative x and y grids, and height and width.
    """
    # Compute the position grid: simple way.
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = len(sizes) + len(ratios)
    h = np.zeros((num_anchors,), dtype=dtype)
    w = np.zeros((num_anchors,), dtype=dtype)
    # Add first anchor boxes with ratio=1.
    h[0] = sizes[0] / img_shape[0]
    w[0] = sizes[0] / img_shape[1]
    di = 1
    if len(sizes) > 1:
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        di += 1
    for i, r in enumerate(ratios):
        h[i + di] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[i + di] = sizes[0] / img_shape[1] * math.sqrt(r)
    return y, x, h, w


def arg_scope(weight_decay=0.0005, data_format='NHWC'):
    """Defines the VGG arg scope.

    Args:
      weight_decay: The l2 regularization coefficient.

    Returns:
      An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            data_format=data_format):
            with slim.arg_scope([custom_layers.pad2d,
                                 custom_layers.l2_normalization,
                                 custom_layers.channel_to_last],
                                data_format=data_format) as sc:
                return sc

# =========================================================================== #
# boxes SSD encoding / decoding.
# =========================================================================== #
def bboxes_encode_layer(labels,
                        bboxes,
                        anchors_layer,
                        num_classes,
                        prior_scaling=[0.1, 0.1, 0.2, 0.2],
                        dtype=tf.float32):
    """Encode groundtruth labels and bounding boxes using SSD anchors from
    one layer.

    Arguments:
      labels: 1D Tensor(int64) containing groundtruth labels;
      bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
      anchors_layer: Numpy array with layer anchors;
      matching_threshold: Threshold for positive match with groundtruth bboxes;
      prior_scaling: Scaling of encoded coordinates.

    Return:
      (target_labels, target_localizations, target_scores): Target Tensors.
    """
    # Anchors coordinates and volume.
    yref, xref, href, wref = anchors_layer
    ymin = yref - href / 2.
    xmin = xref - wref / 2.
    ymax = yref + href / 2.
    xmax = xref + wref / 2.
    vol_anchors = (xmax - xmin) * (ymax - ymin)

    # Initialize tensors...
    shape = (yref.shape[0], yref.shape[1], href.size)
    feat_labels = tf.zeros(shape, dtype=tf.int64)
    feat_scores = tf.zeros(shape, dtype=dtype)

    feat_ymin = tf.zeros(shape, dtype=dtype)
    feat_xmin = tf.zeros(shape, dtype=dtype)
    feat_ymax = tf.ones(shape, dtype=dtype)
    feat_xmax = tf.ones(shape, dtype=dtype)

    def jaccard_with_anchors(bbox):
        """Compute jaccard score between a box and the anchors.
        """
        int_ymin = tf.maximum(ymin, bbox[0])
        int_xmin = tf.maximum(xmin, bbox[1])
        int_ymax = tf.minimum(ymax, bbox[2])
        int_xmax = tf.minimum(xmax, bbox[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        union_vol = vol_anchors - inter_vol \
            + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        jaccard = tf.div(inter_vol, union_vol)
        return jaccard

    def intersection_with_anchors(bbox):
        """Compute intersection between score a box and the anchors.
        """
        int_ymin = tf.maximum(ymin, bbox[0])
        int_xmin = tf.maximum(xmin, bbox[1])
        int_ymax = tf.minimum(ymax, bbox[2])
        int_xmax = tf.minimum(xmax, bbox[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        inter_vol = h * w
        scores = tf.div(inter_vol, vol_anchors)
        return scores

    def condition(i, feat_labels, feat_scores,
                  feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        """Condition: check label index.
        """
        r = tf.less(i, tf.shape(labels))
        return r[0]

    def body(i, feat_labels, feat_scores,
             feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        """Body: update feature labels, scores and bboxes.
        Follow the original SSD paper for that purpose:
          - assign values when jaccard > 0.5;
          - only update if beat the score of other bboxes.
        """
        # Jaccard score.
        label = labels[i]
        bbox = bboxes[i]
        jaccard = jaccard_with_anchors(bbox)
        # Mask: check threshold + scores + no annotations + num_classes.
        # jaccard is bigger than current matched bbox
        mask = tf.greater(jaccard, feat_scores)
        # mask = tf.logical_and(mask, tf.greater(jaccard, matching_threshold))
        # it's not "no annotations"
        mask = tf.logical_and(mask, feat_scores > -0.5)
        # the label value is valid
        mask = tf.logical_and(mask, label < num_classes)
        imask = tf.cast(mask, tf.int64)
        fmask = tf.cast(mask, dtype)
        # Update values using mask.
        feat_labels = imask * label + (1 - imask) * feat_labels
        feat_scores = tf.where(mask, jaccard, feat_scores)

        feat_ymin = fmask * bbox[0] + (1 - fmask) * feat_ymin
        feat_xmin = fmask * bbox[1] + (1 - fmask) * feat_xmin
        feat_ymax = fmask * bbox[2] + (1 - fmask) * feat_ymax
        feat_xmax = fmask * bbox[3] + (1 - fmask) * feat_xmax

        # Check no annotation label: ignore these anchors...
        # TODO, we probably can do without below code, will remove them in the future
        # This is because we've already checked the label previosly, which means feat_scores is already 0,
        # thus belong to negative sample
        # The idea comes from the KITTI dataset where some part of the dataset images are signaled as being not labelled :
        # there may be a car/person/... in these parts, but it has not been segmented. If you don't keep track of these parts,
        # you may end up with the SSD model detecting objects not annotated, and the loss function thinking it is False positive,
        # and pushing for not detecting it. Which is not really what we want !So basically,
        # I set up a mask such that the loss function ignores the anchors which overlap too much with parts of images no-annotated.
        #             interscts = intersection_with_anchors(bbox)
        #             mask = tf.logical_and(interscts > ignore_threshold,
        #                                   label == no_annotation_label)
        #             # Replace scores by -1.
        #             feat_scores = tf.where(mask, -tf.cast(mask, dtype), feat_scores)

        return [i + 1, feat_labels, feat_scores,
                feat_ymin, feat_xmin, feat_ymax, feat_xmax]

    # Main loop definition.
    i = 0
    [i, feat_labels, feat_scores,
     feat_ymin, feat_xmin,
     feat_ymax, feat_xmax] = tf.while_loop(condition, body,
                                           [i, feat_labels, feat_scores,
                                            feat_ymin, feat_xmin,
                                            feat_ymax, feat_xmax])
    # Transform to center / size.
    feat_cy = (feat_ymax + feat_ymin) / 2.
    feat_cx = (feat_xmax + feat_xmin) / 2.
    feat_h = feat_ymax - feat_ymin
    feat_w = feat_xmax - feat_xmin
    # Encode features, convert ground truth bboxes to  shape offset relative to default boxes
    feat_cy = (feat_cy - yref) / href / prior_scaling[0]
    feat_cx = (feat_cx - xref) / wref / prior_scaling[1]
    feat_h = tf.log(feat_h / href) / prior_scaling[2]
    feat_w = tf.log(feat_w / wref) / prior_scaling[3]
    # Use SSD ordering: x / y / w / h instead of ours.
    feat_localizations = tf.stack(
        [feat_cx, feat_cy, feat_w, feat_h], axis=-1)
    return feat_labels, feat_localizations, feat_scores


def _bboxes_decode_layer(feat_localizations,
                               anchors_layer,
                               prior_scaling=[0.1, 0.1, 0.2, 0.2]):
    """Compute the relative bounding boxes from the layer features and
    reference anchor bounding boxes.

    Arguments:
      feat_localizations: Tensor containing localization features.
      anchors: List of numpy array containing anchor boxes.

    Return:
      Tensor Nx4: ymin, xmin, ymax, xmax
    """
    yref, xref, href, wref = anchors_layer

    # Compute center, height and width
    cx = feat_localizations[:, :, :, :, 0] * wref * prior_scaling[0] + xref
    cy = feat_localizations[:, :, :, :, 1] * href * prior_scaling[1] + yref
    w = wref * tf.exp(feat_localizations[:, :, :, :, 2] * prior_scaling[2])
    h = href * tf.exp(feat_localizations[:, :, :, :, 3] * prior_scaling[3])
    # Boxes coordinates.
    ymin = cy - h / 2.
    xmin = cx - w / 2.
    ymax = cy + h / 2.
    xmax = cx + w / 2.
    bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    return bboxes


def bboxes_decode(feat_localizations,
                         anchors,
                         prior_scaling=[0.1, 0.1, 0.2, 0.2],
                         scope='ssd_bboxes_decode'):
    """Compute the relative bounding boxes from the SSD net features and
    reference anchors bounding boxes.

    Arguments:
      feat_localizations: List of Tensors containing localization features.
      anchors: List of numpy array containing anchor boxes.

    Return:
      List of Tensors Nx4: ymin, xmin, ymax, xmax
    """
    with tf.name_scope(scope):
        bboxes = []
        for i, anchors_layer in enumerate(anchors):
            bboxes.append(
                _bboxes_decode_layer(feat_localizations[i], anchors_layer, prior_scaling))
        return bboxes


# =========================================================================== #
# bbox - ground truth box matching
# =========================================================================== #
def _match_no_miss(gt_anchor_labels, gt_anchor_bboxes, gt_anchor_scores, jaccard, gt_labels, gt_bboxes, num_anchors):
    # make sure every ground truth box can be matched to at least one anchor box
    max_inds = tf.cast(tf.argmax(jaccard, axis=1), tf.int32)

    def cond(i, gt_anchors_labels, gt_anchors_bboxes, gt_anchors_scores):
        r = tf.less(i, tf.shape(gt_labels)[0])
        return r

    def body(i, gt_anchors_labels, gt_anchors_bboxes, gt_anchors_scores):
        # upate gt_anchors_labels
        updates = tf.reshape(gt_labels[i], [-1])
        indices = tf.reshape(max_inds[i], [1, -1])
        shape = tf.reshape(num_anchors, [-1])

        new_labels = tf.scatter_nd(indices, updates, shape)
        new_mask = tf.cast(new_labels, tf.bool)
        gt_anchors_labels = tf.where(new_mask, new_labels, gt_anchors_labels)

        # update gt_anchors_bboxes
        updates = tf.reshape(gt_bboxes[i], [1, -1])
        indices = tf.reshape(max_inds[i], [1, -1])
        shape = tf.shape(gt_anchors_bboxes)
        new_bboxes = tf.scatter_nd(indices, updates, shape)
        gt_anchors_bboxes = tf.where(new_mask, new_bboxes, gt_anchors_bboxes)

        # update gt_anchors_scores
        updates = tf.reshape(jaccard[i, max_inds[i]], [-1])
        indices = tf.reshape(max_inds[i], [1, -1])
        shape = tf.reshape(num_anchors, [-1])
        new_scores = tf.scatter_nd(indices, updates, shape)
        gt_anchors_scores = tf.where(new_mask, new_scores, gt_anchors_scores)

        return [i + 1, gt_anchors_labels, gt_anchors_bboxes, gt_anchors_scores]

    i = 0
    [i, gt_anchor_labels, gt_anchor_bboxes, gt_anchor_scores] = tf.while_loop(cond, body, [i, gt_anchor_labels,
                                                                                           gt_anchor_bboxes,
                                                                                           gt_anchor_scores])

    return gt_anchor_labels, gt_anchor_bboxes, gt_anchor_scores


def match_no_labels(gt_anchor_labels, gt_anchor_bboxes, gt_anchor_scores, jaccard, matching_threshold,
                      gt_labels, gt_bboxes, num_anchors):
    # For images without labels, just return all zero tensors
    return gt_anchor_labels, gt_anchor_bboxes, gt_anchor_scores


def match_with_labels(gt_anchor_labels, gt_anchor_bboxes, gt_anchor_scores, jaccard, matching_threshold,
                        gt_labels, gt_bboxes, num_anchors):
    # debugging info
    # jaccard = tf.Print(jaccard, [gt_labels], "gt_labels")
    # match default boxes to any ground truth with jaccard overlap higher than a threshold (0.5).
    mask = tf.reduce_max(jaccard, axis=0) > matching_threshold
    mask_inds = tf.argmax(jaccard, axis=0)
    matched_labels = tf.gather(gt_labels, mask_inds)
    gt_anchor_labels = tf.where(mask, matched_labels, gt_anchor_labels)
    gt_anchor_bboxes = tf.where(mask, tf.gather(gt_bboxes, mask_inds), gt_anchor_bboxes)
    gt_anchor_scores = tf.reduce_max(jaccard, axis=0)

    # matching each ground truth box to the default box with the best jaccard overlap
    use_no_miss = True
    if use_no_miss:
        gt_anchor_labels, gt_anchor_bboxes, gt_anchor_scores = \
            _match_no_miss(gt_anchor_labels, gt_anchor_bboxes, gt_anchor_scores,
                            jaccard, gt_labels, gt_bboxes, num_anchors)

    return gt_anchor_labels, gt_anchor_bboxes, gt_anchor_scores


# =========================================================================== #
# SSD boxes selection.
# =========================================================================== #
def _bboxes_select_layer(predictions_layer, localizations_layer,
                        select_threshold=None,
                        num_classes=21,
                        ignore_class=0,
                        scope=None):
    """Extract classes, scores and bounding boxes from features in one layer.
    Batch-compatible: inputs are supposed to have batch-type shapes.

    Args:
      predictions_layer: A SSD prediction layer;
      localizations_layer: A SSD localization layer;
      select_threshold: Classification threshold for selecting a box. All boxes
        under the threshold are set to 'zero'. If None, no threshold applied.
    Return:
      d_scores, d_bboxes: Dictionary of scores and bboxes Tensors of
        size Batches X N x 1 | 4. Each key corresponding to a class.
    """
    select_threshold = 0.0 if select_threshold is None else select_threshold
    with tf.name_scope(scope, 'ssd_bboxes_select_layer', [predictions_layer, localizations_layer]):
        # Reshape features: Batches x N x N_labels | 4
        p_shape = tfe.get_shape(predictions_layer)
        predictions_layer = tf.reshape(predictions_layer, tf.stack([p_shape[0], -1, p_shape[-1]]))
        l_shape = tfe.get_shape(localizations_layer)
        localizations_layer = tf.reshape(localizations_layer, tf.stack([l_shape[0], -1, l_shape[-1]]))

        d_scores = {}
        d_bboxes = {}
        for c in range(0, num_classes):
            if c != ignore_class:
                # Remove boxes under the threshold.
                scores = predictions_layer[:, :, c]
                fmask = tf.cast(tf.greater_equal(scores, select_threshold), scores.dtype)
                scores = scores * fmask
                bboxes = localizations_layer * tf.expand_dims(fmask, axis=-1)
                # Append to dictionary.
                d_scores[c] = scores
                d_bboxes[c] = bboxes

        return d_scores, d_bboxes


def bboxes_select(predictions_net, localizations_net,
                  select_threshold=None,
                  num_classes=21,
                  ignore_class=0,
                  scope=None):
    """Extract classes, scores and bounding boxes from network output layers.
    Batch-compatible: inputs are supposed to have batch-type shapes.

    Args:
      predictions_net: List of SSD prediction layers;
      localizations_net: List of localization layers;
      select_threshold: Classification threshold for selecting a box. All boxes
        under the threshold are set to 'zero'. If None, no threshold applied.
    Return:
      d_scores, d_bboxes: Dictionary of scores and bboxes Tensors of
        size Batches X N x 1 | 4. Each key corresponding to a class.
    """
    with tf.name_scope(scope, 'ssd_bboxes_select', [predictions_net, localizations_net]):
        l_scores = []
        l_bboxes = []
        for i in range(len(predictions_net)):
            scores, bboxes = _bboxes_select_layer(predictions_net[i],
                                                 localizations_net[i],
                                                 select_threshold,
                                                 num_classes,
                                                 ignore_class)
            l_scores.append(scores)
            l_bboxes.append(bboxes)
        # Concat results.
        d_scores = {}
        d_bboxes = {}
        for c in l_scores[0].keys():
            ls = [s[c] for s in l_scores]
            lb = [b[c] for b in l_bboxes]
            d_scores[c] = tf.concat(ls, axis=1)
            d_bboxes[c] = tf.concat(lb, axis=1)
        return d_scores, d_bboxes
