import tensorflow as tf
from utils import custom_layers
from ssd.ssd_utils import SSDParams

slim = tf.contrib.slim

ssd300_params = SSDParams(model_name='ssd300',
                          img_shape=(300, 300),
                          num_classes=21,
                          no_annotation_label=21,
                          feature_layers=['ssd_block7', 'ssd_block8', 'ssd_block9', 'ssd_block10', 'ssd_block11'],
                          feature_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
                          anchor_size_bounds=[0.15, 0.90],
                          anchor_sizes=[(21., 45.),
                                        (45., 99.),
                                        (99., 153.),
                                        (153., 207.),
                                        (207., 261.),
                                        (261., 315.)],
                          anchor_ratios=[[2, .5],
                                         [2, .5, 3, 1. / 3],
                                         [2, .5, 3, 1. / 3],
                                         [2, .5, 3, 1. / 3],
                                         [2, .5],
                                         [2, .5]],
                          anchor_steps=[8, 16, 32, 64, 100, 300],
                          anchor_offset=0.5,
                          normalizations=[20, -1, -1, -1, -1, -1],
                          prior_scaling=[0.1, 0.1, 0.2, 0.2]
                          )


ssd512_params = SSDParams(model_name='ssd512',
                          img_shape=(512, 512),
                          num_classes=21,
                          no_annotation_label=21,
                          feature_layers=['ssd_block7', 'ssd_block8', 'ssd_block9', 'ssd_block10', 'ssd_block11', 'ssd_block12'],
                          feature_shapes=[(30, 30), (30, 30), (15, 15), (8, 8), (4, 4), (2, 2), (1, 1)],
                          anchor_size_bounds=[0.10, 0.90],
                          anchor_sizes=[(20.48, 51.2),
                                        (51.2, 133.12),
                                        (133.12, 215.04),
                                        (215.04, 296.96),
                                        (296.96, 378.88),
                                        (378.88, 460.8),
                                        (460.8, 542.72)],
                          anchor_ratios=[[2, .5],
                                         [2, .5, 3, 1. / 3],
                                         [2, .5, 3, 1. / 3],
                                         [2, .5, 3, 1. / 3],
                                         [2, .5, 3, 1. / 3],
                                         [2, .5],
                                         [2, .5]],
                          anchor_steps=[8, 16, 32, 64, 128, 256, 512],
                          anchor_offset=0.5,
                          normalizations=[20, -1, -1, -1, -1, -1, -1],
                          prior_scaling=[0.1, 0.1, 0.2, 0.2]
                          )



# TODO: find appropriate end_point for use in SSD for all nets.
feature_layer = {'alexnet_v2': '',
                 'cifarnet': '',
                 'overfeat': '',
                 'vgg_a': 'vgg_a/conv4/conv4_3',
                 'vgg_16': 'vgg_16/conv4/conv4_3',
                 'vgg_19': 'vgg_19/conv4/conv4_3',
                 'inception_v1': '',
                 'inception_v2': '',
                 'inception_v3': 'Mixed_6d',
                 'inception_v4': '',
                 'inception_resnet_v2': '',
                 'lenet': '',
                 'resnet_v1_50': '',
                 'resnet_v1_101': '',
                 'resnet_v1_152': '',
                 'resnet_v1_200': '',
                 'resnet_v2_50': '',
                 'resnet_v2_101': '',
                 'resnet_v2_152': '',
                 'resnet_v2_200': '',
                 'mobilenet_v1': 'Conv2d_11_pointwise',
                 'mobilenet_v1_075': 'Conv2d_11_pointwise',
                 'mobilenet_v1_050': 'Conv2d_11_pointwise',
                 'mobilenet_v1_025': 'Conv2d_11_pointwise',
                 'xception': ''
                 }


def ssd300(net, end_points):
    """
    Implementation of the SSD300 network.

    The default features layers with 300x300 image input are:
      conv4 ==> 38 x 38
      conv7 ==> 19 x 19
      conv8 ==> 10 x 10
      conv9 ==> 5 x 5
      conv10 ==> 3 x 3
      conv11 ==> 1 x 1
    The default image size used to train this network is 300x300.

    No prediction and localization layers included!!!
    """
    # block 6: 3x3 conv
    net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
    net = slim.batch_norm(net)
    net = custom_layers.dropout_with_noise(net)
    end_points['ssd_block6'] = net

    # block 7: 1x1 conv
    net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
    net = slim.batch_norm(net)
    net = custom_layers.dropout_with_noise(net)
    end_points['ssd_block7'] = net

    # block 8/9/10/11: 1x1 and 3x3 convolutions with stride 2 (except lasts)
    end_point = 'ssd_block8'
    with tf.variable_scope(end_point):
        net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
        net = slim.batch_norm(net)
        net = custom_layers.dropout_with_noise(net)
        net = custom_layers.pad2d(net, pad=(1, 1))
        net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        net = slim.batch_norm(net)
        net = custom_layers.dropout_with_noise(net)
    end_points[end_point] = net

    end_point = 'ssd_block9'
    with tf.variable_scope(end_point):
        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
        net = slim.batch_norm(net)
        net = custom_layers.dropout_with_noise(net)
        net = custom_layers.pad2d(net, pad=(1, 1))
        net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        net = slim.batch_norm(net)
        net = custom_layers.dropout_with_noise(net)
    end_points[end_point] = net

    end_point = 'ssd_block10'
    with tf.variable_scope(end_point):
        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
        net = slim.batch_norm(net)
        net = custom_layers.dropout_with_noise(net)
        net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
        net = slim.batch_norm(net)
        net = custom_layers.dropout_with_noise(net)
    end_points[end_point] = net

    end_point = 'ssd_block11'
    with tf.variable_scope(end_point):
        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
        net = slim.batch_norm(net)
        net = custom_layers.dropout_with_noise(net)
        net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
        net = slim.batch_norm(net)
        net = custom_layers.dropout_with_noise(net)
    end_points[end_point] = net

    return net, end_points


def ssd512(net, end_points):
    """
    Implementation of the SSD512 network.

    No prediction and localization layers included!!!

    """
    print(net)

    # Block 6: 3x3 conv
    net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
    net = slim.batch_norm(net)
    net = custom_layers.dropout_with_noise(net)
    end_points['ssd_block6'] = net

    # Block 7: 1x1 conv
    net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
    net = slim.batch_norm(net)
    net = custom_layers.dropout_with_noise(net)
    end_points['ssd_block7'] = net
    print(net)

    # Block 8/9/10/11/12: 1x1 and 3x3 convolutions stride 2 (except last).
    end_point = 'ssd_block8'
    with tf.variable_scope(end_point):
        net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
        net = slim.batch_norm(net)
        net = custom_layers.pad2d(net, pad=(1, 1))
        net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        net = slim.batch_norm(net)
    end_points[end_point] = net
    print(net)

    end_point = 'ssd_block9'
    with tf.variable_scope(end_point):
        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
        net = slim.batch_norm(net)
        net = custom_layers.pad2d(net, pad=(1, 1))
        net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        net = slim.batch_norm(net)
    end_points[end_point] = net
    print(net)

    end_point = 'ssd_block10'
    with tf.variable_scope(end_point):
        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
        net = slim.batch_norm(net)
        net = custom_layers.pad2d(net, pad=(1, 1))
        net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        net = slim.batch_norm(net)
    end_points[end_point] = net
    print(net)

    end_point = 'ssd_block11'
    with tf.variable_scope(end_point):
        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
        net = slim.batch_norm(net)
        net = custom_layers.pad2d(net, pad=(1, 1))
        net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        net = slim.batch_norm(net)
    end_points[end_point] = net
    print(net)
   
    end_point = 'ssd_block12'
    with tf.variable_scope(end_point):
        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
        net = slim.batch_norm(net)
        net = custom_layers.pad2d(net, pad=(1, 1))
        net = slim.conv2d(net, 256, [4, 4], scope='conv4x4', padding='VALID')
        net = slim.batch_norm(net)
    end_points[end_point] = net
    print(net)

    return net, end_points
