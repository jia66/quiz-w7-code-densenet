"""Contains a variant of the densenet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def trunc_normal(stddev): return tf.truncated_normal_initializer(stddev=stddev)


def bn_act_conv_drp(current, num_outputs, kernel_size, scope='block'):
    current = slim.batch_norm(current, scope=scope + '_bn')
    current = tf.nn.relu(current)
    current = slim.conv2d(current, num_outputs, kernel_size, scope=scope + '_conv')
    current = slim.dropout(current, scope=scope + '_dropout')
    return current


def block(net, layers, growth, scope='block'):
    for idx in range(layers):
        bottleneck = bn_act_conv_drp(net, 4 * growth, [1, 1],
                                     scope=scope + '_conv1x1' + str(idx))
        tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],
                              scope=scope + '_conv3x3' + str(idx))
        net = tf.concat(axis=3, values=[net, tmp])
    return net


def densenet(images, num_classes=1001, is_training=False,
             dropout_keep_prob=0.8,
             scope='densenet'):
    """Creates a variant of the densenet model.

      images: A batch of `Tensors` of size [batch_size, height, width, channels].
      num_classes: the number of classes in the dataset.
      is_training: specifies whether or not we're currently training the model.
        This variable will determine the behaviour of the dropout layer.
      dropout_keep_prob: the percentage of activation values that are retained.
      prediction_fn: a function to get predictions out of logits.
      scope: Optional variable_scope.

    Returns:
      logits: the pre-softmax activations, a tensor of size
        [batch_size, `num_classes`]
      end_points: a dictionary from components of the network to the corresponding
        activation.
    """
    growth = 24
    compression_rate = 0.5

    def reduce_dim(input_feature):
        return int(int(input_feature.shape[-1]) * compression_rate)

    end_points = {}

    with tf.variable_scope(scope, 'DenseNet', [images, num_classes]):
        with slim.arg_scope(bn_drp_scope(is_training=is_training,
                                         keep_prob=dropout_keep_prob)) as ssc:
            #pass
            ##########################
            # Put your code here.
            ##########################
            net = slim.conv2d(images, 2*growth, [7,7], stride = 2, scope ='conv1')
            end_points['conv1'] = net

            net = slim.max_pool2d(net, [3,3], padding = 'same', stride=2 ,scope = 'pool1')
            end_points['pool1'] = net

            net = block(net, 6 , growth, scope = 'block1')
            end_points['block1'] = net

            net = bn_act_conv_drp(net, reduce_dim(net), [1,1], scope = 'transition1')
            end_points['transition1'] = net

            net = slim.avg_pool2d(net, [2,2], stride = 2, scope = 'avgpool1')
            end_points['avgpool1'] = net

            net = block(net, 12 , growth, scope = 'block2')
            end_points['block2'] = net

            net = bn_act_conv_drp(net, reduce_dim(net), [1,1], scope = 'transition2')
            end_points['transition2'] = net

            net = slim.avg_pool2d(net, [2,2], stride = 2, scope = 'avgpool2')
            end_points['avgpool2'] = net


            net = block(net, 36, growth, scope='block3')
            end_points['block3'] = net

            net = bn_act_conv_drp(net, reduce_dim(net), [1, 1], scope='transition3')
            end_points['transition3'] = net

            net = slim.avg_pool2d(net, [2, 2], stride=2, scope='avgpool3')
            end_points['avgpool3'] = net


            aux = bn_act_conv_drp(net, reduce_dim(net), [3,3], scope = 'AuxLogits')
            aux = slim.conv2d(aux, int(net.shape[-1]), [3,3], activation_fn = tf.nn.relu, scope = 'AuxLogits_conv1')
            aux = slim.avg_pool2d(aux, aux.shape[1:3])
            aux = slim.flatten(aux)
            aux_logits = slim.fully_connected(aux, num_classes, activation_fn = None, scope = 'Aux_logits')
            end_points['AuxLogits'] = aux_logits


            net = block(net, 24, growth, scope='block4')
            end_points['block4'] = net


            net = slim.batch_norm(net, scope = 'last_batch_norm_relu')
            net = tf.nn.relu(net)

            net = slim.avg_pool2d(net, net.shape[1:3], scope = 'global_average')
            end_points['global_average'] = net

            biases_initializer = tf.constant_initializer(0.1)
            pre_logit = slim.conv2d(net, num_classes, [1,1], biases_initializer = biases_initializer, scope = 'pre_logits')
            end_points['pre_logits'] = net

            logits = tf.squeeze(pre_logit)
            end_points['logits'] = logits

    return logits, end_points


def bn_drp_scope(is_training=True, keep_prob=0.8):
    keep_prob = keep_prob if is_training else 1
    with slim.arg_scope(
        [slim.batch_norm],
            scale=True, is_training=is_training, updates_collections=None):
        with slim.arg_scope(
            [slim.dropout],
                is_training=is_training, keep_prob=keep_prob) as bsc:
            return bsc


def densenet_arg_scope(weight_decay=0.004):
    """Defines the default densenet argument scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False),
        activation_fn=None, biases_initializer=None, padding='same',
            stride=1) as sc:
        return sc


densenet.default_image_size = 224
