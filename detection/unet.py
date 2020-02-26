import tensorflow as tf
if int(tf.__version__[0]) > 1:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
import math

def _create_conv_relu(inputs, name, filters, dropout_ratio, is_training, strides=[1,1], kernel_size=[3,3], padding="SAME", relu=True):
    net = tf.layers.conv2d(inputs=inputs, filters=filters, strides=strides, kernel_size=kernel_size, padding=padding, name="%s_conv" % name)
    if dropout_ratio > 0:
        net = tf.layers.dropout(inputs=net, rate=dropout_ratio, training=is_training, name="%s_dropout" % name)
    net = tf.layers.batch_normalization(net, center=True, scale=False, training=is_training, name="%s_bn" % name)
    if relu:
        net = tf.nn.relu(net) # leaky relu
    return net


def _create_pool(data, name, pool_size=[2,2], strides=[2,2]):
    pool = tf.layers.max_pooling2d(inputs=data, pool_size=pool_size, strides=strides, padding='SAME', name=name)
    return pool


def _contracting_path(data, num_layers, num_filters, dropout_ratio, is_training):
    interim = []

    dim_out = num_filters
    for i in range(num_layers):
        name = "c_%i" % i
        conv1 = _create_conv_relu(data, name + "_1", dim_out, dropout_ratio=dropout_ratio, is_training=is_training)
        conv2 = _create_conv_relu(conv1, name + "_2", dim_out, dropout_ratio=dropout_ratio, is_training=is_training)
        pool = _create_pool(conv2, name)
        data = pool

        dim_out *=2
        interim.append(conv2)

    return (interim, data)


def _expansive_path(data, interim, num_layers, dim_in, dropout_ratio, is_training):
    dim_out = int(dim_in / 2)
    for i in range(num_layers):
        name = "e_%i" % i
        upconv = tf.layers.conv2d_transpose(data, filters=dim_out, kernel_size=2, strides=2, name="%s_upconv" % name)
        concat = tf.concat([interim[len(interim)-i-1], upconv], 3)
        conv1 = _create_conv_relu(concat, name + "_1", dim_out, dropout_ratio=dropout_ratio, is_training=is_training)
        #suffix = "last" if (i == num_layers - 1) else suffix + "_2"
        conv2 = _create_conv_relu(conv1, name + "_2", dim_out, dropout_ratio=dropout_ratio, is_training=is_training)
        data = conv2
        dim_out = int(dim_out / 2)
    return data


def create_unet2(num_layers, num_filters, data, is_training, prev=None, dropout_ratio=0, classes=3):

    (interim, contracting_data) = _contracting_path(data, num_layers, num_filters, dropout_ratio, is_training)

    middle_dim = num_filters * 2**num_layers
    middle_conv_1 = _create_conv_relu(contracting_data, "m_1", middle_dim, dropout_ratio=dropout_ratio, is_training=is_training)
    middle_conv_2 = _create_conv_relu(middle_conv_1, "m_2", middle_dim, dropout_ratio=dropout_ratio, is_training=is_training)
    middle_end = middle_conv_2

    expansive_path = _expansive_path(middle_end, interim, num_layers, middle_dim, dropout_ratio, is_training)
    last_relu = expansive_path

    if prev != None:
        expansive_path = tf.concat([prev, expansive_path], 3)

    conv_logits = _create_conv_relu(expansive_path, "conv_logits", num_filters, dropout_ratio=dropout_ratio, is_training=is_training)
    logits = _create_conv_relu(conv_logits, "logits", classes, dropout_ratio=dropout_ratio, is_training=is_training)

    conv_angle = _create_conv_relu(expansive_path, "conv_angle", num_filters, dropout_ratio=dropout_ratio, is_training=is_training, relu=False)
    angle_pred = _create_conv_relu(conv_angle, "angle_pred", 1, dropout_ratio=dropout_ratio, is_training=is_training, relu=False)
    return logits, last_relu, angle_pred


def loss(logits, labels, weight_map, numclasses=3):
    oh_labels = tf.one_hot(indices=tf.cast(labels, tf.uint8), depth=numclasses, name="one_hot")
    loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=oh_labels)
    weighted_loss = tf.multiply(loss_map, weight_map)
    loss = tf.reduce_mean(weighted_loss, name="weighted_loss")
    #tf.add_to_collection('losses', loss)
    return loss #tf.add_n(tf.get_collection('losses'), name='total_loss')


def angle_loss(angle_pred, angle_labels, weight_map):

    sh = tf.shape(angle_pred)
    angle_pred = tf.reshape(angle_pred, [sh[0],sh[1],sh[2]])
    bg_mask = tf.logical_or(tf.less(angle_pred, 0), tf.less(angle_labels, 0))
    fg_mask = tf.logical_not(bg_mask)

    fg_loss = tf.multiply(tf.boolean_mask(weight_map, fg_mask),
                          tf.square(tf.sin((tf.boolean_mask(angle_pred, fg_mask) - tf.boolean_mask(angle_labels, fg_mask))*math.pi)))
    bg_loss = tf.multiply(tf.boolean_mask(weight_map, bg_mask),
                          tf.square(tf.boolean_mask(angle_pred, bg_mask) - tf.boolean_mask(angle_labels, bg_mask)))

    fg_loss = tf.reduce_mean(fg_loss, name="weighted_angle_loss")
    bg_loss = tf.reduce_mean(bg_loss, name="weighted_bg_angle_loss")

    loss = fg_loss + bg_loss
    #tf.add_to_collection('losses', loss)
    return loss #tf.add_n(tf.get_collection('losses'), name='total_loss')
