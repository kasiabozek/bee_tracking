import tensorflow as tf
if int(tf.__version__[0]) > 1:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

def inception_v3(inputs, is_training, num_classes, dropout_keep_prob=0.8, scope=None):
    """Inception model from http://arxiv.org/abs/1512.00567.
    constructs an Inception v3 network from inputs to the given final endpoint.
    this method can construct the network up to the final inception block.
    Args:
     inputs: a tensor of size [batch_size, height, width, channels].
     min_depth: Minimum depth value (number of channels) for all convolution ops.
       Enforced when depth_multiplier < 1, and not an active constraint when
       depth_multiplier >= 1.
     depth_multiplier: Float multiplier for the depth (number of channels)
       for all convolution ops. The value must be greater than zero. Typical
       usage will be to set this value in (0, 1) to reduce the number of
       parameters or computation cost of the model.
     scope: Optional variable_scope.
    Returns:
     tensor_out: output tensor corresponding to the final_endpoint.
     end_points: a set of activations for external use, for example summaries or
                 losses.
    Raises:
     ValueError: if depth_multiplier <= 0
    """
    #end_points will collect relevant activations for external use, for example
    #summaries or losses.
    end_points = {}

    with tf.name_scope(scope, 'inception_v3', [inputs]):

        # 299 x 299 x 3
        #end_point = 'Conv2d_1a_3x3'
        #net = layers.conv2d(inputs, depth(32), [3, 3], stride=2, scope=end_point)
        #end_points[end_point] = net
        # 160 x 160 x 1
        end_point = 'Conv2d_2a_3x3'
        net = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=[3, 3], activation=tf.nn.relu, padding='VALID', name=end_point)
        end_points[end_point] = net
        # 158 x 158 x 32
        end_point = 'Conv2d_2b_3x3'
        net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME', name=end_point)
        end_points[end_point] = net
        # 158 x 158 x 64
        end_point = 'MaxPool_3a_3x3'
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3], strides=2, padding='VALID', name=end_point)
        end_points[end_point] = net
        # 78 x 78 x 64
        end_point = 'Conv2d_3b_1x1'
        net = tf.layers.conv2d(inputs=net, filters=80, kernel_size=[1, 1], activation=tf.nn.relu, padding='VALID', name=end_point)
        end_points[end_point] = net
        # 78 x 78 x 80
        end_point = 'Conv2d_4a_3x3'
        net = tf.layers.conv2d(inputs=net, filters=192, kernel_size=[3, 3], activation=tf.nn.relu, padding='VALID', name=end_point)
        end_points[end_point] = net
        # 76 x 76 x 192
        end_point = 'MaxPool_5a_3x3'
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3], strides=2, padding='VALID', name=end_point)
        end_points[end_point] = net
        # 38 x 38 x 192

        # Inception blocks
        # mixed: 38 x 38 x 256
        end_point = 'Mixed_5b'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = tf.layers.conv2d(inputs=net, filters=48, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0a_1x1')
                branch_1 = tf.layers.conv2d(inputs=branch_1, filters=64, kernel_size=[5, 5], activation=tf.nn.relu, padding='SAME', name='Conv2d_0b_5x5')
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0a_1x1')
                branch_2 = tf.layers.conv2d(inputs=branch_2, filters=96, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME', name='Conv2d_0b_3x3')
                branch_2 = tf.layers.conv2d(inputs=branch_2, filters=96, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME', name='Conv2d_0c_3x3')
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.layers.average_pooling2d(inputs=net, pool_size=[3, 3], strides=[1, 1], padding='SAME', name='AvgPool_0a_3x3')
                branch_3 = tf.layers.conv2d(inputs=branch_3, filters=32, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net

        # mixed_1: 38 x 38 x 288
        end_point = 'Mixed_5c'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = tf.layers.conv2d(inputs=net, filters=48, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0a_1x1')
                branch_1 = tf.layers.conv2d(inputs=branch_1, filters=64, kernel_size=[5, 5], activation=tf.nn.relu, padding='SAME', name='Conv2d_0b_5x5')
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0a_1x1')
                branch_2 = tf.layers.conv2d(inputs=branch_2, filters=96, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME', name='Conv2d_0b_3x3')
                branch_2 = tf.layers.conv2d(inputs=branch_2, filters=96, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME', name='Conv2d_0c_3x3')
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.layers.average_pooling2d(inputs=net, pool_size=[3, 3], strides=[1, 1], padding='SAME', name='AvgPool_0a_3x3')
                branch_3 = tf.layers.conv2d(inputs=branch_3, filters=64, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net

        # mixed_2: 38 x 38 x 288
        end_point = 'Mixed_5d'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = tf.layers.conv2d(inputs=net, filters=48, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0a_1x1')
                branch_1 = tf.layers.conv2d(inputs=branch_1, filters=64, kernel_size=[5, 5], activation=tf.nn.relu, padding='SAME', name='Conv2d_0b_5x5')
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0a_1x1')
                branch_2 = tf.layers.conv2d(inputs=branch_2, filters=96, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME', name='Conv2d_0b_3x3')
                branch_2 = tf.layers.conv2d(inputs=branch_2, filters=96, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME', name='Conv2d_0c_3x3')
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.layers.average_pooling2d(inputs=net, pool_size=[3, 3], strides=[1, 1], padding='SAME', name='AvgPool_0a_3x3')
                branch_3 = tf.layers.conv2d(inputs=branch_3, filters=64, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net

        # mixed_3: 19 x 19 x 768
        end_point = 'Mixed_6a'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = tf.layers.conv2d(inputs=net, filters=384, kernel_size=[3, 3], strides=(2,2), activation=tf.nn.relu, padding='VALID', name='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_1'):
                branch_1 = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0a_1x1')
                branch_1 = tf.layers.conv2d(inputs=branch_1, filters=96, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME', name='Conv2d_0b_3x3')
                branch_1 = tf.layers.conv2d(inputs=branch_1, filters=96, kernel_size=[3, 3], strides=(2,2), activation=tf.nn.relu, padding='VALID', name='Conv2d_0c_3x3')
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3], strides=(2, 2), padding='VALID', name='MaxPool_0a_3x3')
            net = tf.concat([branch_0, branch_1, branch_2], 3)
        end_points[end_point] = net

        # mixed4: 19 x 19 x 768
        end_point = 'Mixed_6b'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = tf.layers.conv2d(inputs=net, filters=192, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = tf.layers.conv2d(inputs=net, filters=128, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0a_1x1')
                branch_1 = tf.layers.conv2d(inputs=branch_1, filters=128, kernel_size=[1, 7], activation=tf.nn.relu, padding='SAME', name='Conv2d_0b_1x7')
                branch_1 = tf.layers.conv2d(inputs=branch_1, filters=192, kernel_size=[7, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0c_7x1')
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.layers.conv2d(inputs=net, filters=128, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0a_1x1')
                branch_2 = tf.layers.conv2d(inputs=branch_2, filters=128, kernel_size=[1, 7], activation=tf.nn.relu, padding='SAME', name='Conv2d_0b_1x7')
                branch_2 = tf.layers.conv2d(inputs=branch_2, filters=128, kernel_size=[7, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0c_7x1')
                branch_2 = tf.layers.conv2d(inputs=branch_2, filters=128, kernel_size=[1, 7], activation=tf.nn.relu, padding='SAME', name='Conv2d_0d_1x7')
                branch_2 = tf.layers.conv2d(inputs=branch_2, filters=192, kernel_size=[7, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0e_7x1')
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.layers.average_pooling2d(inputs=net, pool_size=[3, 3], strides=[1, 1], padding='SAME', name='AvgPool_0a_3x3')
                branch_3 = tf.layers.conv2d(inputs=branch_3, filters=192, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net

        # mixed_5: 19 x 19 x 768
        end_point = 'Mixed_6c'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = tf.layers.conv2d(inputs=net, filters=192, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = tf.layers.conv2d(inputs=net, filters=160, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0a_1x1')
                branch_1 = tf.layers.conv2d(inputs=branch_1, filters=160, kernel_size=[1, 7], activation=tf.nn.relu, padding='SAME', name='Conv2d_0b_1x7')
                branch_1 = tf.layers.conv2d(inputs=branch_1, filters=192, kernel_size=[7, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0c_7x1')
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.layers.conv2d(inputs=net, filters=160, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0a_1x1')
                branch_2 = tf.layers.conv2d(inputs=branch_2, filters=160, kernel_size=[1, 7], activation=tf.nn.relu, padding='SAME', name='Conv2d_0b_1x7')
                branch_2 = tf.layers.conv2d(inputs=branch_2, filters=160, kernel_size=[7, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0c_7x1')
                branch_2 = tf.layers.conv2d(inputs=branch_2, filters=160, kernel_size=[1, 7], activation=tf.nn.relu, padding='SAME', name='Conv2d_0d_1x7')
                branch_2 = tf.layers.conv2d(inputs=branch_2, filters=192, kernel_size=[7, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0e_7x1')
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.layers.average_pooling2d(inputs=net, pool_size=[3, 3], strides=[1, 1], padding='SAME', name='AvgPool_0a_3x3')
                branch_3 = tf.layers.conv2d(inputs=branch_3, filters=192, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net

        # mixed_6: 19 x 19 x 768
        end_point = 'Mixed_6d'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = tf.layers.conv2d(inputs=net, filters=192, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = tf.layers.conv2d(inputs=net, filters=160, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0a_1x1')
                branch_1 = tf.layers.conv2d(inputs=branch_1, filters=160, kernel_size=[1, 7], activation=tf.nn.relu, padding='SAME', name='Conv2d_0b_1x7')
                branch_1 = tf.layers.conv2d(inputs=branch_1, filters=192, kernel_size=[7, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0c_7x1')
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.layers.conv2d(inputs=net, filters=160, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0a_1x1')
                branch_2 = tf.layers.conv2d(inputs=branch_2, filters=160, kernel_size=[1, 7], activation=tf.nn.relu, padding='SAME', name='Conv2d_0b_1x7')
                branch_2 = tf.layers.conv2d(inputs=branch_2, filters=160, kernel_size=[7, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0c_7x1')
                branch_2 = tf.layers.conv2d(inputs=branch_2, filters=160, kernel_size=[1, 7], activation=tf.nn.relu, padding='SAME', name='Conv2d_0d_1x7')
                branch_2 = tf.layers.conv2d(inputs=branch_2, filters=192, kernel_size=[7, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0e_7x1')
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.layers.average_pooling2d(inputs=net, pool_size=[3, 3], strides=[1, 1], padding='SAME', name='AvgPool_0a_3x3')
                branch_3 = tf.layers.conv2d(inputs=branch_3, filters=192, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net

        # mixed_7: 19 x 19 x 768
        end_point = 'Mixed_6e'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = tf.layers.conv2d(inputs=net, filters=192, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = tf.layers.conv2d(inputs=net, filters=192, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0a_1x1')
                branch_1 = tf.layers.conv2d(inputs=branch_1, filters=192, kernel_size=[1, 7], activation=tf.nn.relu, padding='SAME', name='Conv2d_0b_1x7')
                branch_1 = tf.layers.conv2d(inputs=branch_1, filters=192, kernel_size=[7, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0c_7x1')
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.layers.conv2d(inputs=net, filters=192, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0a_1x1')
                branch_2 = tf.layers.conv2d(inputs=branch_2, filters=192, kernel_size=[1, 7], activation=tf.nn.relu, padding='SAME', name='Conv2d_0b_1x7')
                branch_2 = tf.layers.conv2d(inputs=branch_2, filters=192, kernel_size=[7, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0c_7x1')
                branch_2 = tf.layers.conv2d(inputs=branch_2, filters=192, kernel_size=[1, 7], activation=tf.nn.relu, padding='SAME', name='Conv2d_0d_1x7')
                branch_2 = tf.layers.conv2d(inputs=branch_2, filters=192, kernel_size=[7, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0e_7x1')
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.layers.average_pooling2d(inputs=net, pool_size=[3, 3], strides=[1, 1], padding='SAME', name='AvgPool_0a_3x3')
                branch_3 = tf.layers.conv2d(inputs=branch_3, filters=192, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net

        # mixed_8: 9 x 9 x 1280
        end_point = 'Mixed_7a'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = tf.layers.conv2d(inputs=net, filters=192, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0a_1x1')
                branch_0 = tf.layers.conv2d(inputs=branch_0, filters=320, kernel_size=[3, 3], strides=(2,2), activation=tf.nn.relu, padding='VALID', name='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_1'):
                branch_1 = tf.layers.conv2d(inputs=net, filters=192, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0a_1x1')
                branch_1 = tf.layers.conv2d(inputs=branch_1, filters=192, kernel_size=[1, 7], activation=tf.nn.relu, padding='SAME', name='Conv2d_0b_1x7')
                branch_1 = tf.layers.conv2d(inputs=branch_1, filters=192, kernel_size=[7, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0c_7x1')
                branch_1 = tf.layers.conv2d(inputs=branch_1, filters=192, kernel_size=[3, 3], strides=(2,2), activation=tf.nn.relu, padding='VALID', name='Conv2d_0d_3x3')
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3], strides=[2, 2], padding='VALID', name='MaxPool_0a_3x3')
            net = tf.concat([branch_0, branch_1, branch_2], 3)
        end_points[end_point] = net

        # mixed_9: 9 x 9 x 2048
        end_point = 'Mixed_7b'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = tf.layers.conv2d(inputs=net, filters=320, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = tf.layers.conv2d(inputs=net, filters=384, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0a_1x1')
                branch_1 = tf.concat([tf.layers.conv2d(inputs=branch_1, filters=384, kernel_size=[1, 3], activation=tf.nn.relu, padding='SAME', name='Conv2d_0b_1x3'),
                                      tf.layers.conv2d(inputs=branch_1, filters=384, kernel_size=[3, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0c_3x1')], 3)
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.layers.conv2d(inputs=net, filters=448, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0a_1x1')
                branch_2 = tf.layers.conv2d(inputs=branch_2, filters=384, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME', name='Conv2d_0b_3x3')
                branch_2 = tf.concat([tf.layers.conv2d(inputs=branch_2, filters=384, kernel_size=[1, 3], activation=tf.nn.relu, padding='SAME', name='Conv2d_0c_1x3'),
                                      tf.layers.conv2d(inputs=branch_2, filters=384, kernel_size=[3, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0d_3x1')], 3)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.layers.average_pooling2d(inputs=net, pool_size=[3, 3], strides=[1, 1], padding='SAME', name='AvgPool_0a_3x3')
                branch_3 = tf.layers.conv2d(inputs=branch_3, filters=192, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net

        # mixed_10: 9 x 9 x 2048
        end_point = 'Mixed_7c'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = tf.layers.conv2d(inputs=net, filters=320, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = tf.layers.conv2d(inputs=net, filters=384, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0a_1x1')
                branch_1 = tf.concat([tf.layers.conv2d(inputs=branch_1, filters=384, kernel_size=[1, 3], activation=tf.nn.relu, padding='SAME', name='Conv2d_0b_1x3'),
                                      tf.layers.conv2d(inputs=branch_1, filters=384, kernel_size=[3, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0c_3x1')], 3)
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.layers.conv2d(inputs=net, filters=448, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0a_1x1')
                branch_2 = tf.layers.conv2d(inputs=branch_2, filters=384, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME', name='Conv2d_0b_3x3')
                branch_2 = tf.concat([tf.layers.conv2d(inputs=branch_2, filters=384, kernel_size=[1, 3], activation=tf.nn.relu, padding='SAME', name='Conv2d_0c_1x3'),
                                      tf.layers.conv2d(inputs=branch_2, filters=384, kernel_size=[3, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0d_3x1')], 3)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.layers.average_pooling2d(inputs=net, pool_size=[3, 3], strides=[1, 1], padding='SAME', name='AvgPool_0a_3x3')
                branch_3 = tf.layers.conv2d(inputs=branch_3, filters=192, kernel_size=[1, 1], activation=tf.nn.relu, padding='SAME', name='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net

        with tf.variable_scope('logits'):
            shape = net.get_shape()
            net = tf.layers.average_pooling2d(inputs=net, pool_size=shape[1:3], strides=[1, 1], padding='VALID', name='pool')
            # 1 x 1 x 2048
            net = tf.layers.dropout(inputs=net, rate=dropout_keep_prob, training=is_training, name="dropout")
            net = tf.layers.flatten(net)
            end_points['last'] = net
            logits = tf.layers.dense(inputs=net, units=num_classes, name="logits")
            end_points['logits'] = logits

    return logits, end_points

