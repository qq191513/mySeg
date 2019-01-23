import sys
sys.path.append('../')
import config.config_unet as cfg
lr_init = cfg.lr_init
import tensorflow as tf
num_classes = cfg.num_classes


#交叉熵loss不可用，loss一直是成千上万，改成dice_loss

def bn(inputs, is_training):
    normalized = tf.layers.batch_normalization(
        inputs=inputs,
        axis=-1,
        momentum=0.9,
        epsilon=0.001,
        center=True,
        scale=True,
        training=is_training,
    )
    return normalized


def conv(inputs, filters, kernel_size=[3, 3], activation=tf.nn.relu, l2_reg_scale=None, batchnorm_istraining=None):
    if l2_reg_scale is None:
        regularizer = None
    else:
        regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_scale)
    conved = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        activation=activation,
        kernel_regularizer=regularizer
    )
    if batchnorm_istraining is not None:
        conved = bn(conved, batchnorm_istraining)

    return conved

def residual_block(inputs, filters, kernel_size=[3, 3], activation=tf.nn.relu, l2_reg_scale=None, batchnorm_istraining=None):
    if l2_reg_scale is None:
        regularizer = None
    else:
        regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_scale)
    if batchnorm_istraining is not None:
        bn_inputs = bn(inputs, batchnorm_istraining)
    conved_1 = tf.layers.conv2d(
        inputs=bn_inputs,
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        activation=None,
        kernel_regularizer=regularizer
    )
    conved_2 = tf.layers.conv2d(
        inputs=conved_1,
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        activation=activation,
        kernel_regularizer=regularizer
    )
    add_c= conved_2 + bn_inputs
    if batchnorm_istraining is not None:
        add_c = bn(add_c, batchnorm_istraining)
    return add_c

def pool(inputs):
    pooled = tf.layers.max_pooling2d(inputs=inputs, pool_size=[2, 2], strides=2)
    return pooled

def conv_transpose(inputs, filters, l2_reg_scale=None, batchnorm_istraining=None):
    if l2_reg_scale is None:
        regularizer = None
    else:
        regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_scale)
    conved = tf.layers.conv2d_transpose(
        inputs=inputs,
        filters=filters,
        strides=[2, 2],
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu,
        kernel_regularizer=regularizer
    )
    if batchnorm_istraining is not None:
        conved = bn(conved, batchnorm_istraining)
    return conved

def my_residual_unet(images,is_train,size, l2_reg):
    is_training = True
    start_filters = 32
    keep_prob = 0.8

    multiple = 1
    conv1_1 =conv(images, filters=start_filters*multiple, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
    conv1_2 = residual_block(conv1_1, filters=start_filters*multiple, kernel_size=[3, 3], activation=tf.nn.relu, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
    conv1_3 = residual_block(conv1_2, filters=start_filters*multiple, kernel_size=[3, 3], activation=tf.nn.relu, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
    # （128->64）
    pool1_1 = pool(conv1_3)
    dropout_1 = tf.nn.dropout(pool1_1, keep_prob)

    multiple = 2
    conv2_1 =conv(dropout_1, filters=start_filters*multiple, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
    conv2_2 = residual_block(conv2_1, filters=start_filters*multiple, kernel_size=[3, 3], activation=tf.nn.relu, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
    conv2_3 = residual_block(conv2_2, filters=start_filters*multiple, kernel_size=[3, 3], activation=tf.nn.relu, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
    # （64->32）
    pool2_1 = pool(conv2_3)
    dropout_2 = tf.nn.dropout(pool2_1, keep_prob)

    multiple = 4
    conv3_1 =conv(dropout_2, filters=start_filters*multiple, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
    conv3_2 = residual_block(conv3_1, filters=start_filters*multiple, kernel_size=[3, 3], activation=tf.nn.relu, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
    conv3_3 = residual_block(conv3_2, filters=start_filters*multiple, kernel_size=[3, 3], activation=tf.nn.relu, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
    # （32->16）
    pool3_1 = pool(conv3_3)
    dropout_3 = tf.nn.dropout(pool3_1, keep_prob)

    # Middle （16*16）
    multiple = 8
    convm_1 =conv(dropout_3, filters=start_filters*multiple, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
    convm_2 = residual_block(convm_1, filters=start_filters*multiple, kernel_size=[3, 3], activation=tf.nn.relu, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
    convm_3 = residual_block(convm_2, filters=start_filters*multiple, kernel_size=[3, 3], activation=tf.nn.relu, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)

    # （16->32)
    multiple = 4
    deconv_1= conv_transpose(convm_3, filters=start_filters*multiple, l2_reg_scale=None,batchnorm_istraining=is_training)
    concated_1 = tf.concat([deconv_1, conv3_3], axis=3)
    u_dropout_1 = tf.nn.dropout(concated_1, keep_prob)
    u_conv_1_1 =conv(u_dropout_1, filters=start_filters*multiple, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
    u_conv_1_2 = residual_block(u_conv_1_1, filters=start_filters*multiple, kernel_size=[3, 3], activation=tf.nn.relu, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
    u_conv_1_3 = residual_block(u_conv_1_2, filters=start_filters*multiple, kernel_size=[3, 3], activation=tf.nn.relu, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)

    # （32->64)
    multiple=2
    deconv_2 = conv_transpose(u_conv_1_3, filters=start_filters*multiple, l2_reg_scale=None, batchnorm_istraining=is_training)
    concated_2 = tf.concat([deconv_2, conv2_3], axis=3)
    u_dropout_2 = tf.nn.dropout(concated_2, keep_prob)
    u_conv_2_1 = conv(u_dropout_2, filters=start_filters*multiple, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
    u_conv_2_2 = residual_block(u_conv_2_1, filters=start_filters*multiple, kernel_size=[3, 3], activation=tf.nn.relu,
                                l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
    u_conv_2_3 = residual_block(u_conv_2_2, filters=start_filters*multiple, kernel_size=[3, 3], activation=tf.nn.relu,
                                l2_reg_scale=l2_reg, batchnorm_istraining=is_training)

    # （64->128)
    multiple=1
    u_conv_3_3 = conv_transpose(u_conv_2_3, filters=start_filters*multiple, l2_reg_scale=None, batchnorm_istraining=is_training)
    concated_3 = tf.concat([u_conv_3_3, conv1_3], axis=3)
    u_dropout_3 = tf.nn.dropout(concated_3, keep_prob)
    u_conv_3_1 = conv(u_dropout_3, filters=start_filters*multiple, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
    u_conv_3_2 = residual_block(u_conv_3_1, filters=start_filters*multiple, kernel_size=[3, 3], activation=tf.nn.relu,
                                l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
    u_conv_3_3 = residual_block(u_conv_3_2, filters=start_filters*multiple, kernel_size=[3, 3], activation=tf.nn.relu,
                                l2_reg_scale=l2_reg, batchnorm_istraining=is_training)

    output_layer = conv(u_conv_3_3, filters=1,kernel_size=[1, 1],activation=tf.nn.sigmoid ,l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
    return output_layer
