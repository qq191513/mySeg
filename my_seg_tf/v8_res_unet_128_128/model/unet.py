import sys
sys.path.append('../')
import tools.config.config_unet as cfg
lr_init = cfg.lr_init
import tensorflow as tf
num_classes = cfg.num_classes
# from tools.development_kit import print_tensor
#正常
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

def my_unet(images,is_train,size, l2_reg):
    # self.labels = print_tensor(self.labels,'pool2')
    # 无论如何都要设置成True，否则没有batchnorm。最终测试输出不正确
    is_training =True

    conv1_1 =conv(images, filters=64, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
    conv1_2 = conv(conv1_1, filters=64, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
    pool1 = pool(conv1_2)


    # 1/2, 1/2, 64
    conv2_1 = conv(pool1, filters=128, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
    conv2_2 = conv(conv2_1, filters=128, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
    pool2 = pool(conv2_2)
    # pool2 = print_tensor(pool2,'pool2')

    # 1/4, 1/4, 128
    conv3_1 = conv(pool2, filters=256, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
    conv3_2 = conv(conv3_1, filters=256, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
    pool3 = pool(conv3_2)

    # pool3 = print_tensor(pool3,'pool3')
    # 1/8, 1/8, 256
    conv4_1 = conv(pool3, filters=512, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
    conv4_2 = conv(conv4_1, filters=512, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
    pool4 = pool(conv4_2)
    # pool4 = print_tensor(pool4,'pool4')

    # 1/16, 1/16, 512
    conv5_1 = conv(pool4, filters=1024, l2_reg_scale=l2_reg)
    conv5_2 = conv(conv5_1, filters=1024, l2_reg_scale=l2_reg)
    # conv4_2 = print_tensor(conv4_2,'conv4_2')
    AAA= conv_transpose(conv5_2, filters=512, l2_reg_scale=None,batchnorm_istraining=is_training)
    # AAA = print_tensor(AAA,'AAA ')
    concated1 = tf.concat([AAA, conv4_2], axis=3)
    # concated1 = print_tensor(concated1,'concated1')


    conv_up1_1 = conv(concated1, filters=512, l2_reg_scale=l2_reg)
    conv_up1_2 = conv(conv_up1_1, filters=512, l2_reg_scale=l2_reg)
    concated2 = tf.concat([conv_transpose(conv_up1_2, filters=256, l2_reg_scale=l2_reg,batchnorm_istraining=is_training), conv3_2], axis=3)

    conv_up2_1 = conv(concated2, filters=256, l2_reg_scale=l2_reg)
    conv_up2_2 = conv(conv_up2_1, filters=256, l2_reg_scale=l2_reg)
    concated3 = tf.concat([conv_transpose(conv_up2_2, filters=128, l2_reg_scale=l2_reg,batchnorm_istraining=is_training), conv2_2], axis=3)

    conv_up3_1 = conv(concated3, filters=128, l2_reg_scale=l2_reg)
    conv_up3_2 = conv(conv_up3_1, filters=128, l2_reg_scale=l2_reg)
    concated4 = tf.concat([conv_transpose(conv_up3_2, filters=64, l2_reg_scale=l2_reg,batchnorm_istraining=is_training), conv1_2], axis=3)

    conv_up4_1 = conv(concated4, filters=64, l2_reg_scale=l2_reg,batchnorm_istraining=is_training)
    conv_up4_2 = conv(conv_up4_1, filters=64, l2_reg_scale=l2_reg,batchnorm_istraining=is_training)
    pred = conv(conv_up4_2, filters=num_classes, kernel_size=[1, 1], activation=tf.nn.sigmoid, batchnorm_istraining=is_training)

    # self.pred = print_tensor(self.pred,'pred')
    return pred
