import os


import sys
sys.path.append('../')
import config as cfg
lr_init = cfg.lr_init
import tensorflow as tf
num_classes = cfg.num_classes
from tool.visual_tool import print_tensor
#正常
#交叉熵loss不可用，loss一直是成千上万，改成dice_loss

class Unet:
    def __init__(self, sess,config,is_train,size=(128, 128), l2_reg=None):
        self.sess = sess
        self.name = 'Unet'
        self.end_point = {}
        self.ckpt_dir = config.ckpt_dir
        self.is_train = is_train
        self.images = tf.placeholder(tf.float32, [None,size[0], size[1], 3])
        self.labels = tf.placeholder(tf.float32, [None,size[0], size[1], num_classes])

        pred =self.build(self.images,size, l2_reg)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()



    def predict(self,images):
        result = self.sess.run([self.pred], {self.images:images})
        return result

    def build(self,images,size, l2_reg):


        # self.labels = print_tensor(self.labels,'pool2')
        is_training =self.is_train

        # 1, 1, 3

        conv1_1 = Unet.conv(images, filters=64, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        conv1_2 = Unet.conv(conv1_1, filters=64, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        pool1 = Unet.pool(conv1_2)


        # 1/2, 1/2, 64
        conv2_1 = Unet.conv(pool1, filters=128, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        conv2_2 = Unet.conv(conv2_1, filters=128, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        pool2 = Unet.pool(conv2_2)
        # pool2 = print_tensor(pool2,'pool2')

        # 1/4, 1/4, 128
        conv3_1 = Unet.conv(pool2, filters=256, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        conv3_2 = Unet.conv(conv3_1, filters=256, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        pool3 = Unet.pool(conv3_2)

        # pool3 = print_tensor(pool3,'pool3')
        # 1/8, 1/8, 256
        conv4_1 = Unet.conv(pool3, filters=512, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        conv4_2 = Unet.conv(conv4_1, filters=512, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        pool4 = Unet.pool(conv4_2)
        # pool4 = print_tensor(pool4,'pool4')

        # 1/16, 1/16, 512
        conv5_1 = Unet.conv(pool4, filters=1024, l2_reg_scale=l2_reg)
        conv5_2 = Unet.conv(conv5_1, filters=1024, l2_reg_scale=l2_reg)
        # conv4_2 = print_tensor(conv4_2,'conv4_2')
        AAA= Unet.conv_transpose(conv5_2, filters=512, l2_reg_scale=None,batchnorm_istraining=is_training)
        # AAA = print_tensor(AAA,'AAA ')
        concated1 = tf.concat([AAA, conv4_2], axis=3)
        # concated1 = print_tensor(concated1,'concated1')


        conv_up1_1 = Unet.conv(concated1, filters=512, l2_reg_scale=l2_reg)
        conv_up1_2 = Unet.conv(conv_up1_1, filters=512, l2_reg_scale=l2_reg)
        concated2 = tf.concat([Unet.conv_transpose(conv_up1_2, filters=256, l2_reg_scale=l2_reg,batchnorm_istraining=is_training), conv3_2], axis=3)
        # concated2 = print_tensor(concated2,'concated2')


        conv_up2_1 = Unet.conv(concated2, filters=256, l2_reg_scale=l2_reg)
        conv_up2_2 = Unet.conv(conv_up2_1, filters=256, l2_reg_scale=l2_reg)
        concated3 = tf.concat([Unet.conv_transpose(conv_up2_2, filters=128, l2_reg_scale=l2_reg,batchnorm_istraining=is_training), conv2_2], axis=3)

        conv_up3_1 = Unet.conv(concated3, filters=128, l2_reg_scale=l2_reg)
        conv_up3_2 = Unet.conv(conv_up3_1, filters=128, l2_reg_scale=l2_reg)
        concated4 = tf.concat([Unet.conv_transpose(conv_up3_2, filters=64, l2_reg_scale=l2_reg,batchnorm_istraining=is_training), conv1_2], axis=3)

        conv_up4_1 = Unet.conv(concated4, filters=64, l2_reg_scale=l2_reg,batchnorm_istraining=is_training)
        conv_up4_2 = Unet.conv(conv_up4_1, filters=64, l2_reg_scale=l2_reg,batchnorm_istraining=is_training)
        self.pred = Unet.conv(conv_up4_2, filters=num_classes, kernel_size=[1, 1], activation=tf.nn.sigmoid, batchnorm_istraining=is_training)

        # self.pred = print_tensor(self.pred,'pred')
        return self.pred

        # return Model(self.images, self.pred, self.labels, is_training)


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
            conved = Unet.bn(conved, batchnorm_istraining)

        return conved


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
            conved = Unet.bn(conved, batchnorm_istraining)
        return conved


    def restore(self,name):
        print('restoring model: {} .......'.format(os.path.join(self.ckpt_dir, name)))
        self.saver.restore(self.sess, os.path.join(self.ckpt_dir, name))
