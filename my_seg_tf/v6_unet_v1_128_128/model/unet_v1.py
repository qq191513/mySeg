import os


import sys
sys.path.append('../')
import config as cfg
lr_init = cfg.lr_init
import tensorflow as tf
num_classes = cfg.num_classes
from model.visual_tensor import print_tensor
#正常
#交叉熵loss不可用，loss一直是成千上万，改成dice_loss

class Unet:
    def __init__(self, sess,config,is_train,size=(128, 128), l2_reg=None):
        self.sess = sess
        self.name = 'Unet'
        self.end_point = {}
        self.ckpt_dir = config.ckpt_dir
        self.is_train = is_train
        self.images = tf.placeholder(tf.float32, [None, size[0], size[1], 3])
        self.labels = tf.placeholder(tf.float32, [None, size[0], size[1], num_classes])

        pred =self.build(self.images,size, l2_reg)
        self.compute_loss(pred,self.labels)

        # self.t_vars = tf.get_collection(
        #     tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        # self.sess.run(tf.variables_initializer(self.t_vars))

        self.saver = tf.train.Saver()

        if not tf.gfile.Exists(self.ckpt_dir):
            tf.gfile.MakeDirs(self.ckpt_dir)
        self.summary_writer = tf.summary.FileWriter(self.ckpt_dir)
        self.summary_op = tf.summary.merge(self.loss_summaries)
        # self.summary_op = tf.summary.merge(self.acc_summaries)
        self.optim = tf.train.AdamOptimizer(lr_init)  # use NadamOptmizer
        self.train = self.optim.minimize(self.loss)

    def fit(self, images, labels, summary_step=-1):
        if summary_step >= 0:
          # _, loss_val,acc_val, summary_str = self.sess.run(
          #   [self.train, self.loss, self.acc,self.summary_op],
          #     {self.images:images, self.labels:labels})
          # self.summary_writer.add_summary(summary_str, summary_step)
          _,loss_val, summary_str = self.sess.run(
              [self.train, self.loss, self.summary_op],
              {self.images: images, self.labels: labels})
          # self.summary_writer.add_summary(summary_str, summary_step)

        else :
          # _, loss_val,acc_val = self.sess.run(
          #   [self.train, self.loss,self.acc],
          #     {self.images:images, self.labels:labels})
          _, loss_val = self.sess.run(
              [self.train, self.loss],
              {self.images: images, self.labels: labels})
        return loss_val

    def predict(self, images):
        result = self.sess.run([Unet.pred], {self.images:images})

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

    def predict(self, images):
        result = self.sess.run([self.pred], {self.images:images})
        return result

    def dice_coe(self,output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):

        inse = tf.reduce_sum(output * target, axis=axis)
        if loss_type == 'jaccard':
            l = tf.reduce_sum(output * output, axis=axis)
            r = tf.reduce_sum(target * target, axis=axis)
        elif loss_type == 'sorensen':
            l = tf.reduce_sum(output, axis=axis)
            r = tf.reduce_sum(target, axis=axis)
        else:
            raise Exception("Unknow loss_type")
        # old axis=[0,1,2,3]
        # dice = 2 * (inse) / (l + r)
        # epsilon = 1e-5
        # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
        # new haodong
        dice = (2. * inse + smooth) / (l + r + smooth)
        ##
        dice = tf.reduce_mean(dice, name='dice_coe')
        return dice

    def dice_coef_loss(self,y_true, y_pred, smooth=1):
        dice_loss = 1 - self.dice_coe(y_pred, y_true, axis=[1, 2, 3])
        return dice_loss

    def compute_loss(self,pred,labels):
        # labels = print_tensor(labels, 'self.labels')
        # pred = print_tensor(pred, 'pred')
        # labels =tf.reshape(labels,(-1,cfg.labels_shape[0] * cfg.labels_shape[1] * cfg.labels_shape[2]))
        # pred =tf.reshape(pred,(-1,cfg.labels_shape[0] * cfg.labels_shape[1] * cfg.labels_shape[2]))
        # ss =
        # ss = print_tensor(ss, 'ss')
        # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,
        #                                         logits=pred),axis=-1)
        self.loss = self.dice_coef_loss(labels,pred)
        # cross_entropy = tf.log(tf.clip_by_value(tf.sigmoid(cross_entropy ), 1e-8, 1.0))
        # save_tensor_to_pics(self.sess,pred,feed_dict={self.pred: labels},show='True')
        self.loss_summaries = [
          tf.summary.scalar("cross_entropy", self.loss)]



    def save(self,epoch):
        print('saving model.......')
        self.saver.save(self.sess, os.path.join(self.ckpt_dir, "model_{}.ckpt".format(epoch)))

    def restore(self,name):
        print('restoring model: {}.......'.format(os.path.join(self.ckpt_dir, name)))
        self.saver.restore(self.sess, os.path.join(self.ckpt_dir, name))


# class Model:
#     def __init__(self, inputs, outputs, teacher, is_training):
#         self.inputs = inputs
#         self.outputs = outputs
#         self.teacher = teacher
#         self.is_training = is_training