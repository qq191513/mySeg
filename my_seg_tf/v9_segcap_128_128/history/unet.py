import os

import tensorflow as tf
import sys
sys.path.append('../')
import config as cfg
lr_init = cfg.lr_init
#自己写的Unet，无法解决的bug

class Unet(object):
    def __init__(self, sess, config, is_train):
        self.sess = sess
        self.name = 'Unet'
        self.mask = config.mask
        self.ckpt_dir = config.ckpt_dir
        self.is_train = is_train


        self.images = tf.placeholder(tf.float32, [config.batch_size, config.input_shape[0], config.input_shape[1], config.input_shape[2]]) #initially 512,512,3 for Gray Images
        self.labels = tf.placeholder(tf.float32, [config.batch_size, config.labels_shape[0], config.labels_shape[1], config.labels_shape[2]]) #initially 512,512, 256 for Binary Segmentation
        self.pred = self.build(self.images)


        # self.accuracy = self.compute_acc(self.recons, self.labels)
        self.loss = self.compute_loss(self.pred,self.labels)

        self.t_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.sess.run(tf.variables_initializer(self.t_vars))

        self.saver = tf.train.Saver()

        if not tf.gfile.Exists(self.ckpt_dir):
          tf.gfile.MakeDirs(self.ckpt_dir)
        self.summary_writer = tf.summary.FileWriter(self.ckpt_dir)
        self.summary_op = tf.summary.merge(self.loss_summaries)
        # self.summary_op = tf.summary.merge(self.acc_summaries)
        self.optim = tf.train.AdamOptimizer(lr_init) #use NadamOptmizer
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
          self.summary_writer.add_summary(summary_str, summary_step)


        else :
          # _, loss_val,acc_val = self.sess.run(
          #   [self.train, self.loss,self.acc],
          #     {self.images:images, self.labels:labels})
          _, loss_val = self.sess.run(
              [self.train, self.loss],
              {self.images: images, self.labels: labels})
        return loss_val

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

    def IOU_(self,y_pred, y_true):
        """Returns a (approx) IOU score
        intesection = y_pred.flatten() * y_true.flatten()
        Then, IOU = 2 * intersection / (y_pred.sum() + y_true.sum() + 1e-7) + 1e-7
        Args:
            y_pred (4-D array): (N, H, W, 1)
            y_true (4-D array): (N, H, W, 1)
        Returns:
            float: IOU score
        """
        H, W, _ = y_pred.get_shape().as_list()[1:]

        pred_flat = tf.reshape(y_pred, [-1, H * W])
        true_flat = tf.reshape(y_true, [-1, H * W])

        intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + 1e-7
        denominator = tf.reduce_sum(
            pred_flat, axis=1) + tf.reduce_sum(
            true_flat, axis=1) + 1e-7

        return tf.reduce_mean(intersection / denominator)

    def compute_loss(self, pred,labels):
        # dice_loss = self.dice_coef_loss(labels, pred)


        loss = 1-self.IOU_(pred, labels)
        self.loss_summaries = [
          tf.summary.scalar("loss", loss)]
        return loss




    def build(self, images):
        # with tf.variable_scope(self.name):

        conv1 = self.conv2d(images, 64, 3)
        conv1 = self.conv2d(conv1, 64, 3)
        pool1 = self.maxpooling2d(conv1,[2,2],strides=2)

        conv2 = self.conv2d(pool1, 128, 3)
        conv2 = self.conv2d(conv2, 128, 3)
        pool2 = self.maxpooling2d(conv2,[2,2],strides=2)

        conv3 = self.conv2d(pool2, 256, 3)
        conv3 = self.conv2d(conv3, 256, 3)
        pool3 = self.maxpooling2d(conv3,[2,2],strides=2)

        conv4 = self.conv2d(pool3, 512, 3)
        conv4 = self.conv2d(conv4, 512, 3)

        deconv_1 = self.conv2d_transpose(conv4, 256, 3)
        up5 = tf.concat([deconv_1, conv3], axis=3)
        conv5 = self.conv2d(up5, 256, 3)
        conv5 = self.conv2d(conv5, 256, 3)

        deconv_2 = self.conv2d_transpose(conv5, 128, 3)
        up6 = tf.concat([deconv_2, conv2], axis=3)
        conv6 = self.conv2d(up6, 128, 3)
        conv6 = self.conv2d(conv6, 128, 3)

        deconv_3 = self.conv2d_transpose(conv6, 64, 3)
        up7 = tf.concat([deconv_3, conv1], axis=3)
        conv7 = self.conv2d(up7, 64, 3)
        # conv7 = self.conv2d(conv7, 64, 3)

        conv8 = self.conv2d(conv7, 1, 1,activation=tf.nn.sigmoid)

        # norm_conv8 = tf.norm(conv8,axis=3)
        # out = tf.squeeze(norm_conv8, axis=3)  # tf.squeeze remove the dimensions of value 1

        # norm_conv8 = tf.expand_dims(norm_conv8,axis=-1)
        print("shape of conv8 vector:", conv8.get_shape())


        return conv8

    def conv2d(self, x, channel, kernel, stride=1, padding="SAME",activation=tf.nn.relu):

        # regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_scale)
        return tf.layers.conv2d(x, channel, kernel, stride, padding, activation=None,
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    def maxpooling2d(self,inputs,pool_size, strides,padding='SAME', data_format='channels_last',name=None):
        return tf.layers.max_pooling2d(inputs,pool_size, strides,padding=padding, data_format=data_format,name=name)

    def conv2d_transpose(self, x, channel, kernel, stride=2, padding="SAME",activation=tf.nn.relu):
        return tf.layers.conv2d_transpose(x, channel, kernel, stride, padding,activation=None,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))


    def save(self,epoch):
        print('saving model.......')
        self.saver.save(self.sess, os.path.join(self.ckpt_dir, "model_{}.ckpt".format(epoch)))

    def restore(self,name):
        print('restoring model: {}.......'.format(name))
        self.saver.restore(self.sess, os.path.join(self.ckpt_dir, name))
