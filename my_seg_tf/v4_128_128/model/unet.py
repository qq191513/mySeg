import os

import tensorflow as tf
import sys
sys.path.append('../')
import config as cfg
lr_init = cfg.lr_init

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
        self.loss = self.compute_loss( self.labels, self.pred)

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

    def compute_loss(self, labels,pred):
        dice_loss = self.dice_coef_loss(labels, pred)
        self.loss_summaries = [
          tf.summary.scalar("dice_loss", dice_loss)]
        total_loss = dice_loss
        return total_loss




    def build(self, images):
        # with tf.variable_scope(self.name):

        conv1 = self.conv2d(images, 64, 3)
        conv1 = self.conv2d(conv1, 64, 3)
        pool1 = self.maxpooling2d(conv1,[2,2])

        conv2 = self.conv2d(pool1, 128, 3)
        conv2 = self.conv2d(conv2, 128, 3)
        pool2 = self.maxpooling2d(conv2,[2,2])

        conv3 = self.conv2d(pool2, 256, 3)
        conv3 = self.conv2d(conv3, 256, 3)
        pool3 = self.maxpooling2d(conv3,[2,2])

        conv4 = self.conv2d(pool3, 512, 3)
        conv4 = self.conv2d(conv4, 512, 3)

        up5 = tf.concat([self.conv2d_transpose(conv4,256,3), conv3], axis=3)
        conv5 = self.conv2d(up5, 256, 3)
        conv5 = self.conv2d(conv5, 256, 3)

        up6 = tf.concat([self.conv2d_transpose(conv5,256,3), conv4], axis=3)
        conv6 = self.conv2d(up6, 128, 3)
        conv6 = self.conv2d(conv6, 128, 3)

        up7 = tf.concat([self.conv2d_transpose(conv6,256,3), conv5], axis=3)
        conv7 = self.conv2d(up7, 64, 3)
        conv7 = self.conv2d(conv7, 64, 3)

        conv8 = self.conv2d(conv7, 16, 1)

        out = tf.squeeze(conv8, axis=3)  # tf.squeeze remove the dimensions of value 1
        print("shape of squeezed vector:", out.get_shape())


        return out

    def conv2d(self, x, channel, kernel, stride=1, padding="SAME",activation='relu'):
        return tf.layers.conv2d(x, channel, kernel, stride, padding, activation,kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    def maxpooling2d(self,inputs,pool_size, strides,padding='valid', data_format='channels_last',name=None):
        return tf.layers.max_pooling2d(inputs,pool_size, strides,padding=padding, data_format=data_format,name=name)

    def conv2d_transpose(self, x, channel, kernel, stride=1, padding="SAME"):
        return tf.layers.conv2d_transpose(x, channel, kernel, stride, padding,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))


    def save(self,epoch):
        print('saving model.......')
        self.saver.save(self.sess, os.path.join(self.ckpt_dir, "model_{}.ckpt".format(epoch)))

    def restore(self,name):
        print('restoring model: {}.......'.format(name))
        self.saver.restore(self.sess, os.path.join(self.ckpt_dir, name))
