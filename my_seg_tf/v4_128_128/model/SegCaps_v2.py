import os

import tensorflow as tf


class SegCaps(object):
    def __init__(self, sess, config, is_train):
        self.sess = sess
        self.name = 'SegCaps'
        self.mask = config.mask
        self.ckpt_dir = config.ckpt_dir
        self.is_train = is_train


        self.images = tf.placeholder(tf.float32, [config.batch_size, config.input_shape[0], config.input_shape[1], config.input_shape[2]]) #initially 512,512,3 for Gray Images
        self.labels = tf.placeholder(tf.float32, [config.batch_size, config.labels_shape[0], config.labels_shape[1], config.labels_shape[2]]) #initially 512,512, 256 for Binary Segmentation
        self.v_lens, self.recons = self.build(self.images)
        # TODO : result vector generation
        self.result = tf.round(self.v_lens)
        # self.accuracy = self.compute_acc(self.recons, self.labels)
        self.loss = self.compute_loss(self.v_lens, self.recons, self.images, self.labels, self.mask)

        self.t_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.sess.run(tf.variables_initializer(self.t_vars))

        self.saver = tf.train.Saver()

        if not tf.gfile.Exists(self.ckpt_dir):
          tf.gfile.MakeDirs(self.ckpt_dir)
        self.summary_writer = tf.summary.FileWriter(self.ckpt_dir)
        self.summary_op = tf.summary.merge(self.loss_summaries)
        # self.summary_op = tf.summary.merge(self.acc_summaries)
        self.optim = tf.train.AdamOptimizer(0.01) #use NadamOptmizer


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
        result,recons = self.sess.run([self.result,self.recons], {self.images:images})
        return result,recons

    def compute_loss(self, v_lens, recons, images, labels, mask=True):
        class_loss = tf.reduce_mean(
          labels * tf.square(tf.maximum(0., 0.9 - v_lens)) +
            0.5 * (1. - labels) * tf.square(tf.maximum(0., v_lens - 0.1)))
        if mask:
          recon_loss = tf.reduce_mean(tf.square((images - recons) * labels))
        else:
          recon_loss = tf.reduce_mean(tf.square((images - recons)))
        total_loss = class_loss + 0.0005 * recon_loss

        self.loss_summaries = [
          tf.summary.scalar("class_loss", class_loss),
          tf.summary.scalar("recon_loss", recon_loss),
          tf.summary.scalar("total_loss", total_loss)]


        return total_loss



    # def compute_acc(self,logits,labels):
    #     labels_argmax = tf.argmax(labels, axis=-1, name='labels_argmax', output_type=tf.int64)
    #
    #     softmax_output = tf.nn.softmax(logits, name='softmax_output')
    #
    #     predictions_argmax = tf.argmax(softmax_output, axis=-1, name='predictions_argmax', output_type=tf.int64)
    #     acc_value, acc_update_op = tf.metrics.accuracy(labels=labels_argmax,
    #                                                    predictions=predictions_argmax)
    #     acc_value = tf.identity(acc_value, name='acc_value')
    #
    #     return acc_update_op

    def build(self, images):
        # with tf.variable_scope(self.name):
        x = images
        x = self.conv2d(x, 64, 5)
        x = tf.expand_dims(x, axis=3)  # [N, H, W, t=1, z]
        skip1 = x

        # 1/2
        brach_0_0_x = self.capsule(x, "conv", k=3, s=2, t=2, z=64, routing=3)
        brach_0_1_x = self.capsule(x, "conv", k=1, s=2, t=2, z=64, routing=3)
        x = tf.concat([brach_0_0_x, brach_0_1_x], axis=3)

        skip2 = x

        # 1/4
        brach_1_0_x = self.capsule(x, "conv", k=3, s=2, t=4, z=64, routing=3)
        brach_1_1_x = self.capsule(x, "conv", k=1, s=2, t=4, z=64, routing=3)
        x = tf.concat([brach_1_0_x, brach_1_1_x], axis=3)
        skip3 = x

        # 1/8
        brach_2_0_x = self.capsule(x, "conv", k=3, s=2, t=8, z=32, routing=3)
        brach_2_1_x = self.capsule(x, "conv", k=1, s=2, t=8, z=32, routing=3)
        x = tf.concat([brach_2_0_x, brach_2_1_x], axis=3)

        # 1/4
        x_0 = self.capsule(x, "deconv", k=1, s=2, t=4, z=64, routing=3)
        x_1 = self.capsule(x, "deconv", k=3, s=2, t=4, z=64, routing=3)
        x = tf.concat([x_0,x_1, skip3], axis=3)
        x = self.capsule(x, "conv", k=3, s=1, t=4, z=32, routing=3)

        # 1/2
        x_0 = self.capsule(x, "deconv", k=1, s=2, t=2, z=64, routing=3)
        x_1 = self.capsule(x, "deconv", k=3, s=2, t=2, z=64, routing=3)
        x = tf.concat([x_0,x_1, skip2], axis=3)
        x = self.capsule(x, "conv", k=3, s=1, t=4, z=16, routing=3)

        # 1
        x_0 = self.capsule(x, "deconv", k=1, s=2, t=1, z=64, routing=3)
        x_1 = self.capsule(x, "deconv", k=3, s=2, t=1, z=64, routing=3)
        x = tf.concat([x_0,x_1, skip1], axis=3)
        x = self.capsule(x, "conv", k=1, s=1, t=1, z=16, routing=3)

        squeezed_x = tf.squeeze(x, axis=3)  # tf.squeeze remove the dimensions of value 1
        print("shape of squeezed vector:", x.get_shape())

        # 1. compute length of vector
        v_lens = self.compute_vector_length(squeezed_x)

        # 2. Get masked reconstruction
        x = self.conv2d(squeezed_x, 64, 1)
        x = self.conv2d(x, 128, 1)
        recons = self.conv2d(x, images.get_shape()[-1], 1)

        return v_lens, recons

    def conv2d(self, x, channel, kernel, stride=1, padding="SAME"):
        return tf.layers.conv2d(x, channel, kernel, stride, padding, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    def conv2d_transpose(self, x, channel, kernel, stride=1, padding="SAME"):
        return tf.layers.conv2d_transpose(x, channel, kernel, stride, padding,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    def capsule(self, u, op, k, s, t, z, routing):

        t_1, z_1 = t, z

        shape = u.get_shape() #tf.shape(u)
        N = shape[0]
        t_0 = shape[3]
        z_0 = shape[4]

        u_t_list = [tf.squeeze(u_t, axis=3) for u_t in tf.split(u, t_0, axis=3)]
        u_hat_t_list = []
        for u_t in u_t_list: # u_t: [N, H_0, W_0, z_0]
          if op == "conv":
            u_hat_t = self.conv2d(u_t, t_1*z_1, k, s)
          elif op == "deconv":
            u_hat_t = self.conv2d_transpose(u_t, t_1*z_1, k, s)
          else:
            raise ValueError("Wrong type of operation for capsule")

          shape = u_hat_t.get_shape() #tf.shape(u)
          H_1 = shape[1]
          W_1 = shape[2]
          u_hat_t = tf.reshape(u_hat_t, [N, H_1, W_1, t_1, z_1])
          u_hat_t_list.append(u_hat_t)

        one_kernel = tf.ones([k, k, t_1, 1])
        b = tf.zeros([N, H_1, W_1, t_0, t_1])
        b_t_list = [tf.squeeze(b_t, axis=3) for b_t in tf.split(b, t_0, axis=3)]
        #动态路不需要梯度更新
        u_hat_t_list_sg = [tf.stop_gradient(u_hat_t) for u_hat_t in u_hat_t_list]
        for d in range(routing):
          if d < routing - 1:
            u_hat_t_list_ = u_hat_t_list_sg
          else:
            u_hat_t_list_ = u_hat_t_list

          r_t_mul_u_hat_t_list = []
          for b_t, u_hat_t in zip(b_t_list, u_hat_t_list_):
                # routing softmax
                b_t_max = tf.nn.max_pool(b_t, [1, k, k, 1], [1, 1, 1, 1], "SAME")
                b_t_max = tf.reduce_max(b_t_max, axis=3, keep_dims=True)
                c_t = tf.exp(b_t - b_t_max) # [N, H_1, W_1, t_1]
                sum_c_t = tf.nn.conv2d(c_t, one_kernel, [1, 1, 1, 1], "SAME") # [... , 1]

                r_t = c_t / sum_c_t # [N, H_1, W_1, t_1]
                r_t = tf.expand_dims(r_t, axis=4) # [N, H_1, W_1, t_1, 1]
                r_t_mul_u_hat_t_list.append(r_t * u_hat_t) # [N, H_1, W_1, t_1, z_1]

          p = tf.add_n(r_t_mul_u_hat_t_list) # [N, H_1, W_1, t_1, z_1]


          if d < routing - 1:
            v = self.squash(p)
            b_t_list_ = []
            for b_t, u_hat_t in zip(b_t_list, u_hat_t_list_):
              # b_t     : [N, H_1, W_1, t_1]
              # u_hat_t : [N, H_1, W_1, t_1, z_1]
              # v       : [N, H_1, W_1, t_1, z_1]
              b_t_list_.append(b_t + tf.reduce_sum(u_hat_t * v, axis=4))
            b_t_list = b_t_list_
        v = self.squash(p)
        return v

    def squash(self, p):
        p_norm_sq = tf.reduce_sum(tf.square(p), axis=-1, keep_dims=True)
        p_norm = tf.sqrt(p_norm_sq + 1e-9)
        v = p_norm_sq / (1. + p_norm_sq) * p / p_norm
        return v

    def compute_vector_length(self, x):
        return tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keep_dims=True) + 1e-9)

    def save(self,epoch):
        print('saving model.......')
        self.saver.save(self.sess, os.path.join(self.ckpt_dir, "model_{}.ckpt".format(epoch)))

    def restore(self,name):
        print('restoring model: {}.......'.format(name))
        self.saver.restore(self.sess, os.path.join(self.ckpt_dir, name))
