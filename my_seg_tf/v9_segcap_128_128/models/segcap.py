import sys
sys.path.append('../')
from choice import cfg

lr_init = cfg.lr_init
import tensorflow as tf
num_classes = cfg.num_classes
# from tools.development_kit import print_tensor


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


# def pool(inputs):
#     pooled = tf.layers.max_pooling2d(inputs=inputs, pool_size=[2, 2], strides=2)
#     return pooled
#
# def conv_transpose(inputs, filters, l2_reg_scale=None, batchnorm_istraining=None):
#     if l2_reg_scale is None:
#         regularizer = None
#     else:
#         regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_scale)
#     conved = tf.layers.conv2d_transpose(
#         inputs=inputs,
#         filters=filters,
#         strides=[2, 2],
#         kernel_size=[3, 3],
#         padding='same',
#         activation=tf.nn.relu,
#         kernel_regularizer=regularizer
#     )
#     if batchnorm_istraining is not None:
#         conved = bn(conved, batchnorm_istraining)
#     return conved

def conv2d(x, channel, kernel, stride=1, padding="SAME"):
    return tf.layers.conv2d(x, channel, kernel, stride, padding,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))


def conv2d_transpose(x, channel, kernel, stride=1, padding="SAME"):
    return tf.layers.conv2d_transpose(x, channel, kernel, stride, padding,
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

def squash(p):
    p_norm_sq = tf.reduce_sum(tf.square(p), axis=-1, keep_dims=True)
    p_norm = tf.sqrt(p_norm_sq + 1e-9)
    v = p_norm_sq / (1. + p_norm_sq) * p / p_norm
    return v

def compute_vector_length(x):
    return tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keep_dims=True) + 1e-9)

def capsule(u, op, k, s, t, z, routing):
    t_1, z_1 = t, z

    shape = u.get_shape() #tf.shape(u)
    N = shape[0]
    t_0 = shape[3]
    z_0 = shape[4]

    u_t_list = [tf.squeeze(u_t, axis=3) for u_t in tf.split(u, t_0, axis=3)]
    u_hat_t_list = []
    for u_t in u_t_list: # u_t: [N, H_0, W_0, z_0]
      if op == "conv":
        u_hat_t = conv2d(u_t, t_1*z_1, k, s)
      elif op == "deconv":
        u_hat_t = conv2d_transpose(u_t, t_1*z_1, k, s)
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
        v = squash(p)
        b_t_list_ = []
        for b_t, u_hat_t in zip(b_t_list, u_hat_t_list_):
          # b_t     : [N, H_1, W_1, t_1]
          # u_hat_t : [N, H_1, W_1, t_1, z_1]
          # v       : [N, H_1, W_1, t_1, z_1]
          b_t_list_.append(b_t + tf.reduce_sum(u_hat_t * v, axis=4))
        b_t_list = b_t_list_
    v = squash(p)
    return v



def my_segcap(images,is_train,size, l2_reg):
    is_training =True
    start_s = 2
    keep_prob = 0.8

    # 1  (128 -> 128)
    conv1 =conv(images, filters=16, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
    conv_prime = tf.expand_dims(conv1, axis=3)  # [N, H, W, t=1, z]

    # 1/2  (128 -> 64)
    multiple = 1
    cap1_1 = capsule(conv_prime, "conv", k=3, s=1, t=start_s*multiple, z=32, routing=2)
    cap1_2 = capsule(cap1_1, "conv", k=3, s=2, t=start_s*multiple, z=32, routing=2)
    skip1 = cap1_1

    # 1/4  (64 -> 32)
    multiple = 2
    cap2_1 = capsule(cap1_2, "conv", k=3, s=1, t=start_s*multiple, z=32, routing=2)
    cap2_2 = capsule(cap2_1, "conv", k=3, s=2, t=start_s*multiple, z=32, routing=2)
    skip2 = cap2_1

    # 1/8  (32 -> 16)
    multiple = 4
    cap3_1 = capsule(cap2_2, "conv", k=3, s=1, t=start_s*multiple, z=32, routing=2)
    cap3_2 = capsule(cap3_1, "conv", k=3, s=2, t=start_s*multiple, z=32, routing=2)
    skip3 = cap3_1

    #middle  (16 -> 16)
    multiple = 8
    cap_m_1 = capsule(cap3_2, "conv", k=3, s=1, t=start_s*multiple, z=32, routing=2)
    cap_m_2 = capsule(cap_m_1,"conv", k=3, s=1, t=start_s*multiple, z=32, routing=2)

    # 1/8  (16 -> 32)
    multiple = 4
    u_cap1_1 = capsule(cap_m_2, "deconv", k=3, s=2, t=start_s*multiple, z=32, routing=2)
    u_cap_concat_1 = tf.concat([u_cap1_1, skip3], axis=3)
    u_cap1_2 = capsule(u_cap_concat_1, "conv", k=3, s=1, t=start_s*multiple, z=32, routing=2)
    u_cap1_3 = capsule(u_cap1_2, "conv", k=3, s=1, t=start_s*multiple, z=32, routing=2)

    # 1/4  (32 -> 64)
    multiple = 2
    u_cap2_1 = capsule(u_cap1_3, "deconv", k=3, s=2, t=start_s*multiple, z=32, routing=2)
    u_cap_concat_2 = tf.concat([u_cap2_1, skip2], axis=3)
    u_cap2_2 = capsule(u_cap_concat_2, "conv", k=3, s=1, t=start_s*multiple, z=32, routing=2)
    u_cap2_3 = capsule(u_cap2_2, "conv", k=3, s=1, t=start_s*multiple, z=32, routing=2)

    # 1/2  (64 -> 128)
    multiple = 2
    u_cap3_1 = capsule(u_cap2_3, "deconv", k=3, s=2, t=start_s*multiple, z=32, routing=2)
    u_cap_concat_3 = tf.concat([u_cap3_1, skip1], axis=3)
    u_cap3_2 = capsule(u_cap_concat_3, "conv", k=3, s=1, t=start_s*multiple, z=32, routing=2)


    # 1   (128 -> 128)
    multiple =1
    cap_out_1 = capsule(u_cap3_2, "conv", k=3, s=1, t=start_s*multiple, z=32, routing=2)
    cap_out_2 = capsule(cap_out_1, "conv", k=3, s=1, t=1, z=1, routing=2)
    cap_out_3 = tf.squeeze(cap_out_2, axis=3)
    return cap_out_3

# def my_unet(images,is_train,size, l2_reg):
#     # self.labels = print_tensor(self.labels,'pool2')
#     # 无论如何都要设置成True，否则没有batchnorm。最终测试输出不正确
#     is_training =True
#
#     conv1_1 =conv(images, filters=64, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
#     conv1_2 = conv(conv1_1, filters=64, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
#     pool1 = pool(conv1_2)
#
#
#     # 1/2, 1/2, 64
#     conv2_1 = conv(pool1, filters=128, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
#     conv2_2 = conv(conv2_1, filters=128, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
#     pool2 = pool(conv2_2)
#     # pool2 = print_tensor(pool2,'pool2')
#
#     # 1/4, 1/4, 128
#     conv3_1 = conv(pool2, filters=256, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
#     conv3_2 = conv(conv3_1, filters=256, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
#     pool3 = pool(conv3_2)
#
#     # pool3 = print_tensor(pool3,'pool3')
#     # 1/8, 1/8, 256
#     conv4_1 = conv(pool3, filters=512, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
#     conv4_2 = conv(conv4_1, filters=512, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
#     pool4 = pool(conv4_2)
#     # pool4 = print_tensor(pool4,'pool4')
#
#     # 1/16, 1/16, 512
#     conv5_1 = conv(pool4, filters=1024, l2_reg_scale=l2_reg)
#     conv5_2 = conv(conv5_1, filters=1024, l2_reg_scale=l2_reg)
#     # conv4_2 = print_tensor(conv4_2,'conv4_2')
#     AAA= conv_transpose(conv5_2, filters=512, l2_reg_scale=None,batchnorm_istraining=is_training)
#     # AAA = print_tensor(AAA,'AAA ')
#     concated1 = tf.concat([AAA, conv4_2], axis=3)
#     # concated1 = print_tensor(concated1,'concated1')
#
#
#     conv_up1_1 = conv(concated1, filters=512, l2_reg_scale=l2_reg)
#     conv_up1_2 = conv(conv_up1_1, filters=512, l2_reg_scale=l2_reg)
#     concated2 = tf.concat([conv_transpose(conv_up1_2, filters=256, l2_reg_scale=l2_reg,batchnorm_istraining=is_training), conv3_2], axis=3)
#
#     conv_up2_1 = conv(concated2, filters=256, l2_reg_scale=l2_reg)
#     conv_up2_2 = conv(conv_up2_1, filters=256, l2_reg_scale=l2_reg)
#     concated3 = tf.concat([conv_transpose(conv_up2_2, filters=128, l2_reg_scale=l2_reg,batchnorm_istraining=is_training), conv2_2], axis=3)
#
#     conv_up3_1 = conv(concated3, filters=128, l2_reg_scale=l2_reg)
#     conv_up3_2 = conv(conv_up3_1, filters=128, l2_reg_scale=l2_reg)
#     concated4 = tf.concat([conv_transpose(conv_up3_2, filters=64, l2_reg_scale=l2_reg,batchnorm_istraining=is_training), conv1_2], axis=3)
#
#     conv_up4_1 = conv(concated4, filters=64, l2_reg_scale=l2_reg,batchnorm_istraining=is_training)
#     conv_up4_2 = conv(conv_up4_1, filters=64, l2_reg_scale=l2_reg,batchnorm_istraining=is_training)
#     pred = conv(conv_up4_2, filters=num_classes, kernel_size=[1, 1], activation=tf.nn.sigmoid, batchnorm_istraining=is_training)
#
#     # self.pred = print_tensor(self.pred,'pred')
#     return pred
