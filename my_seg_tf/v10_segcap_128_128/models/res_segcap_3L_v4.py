import sys
sys.path.append('../')
from choice import cfg

import tensorflow as tf
num_classes = cfg.num_classes
from tools.development_kit import print_tensor


# mini u型 两层 减少神经元 32减少到8


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


def conv(inputs, filters, kernel_size=[3, 3], strides=1,activation=tf.nn.relu, l2_reg_scale=None, batchnorm_istraining=None):
    if l2_reg_scale is None:
        regularizer = None
    else:
        regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_scale)
    conved = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
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
# def conv_transpose(inputs, filters, strides=1,l2_reg_scale=None, batchnorm_istraining=None):
#     if l2_reg_scale is None:
#         regularizer = None
#     else:
#         regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_scale)
#     conved = tf.layers.conv2d_transpose(
#         inputs=inputs,
#         filters=filters,
#         strides=strides,
#         kernel_size=[3, 3],
#         padding='same',
#         activation=tf.nn.relu,
#         kernel_regularizer=regularizer
#     )
#     if batchnorm_istraining is not None:
#         conved = bn(conved, batchnorm_istraining)
#     return conved
# def conv2d_transpose(x, channel, kernel, strides=1, padding="SAME"):
#     return tf.layers.conv2d_transpose(x, channel, kernel, strides, padding,
#                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

# def conv2d(x, channel, kernel, strides=1, padding="SAME"):
#     return tf.layers.conv2d(x, channel, kernel, strides, padding,
#                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
#
#
def conv2d(x, channel, kernel, stride=1, padding="SAME",batchnorm_istraining=None):
    conv = tf.layers.conv2d(x, channel, kernel, stride, padding,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    if batchnorm_istraining is not None:
            conv = bn(conv, batchnorm_istraining)
    return conv


def conv2d_transpose(x, channel, kernel, stride=1, padding="SAME",batchnorm_istraining=None):
    dconv = tf.layers.conv2d_transpose(x, channel, kernel, stride, padding,
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    if batchnorm_istraining is not None:
        dconv = bn(dconv, batchnorm_istraining)
    return dconv

def squash(p):
    # p = print_tensor(p, 'p')
    square_value = tf.square(p)
    # square_value = print_tensor(square_value, 'square_value')
    p_norm_sq = tf.reduce_sum(square_value, axis=-1, keep_dims=True)
    # p_norm_sq = print_tensor(p_norm_sq, 'p_norm_sq')
    p_norm = tf.sqrt(p_norm_sq + 1e-9)
    # p_norm = print_tensor(p_norm, 'p_norm')
    v = p_norm_sq / (p_norm + p_norm_sq) * p / p_norm
    # v = print_tensor(v, 'v')
    # v = p_norm_sq / (1 + p_norm_sq) * p / p_norm


    return v

def compute_vector_length(x):
    return tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keep_dims=True) + 1e-9)

def capsule(u, op, k, s, t, z, routing,bn=True):
    t_1, z_1 = t, z

    shape = u.get_shape() #tf.shape(u)
    N = shape[0]
    t_0 = shape[3]
    z_0 = shape[4]

    u_t_list = [tf.squeeze(u_t, axis=3) for u_t in tf.split(u, t_0, axis=3)]
    u_hat_t_list = []
    for u_t in u_t_list: # u_t: [N, H_0, W_0, z_0]
      if op == "conv":
        u_hat_t = conv2d(u_t, t_1*z_1, k, s,batchnorm_istraining=bn)
      elif op == "deconv":
        # u_t = print_tensor(u_t, 'deconv u_t')
        u_hat_t = conv2d_transpose(u_t, t_1*z_1, k, s,batchnorm_istraining=bn)
        # u_hat_t = print_tensor(u_hat_t, 'deconv u_hat_t')

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
    # u_hat_t_list_sg = print_tensor(u_hat_t_list_sg, 'u_hat_t_list_sg')

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
      # p = print_tensor(p, 'p')

      # v = print_tensor(v, 'v')

      if d < routing - 1:
        b_t_list_ = []
        for b_t, u_hat_t in zip(b_t_list, u_hat_t_list_):
          # b_t     : [N, H_1, W_1, t_1]
          # u_hat_t : [N, H_1, W_1, t_1, z_1]
          # v       : [N, H_1, W_1, t_1, z_1]
          b_t_list_.append(b_t + tf.reduce_sum(u_hat_t * p, axis=4))
        b_t_list = b_t_list_
    v = squash(p)
    # v = print_tensor(v, 'final v')
    return v


def residual_cap_block(input,routing,batchnorm_istraining=None):
    shape = input.shape
    [N, H, W, t , z] =shape
    cap_1 = capsule(input, "conv", k=3, s=1, t=t, z=z, routing=routing,bn=False)
    cap_2 = capsule(cap_1, "conv", k=3, s=1, t=t, z=z, routing=routing,bn=True)
    residul = input +cap_2

    return residul

def my_segcap(images,is_train,size, l2_reg):
    is_training =True
    start_s = 4
    keep_prob = 0.8
    end_points =[]
    atom = 8
    # 1  (128 -> 128)
    conv1 =conv(images, filters=8, kernel_size=[1, 1],l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
    conv_prime = tf.expand_dims(conv1, axis=3)  # [N, H, W, t=1, z]
    # conv_prime = print_tensor(conv_prime,'conv_prime')

    # 1/2  (128 -> 64)
    multiple = 1
    cap1_1 = capsule(conv_prime, "conv", k=3, s=1, t=start_s*multiple, z=atom, routing=3,bn=True)
    cap1_2 = residual_cap_block(cap1_1,routing=3)
    # cap1_1 = print_tensor(cap1_1,'cap1_1')
    cap1_3 = capsule(cap1_2, "conv", k=3, s=2, t=start_s*multiple, z=atom, routing=3,bn=True)
    # cap1_2 = print_tensor(cap1_2,'cap1_2')
    skip1 = cap1_2

    # 1/4  (64 -> 32)
    multiple = 2
    cap2_1 = capsule(cap1_3, "conv", k=3, s=1, t=start_s*multiple, z=atom, routing=3,bn=True)
    cap2_2 = residual_cap_block(cap2_1,routing=3)
    # cap2_1 = print_tensor(cap2_1,'cap2_1')
    cap2_3 = capsule(cap2_2, "conv", k=3, s=2, t=start_s*multiple, z=atom, routing=3,bn=True)
    # cap2_2 = print_tensor(cap2_2,'cap2_2')
    skip2 = cap2_2
    # skip2 = print_tensor(skip2,'skip2')

    # 1/8  (32 -> 16)
    multiple = 4
    cap3_1 = capsule(cap2_3, "conv", k=3, s=1, t=start_s*multiple, z=atom, routing=3,bn=True)
    cap3_2 = residual_cap_block(cap3_1,routing=3,batchnorm_istraining=True)
    cap3_3 = capsule(cap3_2, "conv", k=3, s=2, t=start_s*multiple, z=atom, routing=3,bn=True)
    skip3 = cap3_2

    #middle  (16 -> 16)
    # multiple = 4
    # cap_m_1 = capsule(cap2_2, "conv", k=3, s=1, t=start_s*multiple, z=atom, routing=2)
    # cap_m_1 = print_tensor(cap_m_1,'cap_m_1')
    # cap_m_2 = capsule(cap3_2,"conv", k=3, s=1, t=start_s*multiple, z=atom, routing=2)
    cap_m_1 = residual_cap_block(cap3_3,routing=3)
    # cap_m_1 = print_tensor(cap_m_1,'cap_m_1')
    cap_m_2 = residual_cap_block(cap_m_1,routing=3)
    # cap_m_2 = print_tensor(cap_m_2,'cap_m_2')

    # 1/8  (16 -> 32)
    multiple = 4
    u_cap1_1 = capsule(cap_m_2, "deconv", k=3, s=2, t=start_s*multiple, z=atom, routing=3,bn=True)
    u_cap_concat_1 = tf.concat([u_cap1_1, skip3], axis=3)
    # u_cap1_2 = capsule(u_cap_concat_1, "conv", k=3, s=1, t=start_s*multiple, z=atom, routing=2)
    # u_cap1_3 = capsule(u_cap1_2, "conv", k=3, s=1, t=start_s*multiple, z=atom, routing=2)
    u_cap1_2 = residual_cap_block(u_cap_concat_1,routing=3,batchnorm_istraining=True)

    # 1/4  (32 -> 64)
    multiple = 2
    u_cap2_1 = capsule(u_cap1_2, "deconv", k=3, s=2, t=start_s*multiple, z=atom, routing=3,bn=True)
    # u_cap2_1 = print_tensor(u_cap2_1, 'u_cap2_1')
    u_cap_concat_2 = tf.concat([u_cap2_1, skip2], axis=3)
    # u_cap_concat_2 = print_tensor(u_cap_concat_2, 'u_cap_concat_2')
    u_cap2_3 = residual_cap_block(u_cap_concat_2,routing=3)
    # u_cap2_3 = print_tensor(u_cap2_3, 'u_cap2_3')
    # 1/2  (64 -> 128)
    multiple = 1
    u_cap3_1 = capsule(u_cap2_3, "deconv", k=3, s=2, t=start_s*multiple, z=atom, routing=3,bn=True)
    # u_cap3_1 = print_tensor(u_cap3_1, 'u_cap3_1')
    u_cap_concat_3 = tf.concat([u_cap3_1, skip1], axis=3)
    # u_cap_concat_3 = print_tensor(u_cap_concat_3, 'u_cap_concat_3')
    # u_cap3_2 = capsule(u_cap_concat_3, "conv", k=3, s=1, t=start_s*multiple, z=atom, routing=2,bn=True)
    u_cap3_2 = residual_cap_block(u_cap_concat_3,routing=3)
    # u_cap3_2 = print_tensor(u_cap3_2,'u_cap3_2')

    # 1   (128 -> 128)
    # cap_out_1 = capsule(u_cap3_2, "conv", k=3, s=1, t=start_s*multiple, z=atom, routing=2,bn=True)
    # cap_out_2 = capsule(cap_out_1, "conv", k=3, s=1, t=1, z=1, routing=2)
    cap_out_1 = residual_cap_block(u_cap3_2,routing=3)
    # cap_out_1 = print_tensor(cap_out_1,'cap_out_1')
    # cap_out_2 = residual_cap_block(cap_out_1,routing=2,batchnorm_istraining=True)
    cap_out_3 = capsule(cap_out_1, "conv", k=3, s=1, t=1, z=1, routing=3,bn=True)
    cap_out_4 = tf.squeeze(cap_out_3, axis=3)
    # cap_out_4 = print_tensor(cap_out_4,'cap_out_4')
    cap_out_5 = bn(cap_out_4, is_training)
    # cap_out_5 = print_tensor(cap_out_5,'cap_out_5')
    return cap_out_5,end_points
