import sys
sys.path.append('../')
from choice import cfg

import tensorflow as tf
num_classes = cfg.num_classes


#在成功的v2基础上整理代码

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

def squash(p,z_0):
    square_value = tf.square(p)
    # square_value = print_tensor(square_value, 'square_value')
    p_norm_sq = tf.reduce_sum(square_value, axis=-1, keep_dims=True)
    # p_norm_sq = print_tensor(p_norm_sq, 'p_norm_sq')
    p_norm = tf.sqrt(p_norm_sq + 1e-9)
    sqrt_p_norm = tf.sqrt(p_norm + 1e-9)
    # p_norm = print_tensor(p_norm, 'p_norm')
    z_0 = z_0.value
    v = p_norm_sq / (sqrt_p_norm + p_norm_sq) * p / p_norm + 1/z_0/10
    # v = print_tensor(v, 'v')

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
        u_hat_t = conv2d(u_t, t_1*z_1, k, s,batchnorm_istraining=True)
      elif op == "deconv":
        # u_t = print_tensor(u_t, 'deconv u_t')
        u_hat_t = conv2d_transpose(u_t, t_1*z_1, k, s,batchnorm_istraining=True)
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
      v = squash(p,z_0)
      # v = print_tensor(v, 'v')

      if d < routing - 1:
        b_t_list_ = []
        for b_t, u_hat_t in zip(b_t_list, u_hat_t_list_):
          # b_t     : [N, H_1, W_1, t_1]
          # u_hat_t : [N, H_1, W_1, t_1, z_1]
          # v       : [N, H_1, W_1, t_1, z_1]
          b_t_list_.append(b_t + tf.reduce_sum(u_hat_t * v, axis=4))
        b_t_list = b_t_list_

    # v = print_tensor(v, 'final v')
    return v


def residual_cap_block(input,routing,batchnorm_istraining=None):
    shape = input.shape
    [N, H, W, t , z] =shape
    cap_1 = capsule(input, "conv", k=3, s=1, t=t, z=z, routing=routing)
    cap_2 = capsule(cap_1, "conv", k=3, s=1, t=t, z=z, routing=routing)

    residul = input +cap_2
    residul = tf.nn.relu(residul)
    return residul
from collections import OrderedDict
def my_segcap(images,is_train,size, l2_reg):
    is_training =True
    start_s = 2
    atom = 16
    routing = 3
    end_points =OrderedDict()

    # 1  (128 -> 128)
    L1_conv1 =conv(images, filters=atom, kernel_size=[1,1],l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
    conv_prime = tf.expand_dims(L1_conv1, axis=3)  # [N, H, W, t=1, z]

    # 1/2  (128 -> 64)
    multiple = 1
    L2_cap1_1 = residual_cap_block(conv_prime,routing=routing)
    L3_cap1_2 = capsule(L2_cap1_1, "conv", k=3, s=2, t=start_s*multiple, z=atom, routing=routing)
    skip1 = L2_cap1_1

    # 1/4  (64 -> 32)
    multiple = 2
    L4_cap2_1 = residual_cap_block(L3_cap1_2,routing=routing)
    L5_cap2_2 = capsule(L4_cap2_1, "conv", k=3, s=2, t=start_s*multiple, z=atom, routing=routing)
    skip2 = L4_cap2_1

    #middle  (16 -> 16)
    L6_cap_m_1 = residual_cap_block(L5_cap2_2,routing=routing)
    L7_cap_m_2 = residual_cap_block(L6_cap_m_1,routing=routing)

    # 1/4  (32 -> 64)
    multiple = 2
    L8_u_cap2_1 = capsule(L7_cap_m_2, "deconv", k=3, s=2, t=start_s*multiple, z=atom, routing=routing)
    u_cap_concat_2 = tf.concat([L8_u_cap2_1, skip2], axis=3)
    L9_u_cap2_3 = residual_cap_block(u_cap_concat_2,routing=routing)

    # 1/2  (64 -> 128)
    multiple = 1
    L10_u_cap3_1 = capsule(L9_u_cap2_3, "deconv", k=3, s=2, t=start_s*multiple, z=atom, routing=routing)
    u_cap_concat_3 = tf.concat([L10_u_cap3_1, skip1], axis=3)
    L11_u_cap3_2 = capsule(u_cap_concat_3, "conv", k=3, s=1, t=start_s*multiple, z=atom*4, routing=routing)
    L12_u_cap3_3 = residual_cap_block(L11_u_cap3_2,routing=routing)
    L13_u_cap3_4 = residual_cap_block(L12_u_cap3_3,routing=routing)
    L14_u_cap3_5 = capsule(L13_u_cap3_4, "conv", k=3, s=1, t=1, z=atom, routing=routing)
    # L14_u_cap3_5_l_list =tf.split(L14_u_cap3_5,num_or_size_splits=atom,axis=4)
    # L14_u_cap3_5_l_add = tf.add_n(L14_u_cap3_5_l_list)
    # predict = tf.squeeze(L14_u_cap3_5_l_add, axis=[4])

    #  tf.norm默认为Frobenius范数，简称F - 范数，是一种矩阵范数，记为 | |· | | F。
    #  矩阵A的Frobenius范数定义为矩阵A各项元素的绝对值平方的总和
    predict = tf.norm(L14_u_cap3_5,axis=-1)
    predict = bn(predict, is_training)
    # tf.squeeze()

    # 1  (128 -> 128)
    # u_cap_concat_4=cap_out_1
    # [N, H_1, W_1, t_1, z_1] =u_cap_concat_4.get_shape()
    # u_cap_concat_4 = tf.reshape(u_cap_concat_4, [N, H_1, W_1, 1,t_1* z_1])


    #普通输出层
    # cap_out_4 = tf.squeeze(u_cap_concat_4, axis=3)
    # cap_out_7 =conv(cap_out_4, filters=24, kernel_size=[1,1],l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
    # cap_out_8 =conv(cap_out_7, filters=1, kernel_size=[1,1],l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
    # cap_out_9 = bn(cap_out_8, is_training)

    ################   end_points  ##########################
    #用于输出可视化中间层
    end_points['L1_conv1'] = L1_conv1           #Layer 1
    end_points['L2_cap1_1'] = L2_cap1_1         #Layer 2  skip1
    end_points['L3_cap1_2'] = L3_cap1_2         #Layer 3
    end_points['L4_cap2_1']=L4_cap2_1         #Layer 4  skip2
    end_points['L5_cap2_2']=L5_cap2_2         #Layer 5
    end_points['L6_cap_m_1']=L6_cap_m_1       #Layer 6
    end_points['L7_cap_m_2']=L7_cap_m_2       #Layer 7
    end_points['L8_u_cap2_1']=L8_u_cap2_1     #Layer 8  skip2
    end_points['L9_u_cap2_3']=L9_u_cap2_3     #Layer 9
    end_points['L10_u_cap3_1']=L10_u_cap3_1     #Layer 10  skip1
    end_points['L11_u_cap3_2']=L11_u_cap3_2     #Layer 11
    end_points['L12_u_cap3_3']=L12_u_cap3_3   #Layer 12
    end_points['L13_u_cap3_4']=L13_u_cap3_4   #Layer 13
    end_points['L14_u_cap3_5']=L14_u_cap3_5   #Layer 14
    end_points['predict']=predict   #Layer 15
    ################     end       ###########################

    return predict,end_points
