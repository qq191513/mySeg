from PIL import Image
import matplotlib.image as mpimg # mpimg 用于读取图片
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import cv2
import os
def print_tensor(tensor,message=None):
    if message is None:
        message = 'Debug '
    return tf.Print(tensor, [tensor], message=message+': ', summarize=150)

def save_tensor_to_pics(sess,tensor,feed_dict,show,save_path=None,save_name=None):
    np_array = sess.run([tensor],feed_dict)
    if show:
        cv2.imshow('show',np_array)
        cv2.waitKey(1000)
    if save_path is not None:
        cv2.imwrite(os.path.join(save_path,save_name), np_array)


def print_variable(sess,var_list):
    var_name_list = []
    for v in var_list:
        if 'Adam' not in v.name:
            name = v.name
            var_name_list.append(name)
            print('name: ' + name )
    print('all var name: ' ,var_name_list)
    return var_name_list

def plt_imshow_1_pics(data,save_dir=None):
    #调成标准格式和标准维度，免得爆BUG
    data = np.asarray(data)
    if data.ndim == 3:
        if data.shape[2] == 1:
            data = data[:, :, 0]
    plt.imshow(data)
    if save_dir is not None:
        plt.savefig(save_dir)
    plt.show()


def plt_imshow_2_pics(data_1,data_2):
    #调成标准格式和标准维度，免得爆BUG
    data_1 = np.asarray(data_1)
    if data_1.ndim == 3:
        if data_1.shape[2] == 1:
            data_1 = data_1[:, :, 0]

    data_2 = np.asarray(data_2)
    if data_2.ndim == 3:
        if data_2.shape[2] == 1:
            data_2 = data_2[:, :, 0]

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(data_1)
    plt.subplot(1, 2, 2)
    plt.imshow(data_2)
    plt.show()



def plt_imshow_3_pics(data_1,data_2,data_3):
    #调成标准格式和标准维度，免得爆BUG
    data_1 = np.asarray(data_1)
    if data_1.ndim == 3:
        if data_1.shape[2] == 1:
            data_1 = data_1[:, :, 0]

    data_2 = np.asarray(data_2)
    if data_2.ndim == 3:
        if data_2.shape[2] == 1:
            data_2 = data_2[:, :, 0]

    data_3 = np.asarray(data_3)
    if data_3.ndim == 3:
        if data_3.shape[2] == 1:
            data_3 = data_3[:, :, 0]


    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(data_1)
    plt.subplot(1, 3, 2)
    plt.imshow(data_2)
    plt.subplot(1, 3, 3)
    plt.imshow(data_3)
    plt.show()