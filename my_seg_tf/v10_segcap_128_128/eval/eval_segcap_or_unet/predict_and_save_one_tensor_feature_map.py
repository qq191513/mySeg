#coding=utf-8

import numpy as np
import tensorflow as tf
import sys
sys.path.append('../')
import time
import tools.development_kit as dk
import cv2
import os
from data_process.use_seg_tfrecord import create_inputs_seg_hand as create_inputs
from choice import cfg
from choice import model
from tools.visual_tool import plt_imshow_1_pics

##########################   cfg 固定写法   #######################################
is_train=False #True使用训练集，#False使用测试集
restore_model  = True
batch_size = cfg.batch_size
save_list_csv = cfg.save_list_csv
save_mean_csv = cfg.save_mean_csv
save_plot_curve_dir = cfg.save_plot_curve_dir
input_shape = cfg.input_shape
labels_shape= cfg.labels_shape
ckpt =cfg.ckpt
train_data_number = cfg.train_data_number
test_data_number = cfg.test_data_number
predict_tensor_feature_map = cfg.predict_tensor_feature_map
test_opoch = 2
##########################   end   ##########################################
#代码初始化
session_config = dk.set_gpu()
n_batch_train = int(train_data_number //batch_size)
n_batch_test = int(test_data_number //batch_size)

os.makedirs(predict_tensor_feature_map,exist_ok=True)

def MaxMinNormalization(x,Max,Min):
    x = (x - Min) / (Max - Min);
    return x
def Z_ScoreNormalization(x,mu,sigma):
    x = (x - mu) / sigma;
    return x
def predict_and_save_tensor_feature_map_model():
    with tf.Session(config=session_config) as sess:
        # 入口
        train_x, train_y = create_inputs(is_train)
        # train_y = tf.reshape(train_y, labels_shape)
        x = tf.placeholder(tf.float32, shape=input_shape)
        y = tf.placeholder(tf.float32, shape=labels_shape)
        # 构建网络
        prediction,end_points = model(images=x, is_train=is_train, size=input_shape, l2_reg=0.0001)
        # prediction = tf.reshape(prediction, labels_shape)
        # 初始化变量
        coord, threads = dk.init_variables_and_start_thread(sess)
        # 恢复model
        saver = dk.restore_model(sess, ckpt, restore_model=restore_model)
        # 显示参数量
        dk.show_parament_numbers()
        # 测试loop
        start_epoch= 0
        index = 0
        for epoch_n in range(start_epoch, test_opoch):
            for n_batch in range(n_batch_test):
                batch_x, batch_y = sess.run([train_x, train_y])  # 取出一个batchsize的图片。
                batch_x = batch_x / 255.0
                # 3、预测输出一个张量
                since = time.time()
                predict_end_points = sess.run(end_points, feed_dict={x: batch_x})
                seconds = time.time() - since
                predict_end_points = predict_end_points[0]
                shape = predict_end_points.shape
                if len(shape) ==4:
                    batch_size,width,height,channel = shape
                elif len(shape) == 5:
                    batch_size, width, height, channel_1, channel_2 = shape
                    shape = (batch_size, width, height, channel_1 * channel_2)
                    predict_end_points = np.reshape(predict_end_points,shape)
                    channel = channel_1* channel_2
                else:
                    raise Exception('fuck!')
                #切分batch_size
                batch_size_list = np.split(predict_end_points, batch_size,axis=0)
                pics_list = []
                for each_b in batch_size_list:
                    # 切分channel
                    channel_list = np.split(each_b, channel,axis=3)
                    for each_c in channel_list:
                        each_c = np.reshape(each_c, (width, height, 1))
                        # 显示方式1：二值化
                        # 4、预测图转二值化图（非0即1） ，经过试验tensor阈值为0最佳
                        # each_c[each_c >= 0] = 255
                        # each_c[each_c < 0] = 0
                        # pics_list.extend([each_c])

                        # 显示方式2：归一化并转成灰度图处理
                        # each_c = MaxMinNormalization(each_c,Max =np.max(each_c),Min = np.min(each_c))
                        # each_c = cv2.cvtColor(each_c, cv2.COLOR_GRAY2RGB)
                        # pics_list.extend([each_c])

                        # 显示方式3：
                        each_c = Z_ScoreNormalization(each_c, mu=np.average(each_c), sigma=np.std(each_c))
                        each_c = cv2.cvtColor(each_c, cv2.COLOR_GRAY2RGB)
                        pics_list.extend([each_c])

                for pic in pics_list:
                    save_name = '{}.jpg'.format(index)
                    save_name = os.path.join(predict_tensor_feature_map, save_name)
                    cv2_show = True
                    if cv2_show==True:
                        cv2.imshow('pic', pic)
                        cv2.waitKey(500)
                    else:
                        plt_imshow_1_pics(pic)
                        time.sleep(2)
                    print(pic)


                    # cv2.imwrite(save_name,pic*255)
                    index += 1
                    print(save_name)

                # 4、预测图转二值化图（非0即1）
                # predict[predict>=0.5] =1
                # predict[predict< 0.5] =0
                # 5、把batch图片分割一张张图片
                # batch_pre_list = np.split(predict, batch_size, axis=0)
                # pre_pics_list = np.squeeze(batch_pre_list, axis=0)
                # for img, label, predict in zip(batch_x, batch_y, pre_pics_list):
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # label = cv2.cvtColor(label, cv2.COLOR_GRAY2RGB)
                    # predict = cv2.cvtColor(predict, cv2.COLOR_GRAY2RGB)
                    # hstack = np.hstack((img, label,predict))
                    # hstack = cv2.resize(hstack,(512,512))
                    #
                    # save_name = '{}.jpg'.format(index)
                    # save_name = os.path.join(predict_pics_save, save_name)
                    # cv2.imshow('hstack',hstack)
                    # cv2.waitKey(500)
                    # cv2.imwrite(save_name, hstack*255)
                    # index += 1
                    # print(save_name)

        dk.stop_threads(coord, threads)
        # coord.request_stop()
        # coord.join(threads)
