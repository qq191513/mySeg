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
import matplotlib.pyplot as plt
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
##########################   end   ##########################################
#代码初始化
session_config = dk.set_gpu()
n_batch_test = int(test_data_number //batch_size)
print('n_batch_test: ',n_batch_test)
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
        x = tf.placeholder(tf.float32, shape=input_shape)

        # 构建网络
        prediction,end_points = model(images=x, is_train=is_train, size=input_shape, l2_reg=0.0001)
        # 打印模型结构
        dk.print_model_struct(end_points)
        # 初始化变量
        coord, threads = dk.init_variables_and_start_thread(sess)
        # 恢复model
        saver, start_epoch = dk.restore_model(sess, ckpt, restore_model=restore_model)
        # 显示参数量
        dk.show_parament_numbers()
        # 测试loop

        n_batch_index =0
        n_pic_index = 0
        channel_index = 0

        for n_batch in range(n_batch_test):
            n_batch_index += 1
            batch_x, batch_y = sess.run([train_x, train_y])  # 取出一个batchsize的图片。
            batch_x = batch_x / 255.0
            # 3、预测输出一个张量

            for key,value in end_points.items():
                #保存的路径
                save_root = os.path.join(predict_tensor_feature_map, key)
                os.makedirs(save_root,exist_ok=True)
                predict_end_points = sess.run(value, feed_dict={x: batch_x})

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
                pics_list = []  # 清空pics_list
                for each_b in batch_size_list: #循环一个batch_size里面的每张图片图片

                    pic_channel_list = []
                    #一张图片切分成channel个，把每个featu map装进pics_list里面
                    channel_list = np.split(each_b, channel,axis=3)
                    for each_c in channel_list:  # 切分channel（显示每个channel的图片）

                        each_c = np.reshape(each_c, (width, height, 1))  #得到一张图片
                        # 显示方式1：二值化
                        # 4、预测图转二值化图（非0即1） ，经过试验tensor阈值为0最佳
                        # each_c[each_c >= 0] = 255
                        # each_c[each_c < 0] = 0
                        # pics_list.extend([each_c])

                        # 显示方式2：归一化并转成灰度图处理
                        each_c = MaxMinNormalization(each_c,Max =np.max(each_c),Min = np.min(each_c))
                        # each_c = cv2.cvtColor(each_c, cv2.COLOR_GRAY2RGB)
                        # pics_list.extend([each_c])

                        # 显示方式3：
                        # each_c = Z_ScoreNormalization(each_c, mu=np.average(each_c), sigma=np.std(each_c))
                        # each_c = cv2.cvtColor(each_c, cv2.COLOR_GRAY2RGB)
                        pic_channel_list.extend([each_c]) #一张图片所有的channel
                    pics_list.append(pic_channel_list) #把pic_channel_list添加进去

                for pic_channel_list in pics_list:   #第几张图片
                    n_pic_index += 1
                    new_dir ='{}_{}'.format(n_batch_index,n_pic_index)
                    save_dir = os.path.join(save_root,new_dir)
                    os.makedirs(save_dir, exist_ok=True)
                    for pic in pic_channel_list:  #第几个通道
                        channel_index += 1
                        # 保存的文件名
                        save_name = '{}.jpg'.format(channel_index)
                        # 保存的整个路径
                        save_name = os.path.join(save_dir, save_name)

                        cv2_show = True
                        if cv2_show==True:
                            cv2.imshow('pic', pic)
                            cv2.imwrite(save_name, pic * 255)
                            cv2.waitKey(500)
                        else:
                            plt_imshow_1_pics(pic,save_name)
                            time.sleep(1)

                        print(save_name)
                    channel_index = 0
                n_pic_index=0
        n_batch_index = 0

        dk.stop_threads(coord, threads)
        # coord.request_stop()
        # coord.join(threads)


########################废弃代码
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