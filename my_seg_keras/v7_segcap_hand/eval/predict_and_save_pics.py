#coding=utf-8
import cv2
import numpy as np
import random
import sys
sys.path.append('../')
from v1 import config as cfg
import time
from set_gpu import *
from data_process.use_generator import X_Y_Z_iter

##########################   改这里   #######################################
#已经训练好的模型
model_name =os.path.join(cfg.save_weights_path,'model.49')
model = cfg.load_model
#预测的图片集
images_path=cfg.train_images
label_path = cfg.train_label
train_data_number =cfg.train_data_number
#输出路径
output_path ="../../output_predict_result/unet_hand_64_64/"
#选择保存class_pics还是color_pics
choose_save_class_pics =False
choose_save_color_pics =True
########################        end      ########################################
if choose_save_class_pics:
    save_class_pics_dir = os.path.join(output_path, 'pre_class_pics')
    os.makedirs(save_class_pics_dir, exist_ok=True)
if choose_save_color_pics:
    save_color_pics_dir = os.path.join(output_path, 'pre_color_pics')
    os.makedirs(save_color_pics_dir, exist_ok=True)

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(cfg.n_classes)]

def save_class_pics(pr,save_name):
    #将预测值保存成标注图[0,1,2,......n_class]
    pr = pr.reshape((cfg.output_shape[0], cfg.output_shape[1], cfg.n_classes))
    pr = pr.argmax(axis=2)
    pr = cv2.resize(pr, (cfg.input_shape[1], cfg.input_shape[0]))
    cv2.imwrite(save_name, pr)

def save_color_pics(pr,colors,save_name):
    #将预测值保存成彩图
    pr = pr.reshape((cfg.output_shape[0], cfg.output_shape[1], cfg.n_classes))
    pr = pr.argmax(axis=2)
    seg_img = np.zeros((cfg.output_shape[0], cfg.output_shape[1], 3))
    for c in range(cfg.n_classes):
        seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')
    seg_img = cv2.resize(seg_img, (cfg.input_shape[1], cfg.input_shape[0]))
    cv2.imwrite(save_name, seg_img)

def predict_result():
    model.load_weights(os.path.join(model_name))
    model.compile(optimizer=cfg.optimizer, loss=cfg.loss,metrics=cfg.metrics)


    G = X_Y_Z_iter(images_path,label_path , 1,
            cfg.n_classes, cfg.input_shape[0], cfg.input_shape[1], cfg.output_shape[0],
                     cfg.output_shape[1])
    index=0
    for X,Y,Z in G:
        index+=1
        X = np.squeeze(X,axis=0)
        Y = np.squeeze(Y,axis=0)
        # 1、开始计时
        since = time.time()
        # 2、Net开始预测
        pr = model.predict( np.array([X]))[0]
        # 3、计算耗时
        seconds = time.time() - since

        #4、 将预测值保存成标注图
        if choose_save_class_pics:
            save_name = os.path.join(save_class_pics_dir, Z[0])+ '.png'
            save_class_pics(pr, save_name)

        # 5、将预测值保存成彩图
        if choose_save_color_pics:
            save_name = os.path.join(save_color_pics_dir, Z[0])+ '.png'
            save_color_pics(pr, colors, save_name)

        print('{}/{}  time use {:.3f}s,save to {}'.format(index, train_data_number, seconds, save_name))


predict_result()