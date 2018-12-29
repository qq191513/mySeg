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
model_name =cfg.weights_path
model = cfg.select_model
#预测的图片集
images_path=cfg.train_images
label_path = cfg.train_label
train_data_number =cfg.train_data_number
#输出路径
eval_output_path =cfg.eval_output_path
#选择保存class_pics还是color_pics(针对h,w,class类型输出，最后tf.argmax最后一轴，得出一张图，能做多分类)
choose_save_class_pics =True
choose_save_color_pics =True

#选择保存class_pics还是color_pics(针对图片类型类型输出一张图,最后凭感觉设定一个阈值过滤成0，1二值图，也只能做二分类了)
choose_save_segout_pics =False
choose_save_recon_pics =False
threshold= None
########################        end      ########################################

if choose_save_class_pics:
    save_class_pics_dir = os.path.join(eval_output_path, 'pre_class_pics')
    os.makedirs(save_class_pics_dir, exist_ok=True)
if choose_save_color_pics:
    save_color_pics_dir = os.path.join(eval_output_path, 'pre_color_pics')
    os.makedirs(save_color_pics_dir, exist_ok=True)

if choose_save_segout_pics:
    save_segout_pics_dir = os.path.join(eval_output_path, 'pre_segout_pics')
    os.makedirs(save_segout_pics_dir, exist_ok=True)
if choose_save_recon_pics:
    save_recon_pics_dir = os.path.join(eval_output_path, 'pre_recon_pics')
    os.makedirs(save_recon_pics_dir, exist_ok=True)

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

def save_recon_pics(pr,save_name,threshold):
    pr = cv2.resize(pr, (cfg.input_shape[1], cfg.input_shape[0]))
    if threshold:
        pr[pr>threshold] = 1
        pr[pr>threshold] = 0
    else:
        pr = pr *255

    cv2.imwrite(save_name, pr)

def save_segout_pics(pr,save_name,threshold):
    if threshold:
        pr[pr>threshold] = 1
        pr[pr>threshold] = 0
    else:
        pr = pr *255
    pr = cv2.resize(pr, (cfg.input_shape[1], cfg.input_shape[0]))
    cv2.imwrite(save_name, pr)

def predict_result():
    #选择网络
    train_model, eval_model = cfg.select_model
    eval_model.compile(optimizer=cfg.optimizer, loss=cfg.loss,metrics=cfg.metrics)


    G = X_Y_Z_iter(images_path,label_path , 1,
            cfg.n_classes, cfg.input_shape[0], cfg.input_shape[1], cfg.output_shape[0],
                     cfg.output_shape[1])
    index=0
    for X,Y,name in G:
        index+=1
        X = np.squeeze(X,axis=0)
        # Y = np.squeeze(Y,axis=0)
        # 1、开始计时
        since = time.time()
        # 2、Net开始预测
        pre= eval_model.predict( np.array([X]))
        out_seg, recon =pre
        # 3、计算耗时
        seconds = time.time() - since

        #4、 将预测值保存成标注图
        if choose_save_class_pics:
            save_name = os.path.join(save_class_pics_dir, name[0])+ '.png'
            save_class_pics(out_seg, save_name)
            print('{}/{}  time use {:.3f}s,save to {}'.format(index, train_data_number, seconds, save_name))
        # 5、将预测值保存成彩图
        if choose_save_color_pics:
            save_name = os.path.join(save_color_pics_dir, name[0])+ '.png'
            save_color_pics(out_seg, colors, save_name)
            print('{}/{}  time use {:.3f}s,save to {}'.format(index, train_data_number, seconds, save_name))

        # 6、保存分割图
        if choose_save_segout_pics:
            save_name = os.path.join(save_segout_pics_dir, name[0]) + '.png'
            save_segout_pics(out_seg, save_name,threshold)
            print('{}/{}  time use {:.3f}s,save to {}'.format(index, train_data_number, seconds, save_name))
        # 7、保存重构图
        if choose_save_recon_pics:
            save_name = os.path.join(save_recon_pics_dir, name[0])+ '.png'
            save_recon_pics(recon, save_name,threshold)
            print('{}/{}  time use {:.3f}s,save to {}'.format(index, train_data_number, seconds, save_name))

predict_result()