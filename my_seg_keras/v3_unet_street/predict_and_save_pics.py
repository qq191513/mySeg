import glob
import cv2
import numpy as np
import random
from models.unet import UNet
from v1 import config as cfg
from data_process import use_generator
import time

##########################   改这里   #######################################
#已经训练好的模型
model_name = 'model.1'
model = UNet(cfg.input_shape)
#预测的图片集
images_path=cfg.train_images
#输出路径
output_path= "../output_predict_result/"
########################        end      ########################################



#########################   使用GPU  动态申请显存占用 ####################
# 1、使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放内存，所以会导致碎片
# 2、visible_device_list指定使用的GPU设备号；
# 3、allow_soft_placement如果指定的设备不存在，允许TF自动分配设备（这个设置必须有，否则无论如何都会报cudnn不匹配的错误）
# 4、per_process_gpu_memory_fraction  指定每个可用GPU上的显存分配比
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
session_config = tf.ConfigProto(
            device_count={'GPU': 0},  #不能写成小写的gpu，否则无效
            gpu_options={'allow_growth': 1,
                # 'per_process_gpu_memory_fraction': 0.1,
                'visible_device_list': '0'},
                allow_soft_placement=True) #这个设置必须有，否则无论如何都会报cudnn不匹配的错误

sess = tf.Session(config=session_config)
KTF.set_session(sess)

#########################   END   ####################################



# def dim_classes_to_dim_3(dim_classes,dim_3,colors,n_classes):
	#w * h * n_classes的张量转w * h * 3的颜色图片
	# seg_img = np.zeros(shape=restore_pic_shape)
	# the_shape = restore_pic_shape
	# h,w  = the_shape[0],the_shape[1]
	# seg_vec = seg_vec.reshape((h,w, -1)) # h * w * n_classes 做成（h ，w ，n_classes）
    #
	# for c in range(n_classes):
	# 	seg_img[:, :, 0] += ((seg_vec[:, :, 0] == c) * (colors[c][0])).astype('uint8')
	# 	seg_img[:, :, 1] += ((seg_vec[:, :, 1] == c) * (colors[c][1])).astype('uint8')
	# 	seg_img[:, :, 2] += ((seg_vec[:, :, 2] == c) * (colors[c][2])).astype('uint8')
	# seg_img = seg_img / 255.0
	# seg_img = seg_img.astype('float32')
	# return seg_img

def predict_result():
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 随机生成颜色
    colors = [  ( random.randint(0,255),random.randint(0,255),random.randint(0,255)) for _ in range(cfg.n_classes)  ]
    model.load_weights(os.path.join(cfg.save_weights_path,model_name))
    model.compile(optimizer=cfg.optimizer, loss=cfg.loss,metrics=cfg.metrics)


    images = glob.glob(os.path.join(images_path, "*.jpg")) \
         + glob.glob(os.path.join(images_path, "*.png")) + glob.glob(os.path.join(images_path, "*.jpeg"))
    images.sort()
    for index,imgName in enumerate(images):
        outName = os.path.join(output_path,os.path.basename(imgName))
        #图片归一化
        X = use_generator.getImageArr(imgName, cfg.input_shape[0], cfg.input_shape[1], imgNorm="divide")

        # 1、开始计时
        since = time.time()
        # 2、开始预测
        pr = model.predict( np.array([X]))[0]
        # 3、计算耗时
        seconds = time.time() - since

        pr = pr.reshape((cfg.output_shape[0], cfg.output_shape[1], cfg.n_classes))
        pr = pr.argmax(axis=2)
        seg_img = np.zeros((cfg.output_shape[0], cfg.output_shape[1], 3))
        for c in range(cfg.n_classes):
            seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')
        seg_img = cv2.resize(seg_img, (cfg.input_shape[1],cfg.input_shape[0] ))
        cv2.imwrite(outName, seg_img)

        # seg_img = seg_vec_to_pic(pr, cfg.output_shape, colors, cfg.n_classes)
        # cv2.imwrite(outName, seg_img)
        print('time use {:.3f}s,save to {}'.format(seconds,outName))


predict_result()