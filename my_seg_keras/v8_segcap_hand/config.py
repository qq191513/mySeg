#coding=utf-8
from v1.loss import get_loss
from v1.loss import dice_hard
from keras.optimizers import Adam
import os
import sys
sys.path.append('./')
# from models.capsnet import CapsNetR3
# from models.capsnet_v2 import CapsNetR3
from models.capsnet_v3 import CapsNetBasic

##########################   训练集相关   #######################################
project_root = '/home/mo/work/seg_caps/my_seg_keras'
#训练数据集(项目下的相对路径)
train_images = os.path.join(project_root,'dataset/ckpt_dir/Images')
train_label = os.path.join(project_root,'dataset/ckpt_dir/Masks_1')
gray_label = os.path.join(project_root,'dataset/ckpt_dir/gray')

#网络输入图片要reshape成的大小，这里先保持和原图像大小保持一致
# 图片标准格式h*w*c
input_shape = (64,64,3)
#网络输出标签要reshape成的大小，这里先保持和原标签大小保持一致
# 图片标准格式h*w*c
output_shape = (64,64,3)
#类别个数
n_classes = 2
#每次训练多少张图片 (32的话内存被榨干了，16的话还剩4G内存,运行结束后也会崩掉)
train_batch_size=8
#训练图片张数
# train_data_number=1115
train_data_number=10

########################        end      ########################################

##########################   训练配置   ########################################
debug=True
#模型选择

# select_model =CapsNetR3(input_shape, n_classes)

select_model =CapsNetBasic(input_shape, n_classes)

#epochs是训练总次数
epochs =50
#每隔多少epochs保存一次模型,epochs要能整除epochs_save
epochs_save=2
#使用5个进程(5个python3.5运行)训练，这个训练比较刚猛，速度快
use_multiprocessing =True
#使用线程数量，好像并没什么卵用，跟进程数量也没什么关系
workers = 1
#每个epoch训练多少次
train_steps_per_epoch = int(train_data_number / train_batch_size)
# optimizer选择
optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon = 1e-8, decay = 1e-6)
#loss选择
# loss,_ = get_loss('bce_dice')
loss, loss_weighting = get_loss('bce_dice_recon')
#metrics选择
metrics = [dice_hard,'accuracy']
# metrics = ['accuracy']
########################        end      ########################################

##########################   输出相关相关   #######################################
#1、保存训练曲线图
save_history_plt =os.path.join(project_root,
'output_predict_result','caps_hand_64_64','plt_histroy.png')

#2、保存模型路径
save_weights_path = os.path.join(project_root,'weights/hand_64_64_segcap')

#3、恢复路径
# weights_path = None
train_weights_recover_path = None
weights_path = os.path.join(project_root,'weights/hand_64_64_segcap','model.1')

#4、评估输出路径
eval_output_path= os.path.join(project_root,'output_predict_result/segcap_hand_64_64')
########################        end      ########################################

##########################   校验集相关   #######################################
# 是否训练的时候用校验数据集进行评估,一张GPU内存不足可能无法启用该功能,只能设置False了
validate = False

#校验数据集
val_images = "/home/mo/work/seg_caps/my_seg_keras/dataset/dataset_street/images_prepped_test/"
val_label = "/home/mo/work/seg_caps/my_seg_keras/dataset/dataset_street/annotations_prepped_test/"

#训练图片张数
test_data_number=101
#一次测试多少张图片
val_batch_size =2
#每个epoch生成器返回多少波数据
validate_steps_per_epoch = int(test_data_number / val_batch_size)
########################        end      ########################################








