#coding=utf-8
from v1.loss import get_loss
from v1.loss import dice_hard
from keras.optimizers import Adam

##########################   训练集相关   #######################################
#训练数据集
train_images = "/home/mo/work/seg_caps/my_seg_keras/dataset/dataset_street/images_prepped_train/"
train_label = "/home/mo/work/seg_caps/my_seg_keras/dataset/dataset_street/annotations_prepped_train/"
#网络输入图片要reshape成的大小，这里先保持和原图像大小保持一致
# 图片标准格式h*w*c
input_shape = (320,640,3)
#网络输出标签要reshape成的大小，这里先保持和原标签大小保持一致
# 图片标准格式h*w*c
output_shape = (320,640,3)
#类别个数
n_classes = 10
#每次训练多少张图片 (32的话内存被榨干了，16的话还剩4G内存,运行结束后也会崩掉)
train_batch_size=8
#训练图片张数
train_data_number=367
########################        end      ########################################

##########################   训练配置   ########################################
#保存模型路径
save_weights_path= '/home/mo/work/seg_caps/my_seg_keras/weights'
#epochs是训练总次数
epochs =2
#每隔多少epochs保存一次模型,epochs要能整除epochs_save
epochs_save=2
#使用5个进程(5个python3.5运行)训练，这个训练比较刚猛，速度快
use_multiprocessing =True
#使用线程数量，好像并没什么卵用，跟进程数量也没什么关系
workers = 1
#每个epoch训练多少次
train_steps_per_epoch = int(train_data_number / train_batch_size)
# optimizer选择
optimizer=Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon = 0.1, decay = 1e-6)
#loss选择
loss = get_loss('bce_dice')
#metrics选择
metrics = [dice_hard,'accuracy']
# metrics = ['accuracy']
########################        end      ########################################



##########################   校验集相关   #######################################
# 是否训练的时候用校验数据集进行评估,一张GPU内存不足可能无法启用该功能,只能设置False了
validate = False

#校验数据集
val_images = "/home/mo/work/seg_caps/my_seg_keras/dataset_street/images_prepped_test/"
val_label = "/home/mo/work/seg_caps/my_seg_keras/dataset_street/annotations_prepped_test/"

#训练图片张数
test_data_number=101
#一次测试多少张图片
val_batch_size =2
#每个epoch生成器返回多少波数据
validate_steps_per_epoch = int(test_data_number / val_batch_size)
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






