﻿##########################   训练集相关   #######################################
#训练数据集
train_images = "../dataset_street/images_prepped_train/"
train_annotations = "../dataset_street/annotations_prepped_train/"

#网络输入图片要reshape成的大小，这里先保持和原图像大小保持一致
# h*w*c
input_shape = (320,640,3)

#网络输出标签要reshape成的大小，这里先保持和原标签大小保持一致
# h*w*c
output_shape = (320,640,3)

#类别个数
n_classes = 10

#每次训练多少张图片
train_batch_size=2
#训练图片张数
train_data_number=367

#每个epoch生成器返回多少波数据
train_steps_per_epoch = int(train_data_number / train_batch_size)

#epochs是训练总次数
epochs =2

#每隔多少epochs保存一次模型,epochs要能整除epochs_save
epochs_save=2

#使用5个进程训练，这个训练比较刚猛，速度快
use_multiprocessing =True

#使用线程数量，好像并没什么卵用，跟进程数量也没什么关系
workers = 1

#保存模型路径
save_weights_path= 'weights'
########################        end      ########################################


##########################   校验集相关   #######################################
# 是否训练的时候用校验数据集进行评估,一张GPU内存不足可能无法启用该功能,只能设置False了
validate = False

#校验数据集
val_images = "../dataset/dataset_street/images_prepped_test/"
val_annotations = "../dataset/dataset_street/annotations_prepped_test/"

#训练图片张数
test_data_number=101
#一次测试多少张图片
val_batch_size =1
#每个epoch生成器返回多少波数据
validate_steps_per_epoch = int(test_data_number / val_batch_size)
########################        end      ########################################