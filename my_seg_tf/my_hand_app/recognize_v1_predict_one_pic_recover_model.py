import tensorflow as tf
import os
import cv2

import models.config_em as cfg
import numpy as np
from models.config_em import get_coord_add
from models.config_em import search_keyword_files
from models.config_em import read_label_txt_to_dict
from tool.visual_tool import print_variable


##########################   要改的东西   #######################################
import models.capsnet_em as net
data_dir='data'
labels_txt_keywords = 'asl_labels.txt'
latest_model_ckpt = os.path.join('/home/mo/work/seg_caps/my_hand_app/asl_model/asl')
test_pic = '0_8ori.jpg'
num_classes=36
dataset_name='asl'
input_shape=[28,28,1]
##########################   end   ##########################################


if __name__ == "__main__":
	# 1、读图
	im_file = os.path.join(data_dir, test_pic)
	# （1）用opencv方式读图
	image = cv2.cvtColor(cv2.imread(im_file), cv2.COLOR_RGB2BGR)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image,(28,28))
	image = np.expand_dims(image,axis=0)
	image = np.expand_dims(image,axis=-1)
	# 2、图片归一化、resize
	image = image / 255.0

	# 3、GPU设置
	session_config = tf.ConfigProto(
		device_count={'GPU': 0},
		gpu_options={'allow_growth': 1,
					 # 'per_process_gpu_memory_fraction': 0.1,
					 'visible_device_list': '0'},
		allow_soft_placement=True)  ##这个设置必须有，否则无论如何都会报cudnn不匹配的错误

	with tf.Session(config=session_config) as sess:
		#1、定义model
		coord_add = get_coord_add(dataset_name)
		input=  tf.placeholder(tf.float32,[cfg.batch_size,input_shape[0],input_shape[1], input_shape[2]])
		output = net.build_arch(input,coord_add, is_train=False, num_classes=num_classes)

		#2、用global_variables_initializer去初始化
		# sess.run(tf.local_variables_initializer())
		sess.run(tf.global_variables_initializer())

		#3、打印没恢复时的变量
		var_list = tf.global_variables()
		var_name_list = print_variable(sess, var_list)

		#4、打印没恢复时的预测结果
		result = sess.run(output, {input: image})
		print('没恢复的预测结果 ',np.argmax(result,axis=-1))

		#5、根据代码定义进行恢复：tf.train.latest_checkpoint
		model_file = tf.train.latest_checkpoint(latest_model_ckpt)
		saver = tf.train.Saver(var_list=var_list)
		saver.restore(sess, model_file)

		#6、测试恢复model的预测结果
		result = sess.run(output, {input: image})
		result = np.argmax(result, axis=-1)
		print('已经恢复的预测结果 ',result)

		#7、映射标签
		result = str(result[0])
		labels_txt = search_keyword_files(data_dir, labels_txt_keywords)
		labels_maps = read_label_txt_to_dict(labels_txt[0])
		label_name = labels_maps[result]
		print('label_name: ',label_name)












