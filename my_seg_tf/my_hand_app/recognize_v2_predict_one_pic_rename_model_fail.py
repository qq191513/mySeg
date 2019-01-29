import tensorflow as tf
import os
import cv2

model_restore_name = "model_1"
input_shape=[28,28,1]
import models.config_em as cfg
import numpy as np
from models.config_em import get_coord_add

##########################   要改的东西   #######################################
import models.capsnet_em as net
import models.capsnet_em_new_name as new_net

path = '/home/mo/work/caps_face/Matrix-Capsules-EM-Tensorflow-master/capsnet_em_v3/logdir/caps/asl'
num_classes=36
dataset_name='asl'

##########################   end   ##########################################
#命名失败

def print_variable(sess,var_list):
	var_name_list = []
	for v in var_list:
		if 'Adam' not in v.name:
			name = v.name
			var_name_list.append(name)
			# vary = sess.run(v)
			# print('name: ' + name + 'vuale: ', vary)
			print('name: ' + name )
	print('all var name: ' ,var_name_list)
	return var_name_list
if __name__ == "__main__":
	# 1、读图
	im_file = os.path.join('data', '170.png')
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
		result = sess.run([output], {input: image})
		print('没恢复的预测结果 ',np.argmax(result,axis=-1))

		#5、根据代码定义进行恢复：tf.train.latest_checkpoint
		latest= os.path.join('/home/mo/work/seg_caps/my_hand_app/asl_model/asl')
		model_file = tf.train.latest_checkpoint(latest)
		saver = tf.train.Saver(var_list=var_list)
		saver.restore(sess, model_file)

		#6、测试恢复model的预测结果
		result = sess.run([output], {input: image})
		print('已经恢复的预测结果 ',np.argmax(result,axis=-1))

		#7、进行改名字{旧名：新名}

		name_dict = {'relu_conv1/branch_0/weights:0':'caps_relu_conv1/branch_0/weights:0',
					 'relu_conv1/branch_0/biases:0':'caps_relu_conv1/branch_0/biases:0',
					 'relu_conv1/branch_1_1/weights:0': 'caps_relu_conv1/branch_1_1/weights:0',
					 'relu_conv1/branch_1_1/biases:0':'caps_relu_conv1/branch_1_1/biases:0',
					 'relu_conv1/branch_1_2/weights:0': 'caps_relu_conv1/branch_1_2/weights:0',
					 'relu_conv1/branch_1_2/biases:0':'caps_relu_conv1/branch_1_2/biases:0',
					 'relu_conv1/branch_2/weights:0': 'caps_relu_conv1/branch_2/weights:0',
					 'relu_conv1/branch_2/biases:0': 'caps_relu_conv1/branch_2/biases:0',
					 'relu_conv1/branch_3/weights:0': 'caps_relu_conv1/branch_3/weights:0',
					 'relu_conv1/branch_3/biases:0':'caps_relu_conv1/branch_3/biases:0'
					 }
		var_dict = {}
		for old_name,new_name in name_dict.items():
			# v = tf.get_variable(old_name)
			# v = tf.get_collection(key =tf.GraphKeys.TRAINABLE_VARIABLES,scope=old_name)
			for v in var_list:
				if old_name == v.name:
					print(v.name)
					u =tf.Variavle(v, name="value")						# var_dict.update({v: new_name})
					saver = tf.train.Saver({old_name: u})
					break

		# print(var_dict)
		saver = tf.train.Saver(var_dict)
		saver.save(sess, save_path='/home/mo/work/seg_caps/my_hand_app/asl_model/asl_new/model_new' )

		# print('new name: ' + name + 'vuale: ', vary)

		#新名字
				# if 'branch' in name:
				#
				# 	new_name  ='casp'+name
				# 	v = tf.Variable(v, name=new_name)
				# 	name = v.name
				# 	print('new name: ' + name + 'vuale: ', vary)
				# 	all_name.append(name)

		# saver.save(sess, '/home/mo/work/seg_caps/my_hand_app/asl_model/asl_new')

		#7、打印结果
		# result = sess.run([output], {input: image})
		# print('all name: ' ,all_name)
		# print('已经恢复的result ',np.argmax(result,axis=-1))











