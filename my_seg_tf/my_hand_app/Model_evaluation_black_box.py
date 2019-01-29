import tensorflow as tf
import numpy as np
slim = tf.contrib.slim
layers_end_points =None

####################   改这里  ########################################
#导入模型
from slim.nets.vgg import vgg_16
from slim.nets.vgg import vgg_arg_scope

#要测试的如数尺寸
input_size  = [224,224,3]

# 给模型的输入口
inputs = tf.placeholder(tf.float32, shape=(1, input_size[0], input_size[1], input_size[2]))
num_class = 32

#建造模型
with slim.arg_scope(vgg_arg_scope()):
    net, end_points = vgg_16(inputs, num_class)

#只要有logits就可以评估了，中间过程layers_end_points有的话就显示，没有就不显示
logits =net
layers_end_points =end_points

####################   end  ###########################################

def show_parament_numbers():
    from functools import reduce
    from operator import mul
    def get_num_params():
        num_params = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            num_params += reduce(mul, [dim.value for dim in shape], 1)
        return num_params
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxx parament numbers is : %d' % get_num_params())
    print('')

def show_Layers(end_points = None):
    print("Layers")
    for k, v in end_points.items():
        print('name = {}, shape = {}'.format(v.name, v.get_shape()))
    print('')

def show_Parameters():
    print("Parameters")
    for v in slim.get_model_variables():
        print('name = {}, shape = {}'.format(v.name, v.get_shape()))
    print('')

def show_graph():
    writer = tf.summary.FileWriter("./logdir", graph=tf.get_default_graph())

if __name__ == "__main__":

    #显示参数量
    show_parament_numbers()

    #如果记录有每层的情况（顺序字典），则显示每层
    if end_points:
        show_Layers(end_points = end_points)

    #显示所有参数
    show_Parameters()

    #初始化
    init = tf.global_variables_initializer()
    session_config = tf.ConfigProto(
        device_count={'GPU': 0},
        gpu_options={'allow_growth': 1,
                     # 'per_process_gpu_memory_fraction': 0.1,
                     'visible_device_list': '0'},
        allow_soft_placement=True)
    with tf.Session(config=session_config) as sess:
        sess.run(init)
        for i in range(10):
            x = tf.random_normal([1, input_size[0], input_size[1], input_size[2]])
            x = sess.run(x)
            pred = sess.run(logits,feed_dict={inputs: x})
            print(np.argmax(pred, 1))