import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import time
import os
import sys
sys.path.append('../')

from choice import cfg
example_name = {}
##########################   要改的东西   #######################################
#tfrecords文件的路径
tfrecord_path = '/home/mo/work/seg_caps/my_seg_tf/dataset/my_128/had_labled_out_tf'

#要还原原先图片尺寸
origenal_size =[128,128,3]
mask_size = [128,128,1]


bbox = None

#多队列、多线程、batch读图部分
num_threads = 8
batch_size = cfg.batch_size
shuffle_batch =True
#训练多少轮，string_input_producer的num_epochs就写多少，
#否则会爆出OutOfRangeError的错误（意思是消费量高于产出量）
epoch = cfg.epoch

#显示方式
cv2_show = False  # 用opencv显示或plt显示
#######################     end     ############################################

def int64_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def ReadTFRecord(tfrecords,example_name):
    if len(tfrecords) == 1:
        record_queue = tf.train.string_input_producer(tfrecords,num_epochs=epoch+1)#只有一个文件，谈不上打乱顺序
    else:
        # shuffle=False，num_epochs为3，即每个文件复制成3份，再打乱顺序，否则按原顺序
        record_queue = tf.train.string_input_producer(tfrecords,shuffle=True, num_epochs=epoch+1)

    reader = tf.TFRecordReader()
    key, value = reader.read(record_queue)
    features = tf.parse_single_example(value,
            features={
                # 取出key为img_raw和label的数据,尤其是int位数一定不能错!!!
                example_name['image']: tf.FixedLenFeature([],tf.string),
                example_name['mask']: tf.FixedLenFeature([], tf.string),
                example_name['height']: tf.FixedLenFeature([], tf.int64),
                example_name['width']: tf.FixedLenFeature([], tf.int64),
                example_name['channel']: tf.FixedLenFeature([], tf.int64)

            })

    img = tf.decode_raw(features[example_name['image']], tf.uint8)
    mask = tf.decode_raw(features[example_name['mask']], tf.uint8)


    # 注意定义的为int多少位就转换成多少位,否则容易出错!!

    if len(origenal_size) == 2:
        w, h = origenal_size[0],origenal_size[1]
    else:
        w, h, c = origenal_size[0],origenal_size[1],origenal_size[2]
        mask_w,mask_h,mask_c = mask_size[0],mask_size[1],mask_size[2]

    img = tf.reshape(img, [w,h,c])
    mask = tf.reshape(mask,[mask_w,mask_h,mask_c])
    # 加了这个tf.cast并不是图片归一化，纯粹的转换格式
    img = tf.cast(img, tf.float32)
    mask = tf.cast(mask, tf.float32)

    # return img
    return img, mask



def feed_data_method(image,mask):
    if shuffle_batch:
        images,masks = tf.train.shuffle_batch(
            [image,mask],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=batch_size*10,
            min_after_dequeue=batch_size*5,
            allow_smaller_final_batch=False)
    else:
        images,masks = tf.train.batch(
            [image,mask],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=batch_size*10,
            allow_smaller_final_batch=False)
    return images,masks

def plt_imshow_data(data):
    #调成标准格式和标准维度，免得爆BUG
    data = np.asarray(data)
    if data.ndim == 3:
        if data.shape[2] == 1:
            data = data[:, :, 0]
    plt.imshow(data)
    plt.show()

def plt_imshow_two_pics(data_1,data_2):
    #调成标准格式和标准维度，免得爆BUG
    data_1 = np.asarray(data_1)
    if data_1.ndim == 3:
        if data_1.shape[2] == 1:
            data_1 = data_1[:, :, 0]

    data_2 = np.asarray(data_2)
    if data_2.ndim == 3:
        if data_2.shape[2] == 1:
            data_2 = data_2[:, :, 0]

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(data_1)
    plt.subplot(1, 2, 2)
    plt.imshow(data_2)
    plt.show()

def get_files_list(path):
    # work：获取所有文件的完整路径
    files_list = []
    for parent,dirnames,filenames in os.walk(path):
        for filename in filenames:
            files_list.append(os.path.join(parent,filename))
    return files_list

#根据关键字筛选父目录下需求的文件，按列表返回全部完整路径
def search_keyword_files(path,keyword):
    keyword_files_list = []
    files_list = get_files_list(path)
    for file in files_list:
        if keyword in file:
            keyword_files_list.append(file)
    return keyword_files_list

def read_label_txt_to_dict(labels_txt =None):
    if os.path.exists(labels_txt):
        labels_maps = {}
        with open(labels_txt) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line[:-1]  # 去掉换行符
                line_split = line.split(':')
                labels_maps[line_split[0]] = line_split[1]
        return labels_maps
    return None
def show_loaded(data_tfrecord=None):
    print('load tfrecord:')
    for each in data_tfrecord:
        print(each)

def create_inputs_seg_hand(is_train):
    # 根据关键字搜索tfrecord_path目录下的所有tfrecord的相关文件，跟convert_to_segment_tfrecord.py一致，不用变
    train_keywords = 'train'
    test_keywords = 'validation'
    # 解码部分：填入解码键值和原图大小以便恢复，跟convert_to_segment_tfrecord.py一致，不用变
    example_name['image'] = 'image/encoded'
    example_name['mask'] = 'mask/encoded'
    example_name['height'] = 'image/height'
    example_name['width'] = 'image/width'
    example_name['channel'] = 'image/channel'

    if is_train:
        data_tfrecord = search_keyword_files(tfrecord_path,train_keywords)
    else:
        data_tfrecord = search_keyword_files(tfrecord_path,test_keywords)
    show_loaded(data_tfrecord)

    image,mask = ReadTFRecord(data_tfrecord,example_name)    #恢复原始数据
    images,masks = feed_data_method(image,mask)                   #喂图方式
    return images,masks

if  __name__== '__main__':
    # bug：包和包之间无法互相调用，只能注释另一个，或者放from xxx.preprocess import xxx放这里
    from data_process.preprocess import augmentImages
    images,masks = create_inputs_seg_hand(is_train = False)
    # 标签
    labels_txt_keywords = 'labels.txt'
    # labels_txt = search_keyword_files(tfrecord_path, labels_txt_keywords)
    # labels_maps = read_label_txt_to_dict(labels_txt=labels_txt[0])   #标签映射
    #观察自己设置的参数是否符合心意，合适的话在别的项目中直接调用 create_inputs_xxx() 函数即可喂数据

    session_config = tf.ConfigProto(
        device_count={'GPU': 0},
        gpu_options={'allow_growth': 1,
                     # 'per_process_gpu_memory_fraction': 0.1,
                     'visible_device_list': '0'},
        allow_soft_placement=True)  ##这个设置必须有，否则无论如何都会报cudnn不匹配的错误,BUG十分隐蔽，真是智障
    with tf.Session(config=session_config) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 输出100个batch观察
        for i in range(100):
            batch_size_n = 1  #观察batch_size的第n张图片
            pics,pics_masks = sess.run([images,masks])  # 取出一个batchsize的图片
            ##########################   数据增强   ###################################
            pics = pics / 255  # 归一化
            pics, pics_masks = augmentImages(pics,pics_masks)
            ##########################   end   #######################################
            pics = np.split(pics, batch_size, axis=0)  # 按图片数量切分batchsize张
            pic = pics[batch_size_n]
            pic = np.squeeze(pic,axis=0)  # 去掉维度是1的维度，四维变成三维

            pics_masks = np.split(pics_masks, batch_size, axis=0)  # 按图片数量切分batchsize张
            pics_mask = pics_masks[batch_size_n]
            pics_mask = np.squeeze(pics_mask,axis=0)  # 去掉维度是1的维度，四维变成三维
            pics_mask = cv2.cvtColor(pics_mask, cv2.COLOR_GRAY2BGR)  # 一通道转三通道
            hstack = np.hstack((pic, pics_mask))  # 水平拼接
            if cv2_show:
                cv2.imshow('img and mask', hstack)
                cv2.waitKey(2000)
                cv2.destroyAllWindows()
            else:
                plt_imshow_data(hstack)
                time.sleep(2)

        coord.request_stop()
        coord.join(threads)