import math
import os
import random
import sys
from PIL import Image
import sys
# sys.path.append('./')
import tensorflow as tf
import matplotlib.image as mpimg # mpimg 用于读取图片

LABELS_FILENAME = 'labels.txt'
paramenmt_set = {}
example_name = {}

##########################要改的东西#######################################
paramenmt_set['src'] = '/home/mo/work/seg_caps/my_seg_tf/dataset/my_128/train/had_labled_out_img_128'  #源图片集
paramenmt_set['mask'] = '/home/mo/work/seg_caps/my_seg_tf/dataset/my_128/train/had_labled_out_label_128'  #对应的mask路径
paramenmt_set['dst_folder'] = '/home/mo/work/seg_caps/my_seg_tf/dataset/my_128/had_labled_out_tf' #存放的目录
paramenmt_set['dst_file'] = 'seg_hand' #生成的文件名前缀

paramenmt_set['_NUM_VALIDATION'] = 0 #测试张数
paramenmt_set['_RANDOM_SEED'] = 33  #随机洗乱种子
paramenmt_set['_NUM_SHARDS'] = 1 #tfrecord分割块数


#填入编码键值，自己随便取
example_name['image'] = 'image/encoded'
example_name['mask'] = 'mask/encoded'
example_name['height'] = 'image/height'
example_name['width'] = 'image/width'
example_name['channel'] = 'image/channel'
###########################################################################

def int64_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def float_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def image_to_tfexample(example_name,image_data, mask_data, width,height,channel):
  return tf.train.Example(features=tf.train.Features(feature={
      example_name['image']: bytes_feature(image_data),
      example_name['mask']: bytes_feature(mask_data),
      example_name['width']: int64_feature(width),
      example_name['height']: int64_feature(height),
      example_name['channel']:int64_feature(channel),
  }))


def write_label_file(labels_to_class_names, dataset_dir,
                     filename=LABELS_FILENAME):
  labels_filename = os.path.join(dataset_dir, filename)
  with tf.gfile.Open(labels_filename, 'w') as f:
    for label in labels_to_class_names:
      class_name = labels_to_class_names[label]
      f.write('%d:%s\n' % (label, class_name))

def has_labels(dataset_dir, filename=LABELS_FILENAME):
  return tf.gfile.Exists(os.path.join(dataset_dir, filename))

def read_label_file(dataset_dir, filename=LABELS_FILENAME):
  labels_filename = os.path.join(dataset_dir, filename)
  with tf.gfile.Open(labels_filename, 'rb') as f:
    lines = f.read().decode()
  lines = lines.split('\n')
  lines = filter(None, lines)

  labels_to_class_names = {}
  for line in lines:
    index = line.index(':')
    labels_to_class_names[int(line[:index])] = line[index+1:]
  return labels_to_class_names



def _get_filenames_and_classes(dataset_dir):
  # flower_root = os.path.join(dataset_dir, 'flower_photos')
  # directories = []
  # class_names = []
  # for filename in os.listdir(dataset_dir):
  #   path = os.path.join(dataset_dir, filename)
  #   if os.path.isdir(path):
  #     directories.append(path)
  #     class_names.append(filename)
  class_names=[]
  photo_filenames = []

  class_names.append(os.path.basename(dataset_dir))

  for filename in os.listdir(dataset_dir):
      filename =os.path.join(dataset_dir,filename)
      photo_filenames.append(filename)

  return photo_filenames, sorted(class_names)


def _get_dataset_filename(paramenmt_set, split_name, shard_id):
  output_filename = paramenmt_set['dst_file']+ '_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, paramenmt_set['_NUM_SHARDS'])
  return os.path.join(paramenmt_set['dst_folder'], output_filename)

def finddir(startdir, target):# 查找target关键字的文件是否在startdir目录之下，在则返回完整路径
    for new_dir in os.listdir(startdir):  # 列表出该目录下的所有文件(返回当前目录'.')
        # print(new_dir)
        if target in new_dir:
            print(" 找到啦！！！！！！！！！")
            dir_path = os.path.join(startdir,new_dir)
            return dir_path
    print(target + "不存在!")
    return None


def _convert_dataset(example_name,split_name, filenames, class_names_to_ids, paramenmt_set):
  assert split_name in ['train', 'validation']

  num_per_shard = int(math.ceil(len(filenames) / float(paramenmt_set['_NUM_SHARDS'])))

  with tf.Graph().as_default():

    # image_reader = ImageReader()
    session_config = tf.ConfigProto(
        device_count={'GPU': 0},
        gpu_options={'allow_growth': 1,
                     # 'per_process_gpu_memory_fraction': 0.1,
                     'visible_device_list': '0'},
        allow_soft_placement=True)  ##这个设置必须有，否则无论如何都会报cudnn不匹配的错误,BUG十分隐蔽，真是智障
    with tf.Session(config=session_config) as sess:
       for shard_id in range(paramenmt_set['_NUM_SHARDS']):
        output_filename = _get_dataset_filename(
        paramenmt_set, split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d\r\n' % (
                i+1, len(filenames), shard_id))
            sys.stdout.flush()

            # Read the filename:
            # image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()

            img = Image.open(filenames[i])
            img_raw = img.tobytes()  # 将图片转化为原生bytes

            # 求图片的whith,height,channel
            pic_size = get_pic_size(filenames[i])
            width, height, channel = pic_size
            print(' img_raw: ',pic_size)
            paramenmt_set['channels_number'] = channel  # 图片通道数

            #mask部分
            dirname = os.path.basename(filenames[i])
            dirname = dirname.split('.')[0]
            mask_dirname = finddir(paramenmt_set['mask'],dirname)

            # 求图片的whith,height,channel
            pic_size = get_pic_size(mask_dirname)
            width, height, channel = pic_size
            print(' mask: ',pic_size)

            mask_pic = Image.open(mask_dirname)
            # mask_pic_channel = mask_pic.layers
            mask_pic = mask_pic.tobytes()  # 将图片转化为原生bytes

            example = image_to_tfexample(
                example_name,img_raw, mask_pic, width, height,channel)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()

def _dataset_exists(dataset_dir):
  for split_name in ['train', 'validation']:
    for shard_id in range(paramenmt_set['_NUM_SHARDS']):
      output_filename = _get_dataset_filename(
          dataset_dir, split_name, shard_id)
      if not tf.gfile.Exists(output_filename):
        return False
  return True

def get_pic_size(picture):
    lena = mpimg.imread(picture)  # 读取和代码处于同一目录下的 lena.png
    (whith, height, channel) =lena.shape
    return whith, height, channel

def convert_pics_to_tfrecord(example_name,paramenmt_set):
  if not tf.gfile.Exists(paramenmt_set['dst_folder']):
    tf.gfile.MakeDirs(paramenmt_set['dst_folder'])

  # if _dataset_exists(paramenmt_set['dst']):
  #   print('Dataset files already exist. Exiting without re-creating them.')
  #   return

  photo_filenames, class_names = _get_filenames_and_classes(paramenmt_set['src'])
  class_names_to_ids = dict(zip(class_names, range(len(class_names))))

  # Divide into train and test:
  random.seed(paramenmt_set['_RANDOM_SEED'])
  random.shuffle(photo_filenames)
  random.shuffle(photo_filenames)
  random.shuffle(photo_filenames)


  training_filenames = photo_filenames[paramenmt_set['_NUM_VALIDATION']:]
  if len(training_filenames):
      _convert_dataset(example_name,'train', training_filenames, class_names_to_ids,
                       paramenmt_set)

  validation_filenames = photo_filenames[:paramenmt_set['_NUM_VALIDATION']]
  if len(validation_filenames):
      _convert_dataset(example_name,'validation', validation_filenames, class_names_to_ids,
                       paramenmt_set)

  # # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(class_names)), class_names))
  write_label_file(labels_to_class_names, paramenmt_set['dst_folder'] )

  print('\nFinished converting the dataset!')


# 开始转换
convert_pics_to_tfrecord(example_name,paramenmt_set)