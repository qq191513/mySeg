import cv2
import numpy as np
import sys
sys.path.append('../')

import tensorflow as tf
import tools.config.config_unet as cfg
import matplotlib.pyplot as plt
import time
from keras.preprocessing.image import random_rotation, random_shift, random_zoom, random_shear
from scipy.ndimage.interpolation import map_coordinates

##########################   要改的东西   #######################################
#显示方式
cv2_show = False  # 用opencv显示或plt显示
#######################     end     ############################################
GRAYSCALE = True
RESOLUTION = 512
COCO_BACKGROUND = (68, 1, 84, 255)
MASK_BACKGROUND = (0, 0, 0, 0)
DEFAULT_RGB_SCALE_FACTOR = 256000.0
DEFAULT_GRAY_SCALE_FACTOR = {np.uint8: 100.0,
                             np.uint16: 1000.0,
                             np.int32: DEFAULT_RGB_SCALE_FACTOR}


# Function to distort image
def elastic_transform(image, alpha=2000, sigma=40, alpha_affine=40, random_state=None):
  if random_state is None:
    random_state = np.random.RandomState(None)

  shape = image.shape
  shape_size = shape[:2]

  # Random affine
  center_square = np.float32(shape_size) // 2
  square_size = min(shape_size) // 3
  pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                     center_square - square_size])
  pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
  M = cv2.getAffineTransform(pts1, pts2)
  for i in range(shape[2]):
    image[:, :, i] = cv2.warpAffine(image[:, :, i], M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
  image = image.reshape(shape)

  blur_size = int(4 * sigma) | 1

  dx = cv2.GaussianBlur((random_state.rand(*shape_size) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
  dy = cv2.GaussianBlur((random_state.rand(*shape_size) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma) * alpha

  x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
  indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

  def_img = np.zeros_like(image)
  for i in range(shape[2]):
    def_img[:, :, i] = map_coordinates(image[:, :, i], indices, order=1).reshape(shape_size)

  return def_img


def salt_pepper_noise(image, salt=0.2, amount=0.004):
  row, col, chan = image.shape
  num_salt = np.ceil(amount * row * salt)
  num_pepper = np.ceil(amount * row * (1.0 - salt))

  for n in range(chan // 2):  # //2 so we don't augment the mask
    # Add Salt noise
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[0:2]]
    image[coords[0], coords[1], n] = 1

    # Add Pepper noise
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[0:2]]
    image[coords[0], coords[1], n] = 0

  return image


def flip_axis(x, axis):
  x = np.asarray(x).swapaxes(axis, 0)
  x = x[::-1, ...]
  x = x.swapaxes(0, axis)
  return x


def image2float_array(image, scale_factor=None):

  image_array = np.array(image)
  image_dtype = image_array.dtype
  image_shape = image_array.shape

  channels = image_shape[2] if len(image_shape) > 2 else 1
  assert 2 <= len(image_shape) <= 3

  if channels == 3:
    # RGB image needs to be converted to 24 bit integer.
    float_array = np.sum(image_array * [65536, 256, 1], axis=2)
    if scale_factor is None:
      scale_factor = DEFAULT_RGB_SCALE_FACTOR
  else:
    if scale_factor is None:
      scale_factor = DEFAULT_GRAY_SCALE_FACTOR[image_dtype.type]
    float_array = image_array.astype(np.float32)
  scaled_array = float_array / scale_factor

  return scaled_array


def image_resize2square(image, desired_size=None):
  '''
  Transform image to a square image with desired size(resolution)
  Padding image with black color which defined as MASK_BACKGROUND
  '''

  # initialize dimensions of the image to be resized and
  # grab the image size
  old_size = image.shape[:2]

  # if both the width and height are None, then return the
  # original image
  if desired_size is None or (old_size[0] == desired_size and old_size[1] == desired_size):
    return image

  # calculate the ratio of the height and construct the
  # dimensions
  ratio = float(desired_size) / max(old_size)
  new_size = tuple([int(x * ratio) for x in old_size])

  # new_size should be in (width, height) format
  resized = cv2.resize(image, (new_size[1], new_size[0]))

  delta_w = desired_size - new_size[1]
  delta_h = desired_size - new_size[0]
  top, bottom = delta_h // 2, delta_h - (delta_h // 2)
  left, right = delta_w // 2, delta_w - (delta_w // 2)

  new_image = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=MASK_BACKGROUND)

  # return the resized image
  return new_image


def image_enhance(image, shift):
  '''
  Input image is a numpy array with unit8 grayscale.
  This function will enhance the bright by adding num to each pixel.
  perform normalization
  '''
  if shift > 0:
    for i in range(shift):
      image += 1
      # If pixel value == 0 which means the value = 256 but overflow to 0
      # shift the overflow pix values to 255.
      image[image == 0] = 255

  return image


def process_image(img, shift, resolution):
  '''
  Pre-process image before store in numpy file.
	  shift: shift all pixels a distance with the shift value to avoid black color in image.
	  resolution: change image resolution to fit model.
  '''
  # Add 5 for each pixel on the grayscale image.
  img = image_enhance(img, shift=shift)

  # The source image should be 512X512 resolution.
  img = image_resize2square(img, resolution)

  return img


def change_background_color(img, original_color, new_color):
  '''
  Convert mask color of 4 channels png image to new color
  '''

  r1, g1, b1, a1 = original_color[0], original_color[1], original_color[2], original_color[3]  # Original value
  # mask background color (0,0,0,0)
  r2, g2, b2, a2 = new_color[0], new_color[1], new_color[2], new_color[3]  # Value that we want to replace it with

  red, green, blue, alpha = img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]
  mask = (red == r1) & (green == g1) & (blue == b1) & (alpha == a1)
  img[:, :, :4][mask] = [r2, g2, b2, a2]
  return img


def convert_mask_data(mask, resolution=RESOLUTION, from_background_color=COCO_BACKGROUND,
                      to_background_color=MASK_BACKGROUND):
  '''
  1. Resize mask to square with size of resolution.
  2. Change back ground color to black
  3. Change pixel value to 1 for masking
  4. Change pixel value to 0 for non-masking area
  5. Reduce data type to uint8 to reduce the file size of mask.
  '''
  mask = image_resize2square(mask, resolution)

  mask = change_background_color(mask, from_background_color, to_background_color)
  if GRAYSCALE == True:
    # Only need one channel for black and white
    mask = mask[:, :, :1]
  else:
    mask = mask[:, :, :1]  # keep 3 channels for RGB. Remove alpha channel.

  mask[mask >= 1] = 1  # The mask. ie. class of Person
  mask[mask != 1] = 0  # Non Person / Background
  mask = mask.astype(np.uint8)
  return mask


def convert_img_data(img, dims=4, resolution=RESOLUTION):
  '''
  Convert image data by
  1. Shift RGB channel with value 1 to avoid pure black color.
  2. Resize image to square
  3. Normalized data
  4. reshape to require dimension 3 or 4
  '''
  img = img[:, :, :3]
  if GRAYSCALE == True:
    # Add 1 for each pixel and change resolution on the image.
    img = process_image(img, shift=1, resolution=resolution)

    # Translate the image to 24bits grayscale by PILLOW package
    img = image2float_array(img, 16777216 - 1)  # 2^24=16777216
    if dims == 3:
      # Reshape numpy from 2 to 3 dimensions
      img = img.reshape([img.shape[0], img.shape[1], 1])
    else:  # dimension = 4
      img = img.reshape([1, img.shape[0], img.shape[1], 1])
  else:  # Color image with 3 channels
    # Add 1 for each pixel and change resolution on the image.
    img = process_image(img, shift=1, resolution=resolution)
    if dims == 3:
      # Keep RGB channel, remove alpha channel
      img = img[:, :, :3]
    else:  # dimensions = 4
      img = img[:, :, :, :3]
  return img
def plt_imshow_data(data):
  # 调成标准格式和标准维度，免得爆BUG
  data = np.asarray(data)
  if data.ndim == 3:
    if data.shape[2] == 1:
      data = data[:, :, 0]
  plt.imshow(data)
  plt.show()

def augmentImages(batch_of_images, batch_of_masks):
  for i in range(len(batch_of_images)):
    img_and_mask = np.concatenate((batch_of_images[i, ...], batch_of_masks[i, ...]), axis=2)
    if img_and_mask.ndim == 4:  # This assumes single channel data. For multi-channel you'll need
      # change this to put all channel in slices channel
      orig_shape = img_and_mask.shape
      img_and_mask = img_and_mask.reshape((img_and_mask.shape[0:3]))

    if np.random.randint(0, 10) == 7:
      img_and_mask = random_rotation(img_and_mask, rg=45, row_axis=0, col_axis=1, channel_axis=2,
                                     fill_mode='constant', cval=0.)

    # if np.random.randint(0, 5) == 3:
    # img_and_mask = elastic_transform(img_and_mask, alpha=1000, sigma=80, alpha_affine=50)
    #这个elastic_transform导致显示的东西像屎一样，干脆不要了
    if np.random.randint(0, 10) == 7:
      img_and_mask = random_shift(img_and_mask, wrg=0.2, hrg=0.2, row_axis=0, col_axis=1, channel_axis=2,
                                  fill_mode='constant', cval=0.)

    if np.random.randint(0, 10) == 7:
      img_and_mask = random_shear(img_and_mask, intensity=16, row_axis=0, col_axis=1, channel_axis=2,
                                  fill_mode='constant', cval=0.)

    if np.random.randint(0, 10) == 7:
      img_and_mask = random_zoom(img_and_mask, zoom_range=(0.75, 0.75), row_axis=0, col_axis=1, channel_axis=2,
                                 fill_mode='constant', cval=0.)

    if np.random.randint(0, 10) == 7:
      img_and_mask = flip_axis(img_and_mask, axis=1)

    if np.random.randint(0, 10) == 7:
      img_and_mask = flip_axis(img_and_mask, axis=0)

    if np.random.randint(0, 10) == 7:
      salt_pepper_noise(img_and_mask, salt=0.2, amount=0.04)

    if batch_of_images.ndim == 4:
      batch_of_images[i, ...] = img_and_mask[..., 0:3]
      batch_of_masks[i, ...] = np.expand_dims(img_and_mask[...,3],axis=-1)


    # Ensure the masks did not get any non-binary values.
    batch_of_masks[batch_of_masks > 0.5] = 1
    batch_of_masks[batch_of_masks <= 0.5] = 0

  return (batch_of_images, batch_of_masks)

def augmentImages_test():
  # bug：包和包之间无法互相调用，只能注释另一个，或者放from xxx.preprocess import xxx放这里
  from data_process.use_seg_tfrecord import create_inputs_seg_hand
  images, masks = create_inputs_seg_hand(is_train=True)
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
      batch_size_n = 1  # 观察batch_size的第n张图片
      pics, pics_masks = sess.run([images, masks])  # 取出一个batchsize的图片
      ##########################   主要调用的东西   ############################
      pics = pics / 255
      x_aug, y_aug = augmentImages(pics, pics_masks)
      ##########################   end   #######################################

      pics = np.split(x_aug, cfg.batch_size, axis=0)  # 按图片数量切分batchsize张
      pic = pics[batch_size_n]
      # pic = pic  255
      pic = np.squeeze(pic, axis=0)  # 去掉维度是1的维度，四维变成三维

      pics_masks = np.split(y_aug, cfg.batch_size, axis=0)  # 按图片数量切分batchsize张
      pics_mask = pics_masks[batch_size_n]
      pics_mask = np.squeeze(pics_mask, axis=0)  # 去掉维度是1的维度，四维变成三维
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
#if  __name__== '__main__':这句化要加上去，否则导入包的时候会运行 augmentImages_test()函数


if  __name__== '__main__':
  augmentImages_test()








