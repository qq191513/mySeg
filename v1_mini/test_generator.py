import numpy as np
import matplotlib.pyplot as plt
import config  as cfg
import time
from use_generator import imageSegmentationGenerator

def split_batch_to_pic_list(img):
	#一个batch_size的四维张量图片分割成一组列表图片(三维)
	batch_size = img.shape[0]
	img_list = np.split(img,batch_size,axis=0)
	for index,img in enumerate(img_list):
		img = np.squeeze(img,axis=0)
		img_list[index] = img
	return img_list

def plt_imshow_data(data):
    #调成标准格式和标准维度，免得爆BUG
    data = np.asarray(data)

    if data.ndim == 3:
        if data.shape[2] == 1:
            data = data[:, :, 0]
    plt.imshow(data)
    plt.show()
    time.sleep(2)

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


    plt.subplot(1, 2, 1)
    plt.imshow(data_1)
    plt.subplot(1, 2, 2)
    plt.imshow(data_2)
    plt.show()

def seg_vec_to_pic(seg_vec,restore_pic_shape,colors,n_classes):
	#长度为w * h * n_classes的向量转w * h * 3的图片，随机生成颜色
	seg_img = np.zeros(shape=restore_pic_shape)
	the_shape = restore_pic_shape
	w, h = the_shape[0],the_shape[1]
	seg_vec = seg_vec.reshape((w, h, -1)) #w * h * n_classes 做成（w ，h ，n_classes）

	for c in range(n_classes):
		seg_img[:, :, 0] += ((seg_vec[:, :, 0] == c) * (colors[c][0])).astype('uint8')
		seg_img[:, :, 1] += ((seg_vec[:, :, 1] == c) * (colors[c][1])).astype('uint8')
		seg_img[:, :, 2] += ((seg_vec[:, :, 2] == c) * (colors[c][2])).astype('uint8')
	seg_img = seg_img / 255.0
	seg_img = seg_img.astype('float32')
	return seg_img

def use_generator_to_show(images_path , segs_path ,  batch_size,
	n_classes , input_height , input_width , output_height , output_width):
	batch_size_n = 0  #取第batch_size_n张图片观察
	plt.figure()
	colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in
			  range(n_classes)]
	#使用Generator，返回一个batch_size的 im_fn和seg_fn
	for im_fn , seg_vec in imageSegmentationGenerator(images_path , segs_path ,  batch_size,
	n_classes , input_height , input_width , output_height , output_width):
		pics_group = split_batch_to_pic_list(im_fn)  #batchsize切成图片列表
		pic = pics_group[batch_size_n]  #取第batch_size_n张图片观察
		# plt_imshow_data(pic)  #用plt显示
		print('img shape: ',im_fn.shape)
		print('seg shape: ',seg_vec.shape)
		seg_vec = split_batch_to_pic_list(seg_vec)  # batchsize切成图片列表
		seg_vec = seg_vec[batch_size_n]  # 取第batch_size_n张图片观察
		seg_img = seg_vec_to_pic(seg_vec,pic.shape,colors,n_classes)

		plt_imshow_two_pics(pic,seg_img)  #用plt显示
		time.sleep(1)


print('train dataset')
use_generator_to_show(cfg.train_images , cfg.train_annotations ,  cfg.train_batch_size,
			cfg.n_classes, cfg.input_shape[0], cfg.input_shape[1], cfg.output_shape[0],cfg.output_shape[1])

# print('valid dataset')
# use_generator_to_show(cfg.val_images , cfg.val_annotations ,  cfg.train_batch_size,
# 	cfg.n_classes , cfg.input_height , cfg.input_width , cfg.output_height , cfg.output_width)

