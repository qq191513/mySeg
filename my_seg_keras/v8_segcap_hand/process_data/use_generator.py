import cv2
import glob
import itertools
import numpy as np
import os
import sys
sys.path.append('../')
from v1 import config as cfg

##########################   改这里   #######################################
#使用二值化标签(h,w,1)还是多分类标签(h,w,class)
use_binary_label = False
#标签做成一个向量
label_vec = False
#debug模式每次只训练20张图
debug =cfg.debug
##########################   end   #######################################
def get_image_arr( path ,height,width,imgNorm="sub_mean"):
	##简单处理原图，图片归一化

	img = cv2.imread(path, 1)
	if imgNorm == "sub_and_divide":
		img = np.float32(cv2.resize(img, (width,height))) / 127.5 - 1
	elif imgNorm == "sub_mean":
		img = cv2.resize(img, (width,height ))
		img = img.astype(np.float32)
		img[:,:,0] -= 103.939
		img[:,:,1] -= 116.779
		img[:,:,2] -= 123.68
	elif imgNorm == "divide":
		img = cv2.resize(img, (width,height))
		img = img.astype(np.float32)
		img = img/255.0

	return img

def get_label_arr( path , nClasses ,height,width):
	# 处理标签图
	# 每个种类作为一个数字，最终就变成维度是width * height * class的三维张量（空白的是0, 有目标的都是1）
	# 最终再flatten成 向量 * class的二维向量
	seg_labels = np.zeros(( height,width, nClasses ))
	try:
		label = cv2.imread(path, 1)
		label = cv2.resize(label, (width,height))
		label = label[:, : , 1]  #png分割图（标签）的第0个维度（有0，1，2维度）就是0到nClasses的维度

		for c in range(nClasses):
			#(img == c).astype(int)这句话是True则为1
			fuck =(label == c)
			fuck1 =(label == c).astype(int)
			seg_labels[: , : , c ] = (label == c ).astype(int)

	except Exception as e:
		print(e)
	if label_vec:
		seg_labels = np.reshape(seg_labels, (width*height,nClasses))
	else:
		seg_labels = np.reshape(seg_labels, (width ,height,nClasses))
	return seg_labels

def get_label_binary_arr( path , nClasses ,height,width):
	#专用二值化形式便签（也只能用于分割单个目标了）
	# 返回维度是width * height * 1 的三维张量（空白的是0, 有目标的都是1）
	# 最终再flatten成 向量 * class的二维向量
	try:
		label = cv2.imread(path, 1)
		label = cv2.resize(label, (width,height))
		seg_labels = label[:, : , 1]  #png分割图（标签）的第0个维度（有0，1，2维度）就是0到nClasses的维度

	except Exception as e:
		print(e)
	if label_vec:
		seg_labels = np.reshape(seg_labels, ( width*height ,1))
	else:
		seg_labels = np.reshape(seg_labels, (width , height, 1))
	return seg_labels


def get_gray_arr(path, height, width, imgNorm="divide"):
	#灰度图，填-1读取出来的就是单通道了，否则是三通道
	gray = cv2.imread(path, -1)
	if imgNorm == "sub_and_divide":
		gray = np.float32(cv2.resize(gray, (width,height))) / 127.5 - 1

	elif imgNorm == "divide":
		gray = cv2.resize(gray, (width,height))
		gray = gray.astype(np.float32)
		gray = gray/255.0
	if label_vec:
		gray = np.reshape(gray, (width*height,1))
	else:
		gray = np.reshape(gray, (width ,height,1))

	return gray

def imageSegmentationGenerator( images_path , segs_path ,  batch_size,  n_classes ,input_height,input_width ,output_height, output_width ):

	images = glob.glob( os.path.join(images_path ,"*.jpg" ) ) \
			 + glob.glob( os.path.join(images_path , "*.png" ) ) +  glob.glob( os.path.join(images_path, "*.jpeg" ) )
	images.sort()
	segmentations  =  glob.glob( os.path.join(segs_path ,"*.jpg" ) ) \
					  + glob.glob( os.path.join(segs_path , "*.png" ) ) +  glob.glob( os.path.join(segs_path, "*.jpeg" ) )
	segmentations.sort()

	assert len( images ) == len(segmentations)
	for im , seg in zip(images,segmentations):
		assert(im.split('/')[-1].split(".")[0] ==  seg.split('/')[-1].split(".")[0] )

	zipped = itertools.cycle(zip(images,segmentations) )

	while True:
		X = []
		Y = []
		for _ in range(batch_size):
			im, seg = next(zipped)
			X.append(get_image_arr(im, input_height, input_width, imgNorm="divide"))
			if use_binary_label:
				Y.append(get_label_binary_arr(seg, n_classes, output_height, output_width))
			else:
				Y.append(get_label_arr(seg, n_classes, output_height, output_width))
		yield np.array(X), np.array(Y)



def X_Y_Z_iter( images_path , segs_path ,  batch_size,  n_classes ,input_height,input_width ,output_height, output_width ):
# Z代表返回了想要的文件名，用于保存图片的命名

	images = glob.glob( os.path.join(images_path ,"*.jpg" ) ) + glob.glob( os.path.join(images_path , "*.png" ) ) +  glob.glob( os.path.join(images_path, "*.jpeg" ) )
	images.sort()
	segmentations  =  glob.glob( os.path.join(segs_path ,"*.jpg" ) ) + glob.glob( os.path.join(segs_path , "*.png" ) ) +  glob.glob( os.path.join(segs_path, "*.jpeg" ) )
	segmentations.sort()
	file_name_list = []

	assert len( images ) == len(segmentations)
	for img , seg in zip(images,segmentations):
		assert(img.split('/')[-1].split(".")[0] ==  seg.split('/')[-1].split(".")[0] )
		file_name_list.append(img.split('/')[-1].split(".")[0])

	if debug:
		images = images[0:10]
		segmentations = segmentations[0:10]
		file_name_list= file_name_list[0:10]
	zipped = iter(zip(images,segmentations,file_name_list) )


	while True:
		X = []
		Y = []
		Z = []
		for _ in range( batch_size):
			im , seg, file_name = next(zipped)
			X.append(get_image_arr(im, input_height, input_width, imgNorm="divide") )
			Y.append( get_label_arr( seg , n_classes ,output_height,output_width ))
			Z.append(file_name)
		yield np.array(X) , np.array(Y), np.array(Z)


def X_Y_G_Z_iter( images_path , segs_path ,gray_path, batch_size,  n_classes ,input_height,input_width ,output_height, output_width ):
# Z代表返回了想要的文件名，用于保存图片的命名
# G返回了想要的灰度图，用于重构

	images = glob.glob(os.path.join(images_path, "*.jpg")) + glob.glob(os.path.join(images_path, "*.png")) + glob.glob(
		os.path.join(images_path, "*.jpeg"))
	images.sort()
	segmentations = glob.glob(os.path.join(segs_path, "*.jpg")) + glob.glob(os.path.join(segs_path, "*.png")) + glob.glob(
		os.path.join(segs_path, "*.jpeg"))
	segmentations.sort()
	gray_list = glob.glob(os.path.join(gray_path, "*.jpg")) + glob.glob(os.path.join(gray_path, "*.png")) + glob.glob(
		os.path.join(gray_path, "*.jpeg"))
	gray_list.sort()
	file_name_list = []

	assert len(images) == len(segmentations)
	for im, seg ,gra in zip(images, segmentations,gray_list):
		assert (im.split('/')[-1].split(".")[0] == seg.split('/')[-1].split(".")[0])
		assert (im.split('/')[-1].split(".")[0] == gra.split('/')[-1].split(".")[0])
		file_name_list.append(im.split('/')[-1].split(".")[0])

	zipped = iter(zip(images, segmentations, gray_list,file_name_list))

	while True:
		X = []
		Y = []
		G = []
		Z = []
		for _ in range(batch_size):
			im, seg, gray,file_name = next(zipped)
			X.append(get_image_arr(im, output_height, output_width, imgNorm="divide"))
			Y.append(get_label_binary_arr(seg, n_classes, output_height, output_width))

			Z.append(file_name)
		yield np.array(X), np.array(Y), np.array(G),np.array(Z)


def segcap_data_Generator( images_path , segs_path ,gray_path,  batch_size,  n_classes ,input_height,input_width ,output_height, output_width ):

	images = glob.glob( os.path.join(images_path ,"*.jpg" ) ) \
			 + glob.glob( os.path.join(images_path , "*.png" ) ) +  glob.glob( os.path.join(images_path, "*.jpeg" ) )
	images.sort()
	labels_list  =  glob.glob( os.path.join(segs_path ,"*.jpg" ) ) \
					  + glob.glob( os.path.join(segs_path , "*.png" ) ) +  glob.glob( os.path.join(segs_path, "*.jpeg" ) )
	labels_list.sort()
	gray_list = glob.glob(os.path.join(gray_path, "*.jpg")) + glob.glob(os.path.join(gray_path, "*.png")) + glob.glob(
		os.path.join(gray_path, "*.jpeg"))
	gray_list.sort()

	assert len( images ) == len(labels_list)
	for im , seg in zip(images,labels_list):
		assert(im.split('/')[-1].split(".")[0] ==  seg.split('/')[-1].split(".")[0] )

	if debug:
		images = images[0:10]
		labels_list = labels_list[0:10]
		gray_list = gray_list[0:10]
	zipped = itertools.cycle(zip(images,labels_list,gray_list) )
	while True:
		X = []
		Y = []
		G = []

		for _ in range(batch_size):
			im, seg, gray = next(zipped)
			img = get_image_arr(im, input_height, input_width, imgNorm="divide")
			G.append(get_gray_arr(gray, output_height, output_width, imgNorm="divide"))
			X.append(img)
			if use_binary_label:
				label_binary = get_label_binary_arr(seg, n_classes, output_height, output_width)
				Y.append(label_binary)
			else:
				label_classes = get_label_arr(seg, n_classes, output_height, output_width)
				Y.append(label_classes)
			# mul.append(img*label_binary)

		yield ([np.array(X), np.array(Y)],[np.array(Y), np.array(G)])