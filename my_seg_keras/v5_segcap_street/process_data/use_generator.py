import cv2
import glob
import itertools
import numpy as np
import os

def getImageArr( path ,height,width,imgNorm="sub_mean"):
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

def getSegmentationArr( path , nClasses ,height,width):
# 处理标签图
# 每个种类作为一个数字，最终就变成维度是width * height * class的三维张量（空白的是0, 有目标的都是1）
# 最终再flatten成 向量 * class的二维向量
	seg_labels = np.zeros(( height,width, nClasses ))
	try:
		img = cv2.imread(path, 1)
		img = cv2.resize(img, (width,height))
		img = img[:, : , 1]  #png分割图（标签）的第0个维度（有0，1，2维度）就是0到nClasses的维度

		for c in range(nClasses):
			#(img == c).astype(int)这句话是True则为1
			fuck =(img == c)
			fuck1 =(img == c).astype(int)
			seg_labels[: , : , c ] = (img == c ).astype(int)

	except Exception as e:
		print(e)
		
	seg_labels = np.reshape(seg_labels, ( width*height, nClasses ))
	return seg_labels

def imageSegmentationGenerator( images_path , segs_path ,  batch_size,  n_classes ,input_height,input_width ,output_height, output_width ):

	images = glob.glob( os.path.join(images_path ,"*.jpg" ) ) \
	+ glob.glob( os.path.join(images_path , "*.png" ) ) +  glob.glob( os.path.join(images_path, "*.jpeg" ) )
	images.sort()
	segmentations  =  glob.glob( os.path.join(segs_path ,"*.jpg" ) )\
	+ glob.glob( os.path.join(segs_path , "*.png" ) ) +  glob.glob( os.path.join(segs_path, "*.jpeg" ) )
	segmentations.sort()

	assert len( images ) == len(segmentations)
	for im , seg in zip(images,segmentations):
		assert(im.split('/')[-1].split(".")[0] ==  seg.split('/')[-1].split(".")[0] )

	zipped = itertools.cycle(zip(images,segmentations) )

	while True:
		X = []
		Y = []
		for _ in range( batch_size):
			im , seg = next(zipped)
			X.append( getImageArr(im ,input_height,input_width, imgNorm="divide"))
			Y.append( getSegmentationArr( seg , n_classes ,output_height,output_width ))

		yield np.array(X) , np.array(Y)

def X_Y_Z_iter( images_path , segs_path ,  batch_size,  n_classes ,input_height,input_width ,output_height, output_width ):

    images = glob.glob( os.path.join(images_path ,"*.jpg" ) ) + glob.glob( os.path.join(images_path , "*.png" ) ) +  glob.glob( os.path.join(images_path, "*.jpeg" ) )
    images.sort()
    segmentations  =  glob.glob( os.path.join(segs_path ,"*.jpg" ) ) + glob.glob( os.path.join(segs_path , "*.png" ) ) +  glob.glob( os.path.join(segs_path, "*.jpeg" ) )
    segmentations.sort()
    file_name_list = []
    assert len( images ) == len(segmentations)
    for im , seg in zip(images,segmentations):
        assert(im.split('/')[-1].split(".")[0] ==  seg.split('/')[-1].split(".")[0] )
        file_name_list.append(im.split('/')[-1].split(".")[0])

    zipped = iter(zip(images,segmentations,file_name_list) )

    while True:
        X = []
        Y = []
        Z = []
        for _ in range( batch_size):
            im , seg, file_name = next(zipped)
            X.append(getImageArr(im, input_height, input_width, imgNorm="divide") )
            Y.append( getSegmentationArr( seg , n_classes ,output_height,output_width ))
            Z.append(file_name)
        yield np.array(X) , np.array(Y), np.array(Z)