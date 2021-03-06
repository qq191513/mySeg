from v1 import config as cfg
import glob
import numpy as np
import cv2
import random
import os
def show_dataset( images_path , segs_path ,  n_classes ):

	assert images_path[-1] == '/'
	assert segs_path[-1] == '/'

	images = glob.glob( os.path.join(images_path ,"*.jpg" ) ) \
	+ glob.glob( os.path.join(images_path , "*.png" ) ) +  glob.glob( os.path.join(images_path, "*.jpeg" ) )
	images.sort()
	segmentations  =  glob.glob( os.path.join(segs_path ,"*.jpg" ) )\
	+ glob.glob( os.path.join(segs_path , "*.png" ) ) +  glob.glob( os.path.join(segs_path, "*.jpeg" ) )
	segmentations.sort()

	colors = [( random.randint(0,255),random.randint(0,255),random.randint(0,255)) for _ in range(n_classes)]

	assert len( images ) == len(segmentations)

	for im_fn , seg_fn in zip(images,segmentations):
		assert(  im_fn.split('/')[-1] ==  seg_fn.split('/')[-1] )

		img = cv2.imread( im_fn )
		seg = cv2.imread( seg_fn )
		print(np.unique( seg ))

		seg_img = np.zeros_like( seg )

		for c in range(n_classes):
			seg_img[:,:,0] += ( (seg[:,:,0] == c )*( colors[c][0] )).astype('uint8')
			seg_img[:,:,1] += ((seg[:,:,0] == c )*( colors[c][1] )).astype('uint8')
			seg_img[:,:,2] += ((seg[:,:,0] == c )*( colors[c][2] )).astype('uint8')

		cv2.imshow("img" , img )
		cv2.imshow("seg_img" , seg_img )
		cv2.waitKey(25)





show_dataset(cfg.train_images , cfg.train_annotations ,  cfg.train_batch_size)
show_dataset(cfg.val_images , cfg.val_annotations ,  cfg.train_batch_size)
