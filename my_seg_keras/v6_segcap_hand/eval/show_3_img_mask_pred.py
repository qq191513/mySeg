import sys
sys.path.append('../')
import glob
import numpy as np
import cv2
import random
import os
from keras import backend as K
##########################   改这里   #######################################

# images = cfg.val_images
#  = cfg.val_annotations
images_path = "/home/mo/work/seg_caps/my_seg_keras/dataset/hand_128_128/Images"
label_path = "/home/mo/work/seg_caps/my_seg_keras/dataset/hand_128_128/Masks_1"
pre_out_path  ="../../output_predict_result/hand_128_128/test/"
vstack_pics = False
hstack_pics = True
show_size = (512,512)
########################        end      ########################################



def show_dataset( images_path , segs_path , pre_out_path, n_classes ):
    font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
    images = glob.glob( os.path.join(images_path ,"*.jpg" ) ) + glob.glob( os.path.join(images_path , "*.png" ) ) +  glob.glob( os.path.join(images_path, "*.jpeg" ))
    images.sort()
    segmentations = glob.glob( os.path.join(segs_path ,"*.jpg" ) )+ glob.glob( os.path.join(segs_path , "*.png" ) ) +  glob.glob( os.path.join(segs_path, "*.jpeg" ))
    segmentations.sort()
    pre = glob.glob(os.path.join(pre_out_path, "*.jpg"))+ glob.glob(os.path.join(pre_out_path, "*.png")) + glob.glob(os.path.join(pre_out_path, "*.jpeg"))
    pre.sort()

    colors = [( random.randint(0,255),random.randint(0,255),random.randint(0,255)) for _ in range(n_classes)]

    assert len( images ) == len(segmentations) ==len(pre)

    for im_fn,seg_fn,pre_fn in zip(images,segmentations,pre):

        show_pos = (10, 10)
        #1、读原图
        img = cv2.imread( im_fn )
        # 添加文字，1.2表示字体大小，（0,40）是初始的位置，(255,255,255)表示颜色，2表示粗细
        img = cv2.putText(img, 'img',show_pos, font, 0.5, (255, 255,255),1)


        #2、读label
        seg = cv2.imread( seg_fn )
        print(np.unique( seg ))
        seg_img = np.zeros_like( seg )
        for c in range(n_classes):
            seg_img[:,:,0] += ( (seg[:,:,0] == c )*( colors[c][0] )).astype('uint8')
            seg_img[:,:,1] += ((seg[:,:,0] == c )*( colors[c][1] )).astype('uint8')
            seg_img[:,:,2] += ((seg[:,:,0] == c )*( colors[c][2] )).astype('uint8')

        # 添加文字，1.2表示字体大小，（0,40）是初始的位置，(255,255,255)表示颜色，2表示粗细
        seg_img = cv2.putText(seg_img, 'label', show_pos, font,  0.5, (255, 255,255),1)

        #3、读预测图
        h,w,c= seg.shape
        pre = cv2.imread(pre_fn)
        pre = cv2.resize(pre, (w, h))
        # 添加文字，1.2表示字体大小，（0,40）是初始的位置，(255,255,255)表示颜色，2表示粗细
        pre = cv2.putText(pre, 'predict', show_pos, font,  0.5, (255, 255,255),1)

        if show_size:
            img = cv2.resize(img,show_size,interpolation=cv2.INTER_CUBIC)
            seg_img = cv2.resize(seg_img,show_size,interpolation=cv2.INTER_CUBIC)
            pre = cv2.resize(pre,show_size,interpolation=cv2.INTER_CUBIC)

        if vstack_pics:
            vmerge = np.vstack((img,seg_img, pre))  # 垂直拼接
            cv2.imshow("1: img  2: label 3: predict" , vmerge )
        elif hstack_pics:
            hstack = np.hstack((img,seg_img, pre))  # 垂直拼接
            cv2.imshow("1: img  2: label 3: predict" , hstack )
        else:
            cv2.imshow("img" , img )
            cv2.imshow("seg_img" , seg_img )
            cv2.imshow("pre" , pre )
        cv2.waitKey(1000)
    # 4、程序结束退出，不加这个会报错,因为config文件设置了GPU模式，但是这里又不去使用它
    K.clear_session()




# show_dataset(cfg.train_images , cfg.train_annotations ,pre_out_path,  cfg.train_batch_size)
show_dataset(images_path , label_path , pre_out_path, 10)
