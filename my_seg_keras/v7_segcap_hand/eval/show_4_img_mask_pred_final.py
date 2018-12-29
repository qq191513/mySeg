import sys
sys.path.append('../')
from v1 import config as cfg
import glob
import numpy as np
import cv2
import random
import os
from keras import backend as K
##########################   改这里   #######################################

# images = cfg.val_images
#  = cfg.val_annotations
images_path =cfg.train_images
label_path = cfg.train_label
pre_color_pics ="../../output_predict_result/unet_hand_64_64/pre_color_pics"
pre_class_pics ="../../output_predict_result/unet_hand_64_64/pre_class_pics"

merge_show = True
show_size = (300,300)
show_pos = (10, 10)
fon_size = 0.5
########################        end      ########################################



def show_dataset( images_path , segs_path , pre_color_pics,pre_class_pics, n_classes ):
    font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
    images = glob.glob( os.path.join(images_path ,"*.jpg" ) ) + glob.glob( os.path.join(images_path , "*.png" ) ) +  glob.glob( os.path.join(images_path, "*.jpeg" ))
    images.sort()
    segmentations = glob.glob( os.path.join(segs_path ,"*.jpg" ) )+ glob.glob( os.path.join(segs_path , "*.png" ) ) +  glob.glob( os.path.join(segs_path, "*.jpeg" ))
    segmentations.sort()
    pre_color = glob.glob(os.path.join(pre_color_pics, "*.jpg"))+ glob.glob(os.path.join(pre_color_pics, "*.png")) + glob.glob(os.path.join(pre_color_pics, "*.jpeg"))
    pre_color.sort()
    pre_class = glob.glob(os.path.join(pre_class_pics, "*.jpg"))+ glob.glob(os.path.join(pre_class_pics, "*.png")) + glob.glob(os.path.join(pre_class_pics, "*.jpeg"))
    pre_class.sort()

    colors = [( random.randint(0,255),random.randint(0,255),random.randint(0,255)) for _ in range(n_classes)]

    assert len(images) == len(segmentations)
    assert len(images) == len(pre_color)
    assert len(images) == len(pre_class)
    i = 0
    len_img = len(images)
    for im_fn,seg_fn,pre_color_fn,pre_class_fn in zip(images,segmentations,pre_color,pre_class):
        i += 1

        #1、读原图
        img_1 = cv2.imread( im_fn )
        # 添加文字，fon_size 表示字体大小，（0,40）是初始的位置，(255,255,255)表示颜色，2表示粗细
        img = cv2.putText(img_1, 'img',show_pos, font, fon_size, (255, 255,255),1)

        #2、读label
        seg = cv2.imread( seg_fn )
        print('%d/%d' % (i,len_img),np.unique( seg ))
        seg_img = np.zeros_like( seg )
        for c in range(n_classes):
            seg_img[:,:,0] += ( (seg[:,:,0] == c )*( colors[c][0] )).astype('uint8')
            seg_img[:,:,1] += ((seg[:,:,0] == c )*( colors[c][1] )).astype('uint8')
            seg_img[:,:,2] += ((seg[:,:,0] == c )*( colors[c][2] )).astype('uint8')

        # 添加文字，1.2表示字体大小，（0,40）是初始的位置，(255,255,255)表示颜色，2表示粗细
        seg_img = cv2.putText(seg_img, 'label', show_pos, font,  fon_size, (255, 255,255),1)

        #3、读预测图
        h,w,c= seg.shape
        pre_color = cv2.imread(pre_color_fn)
        # pre_color = cv2.resize(pre_color, (w, h))
        # 添加文字，1.2表示字体大小，（0,40）是初始的位置，(255,255,255)表示颜色，2表示粗细
        pre_color = cv2.putText(pre_color, 'predict', show_pos, font,  fon_size, (255, 255,255),1)

        #4、分割效果
        pre_class = cv2.imread(pre_class_fn)
        final_result = np.multiply(img,pre_class)
        final_result = cv2.putText(final_result, 'result', show_pos, font,  fon_size, (255, 255,255),1)



        if show_size:
            img = cv2.resize(img,show_size,interpolation=cv2.INTER_CUBIC)
            seg_img = cv2.resize(seg_img,show_size,interpolation=cv2.INTER_CUBIC)
            pre_color = cv2.resize(pre_color,show_size,interpolation=cv2.INTER_CUBIC)
            final_result = cv2.resize(final_result, show_size, interpolation=cv2.INTER_CUBIC)

        if merge_show:
            hstack_1 = np.hstack((img,seg_img))  # 水平拼接
            hstack_2 = np.hstack((pre_color,final_result))  # 水平拼接
            merge_4 = np.vstack((hstack_1,hstack_2))  # 垂直拼接
            cv2.imshow('show', merge_4 )
        else:
            cv2.imshow("img" , img )
            cv2.imshow("seg_img" , seg_img )
            cv2.imshow("pre_color" , pre_color )
            cv2.imshow("final_result" , final_result)
        cv2.waitKey(500)
    # 4、程序结束退出，不加这个会报错,因为config文件设置了GPU模式，但是这里又不去使用它
    K.clear_session()




# show_dataset(cfg.train_images , cfg.train_annotations ,pre_color_color_path,  cfg.train_batch_size)
show_dataset(images_path , label_path , pre_color_pics,pre_class_pics, cfg.n_classes)
