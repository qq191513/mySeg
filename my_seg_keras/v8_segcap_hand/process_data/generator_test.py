import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from v1 import config as cfg
import time
from data_process.use_generator import imageSegmentationGenerator
from data_process.use_generator import X_Y_G_Z_iter
from data_process.use_generator import use_binary_label
from data_process.use_generator import segcap_data_Generator
##########################   改这里   #######################################
images_path =cfg.train_images
label_path =  cfg.train_label
gray_path = cfg.gray_label
batch_size = cfg.train_batch_size
n_classes =2
input_height = cfg.input_shape[0]
input_width = cfg.input_shape[1]
output_height = cfg.output_shape[0]
output_width = cfg.output_shape[1]
##########################   end   #######################################

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
    h,w  = restore_pic_shape[0],restore_pic_shape[1]
    seg_vec = seg_vec.reshape((h,w, -1)) # h * w * n_classes 做成（h ，w ，n_classes）
    # 共n_classes层，每层都弄一种颜色
    for c in range(n_classes):
        seg_img[:, :, 0] += (seg_vec[:, :, c] * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += (seg_vec[:, :, c] * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += (seg_vec[:, :, c] * (colors[c][2])).astype('uint8')
    seg_img = seg_img / 255.0
    seg_img = seg_img.astype('float32')
    return seg_img

def use_generator_to_show(images_path , label_path ,  batch_size,
    n_classes ,input_height,input_width , output_height,output_width ):
    batch_size_n = 0  #取第batch_size_n张图片观察
    plt.figure()
    colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in
              range(n_classes)]
    #使用Generator，返回一个batch_size的 im_fn和seg_fn
    for im_fn , seg_vec in imageSegmentationGenerator(images_path , label_path , batch_size,
    n_classes , input_height ,input_width, output_height,output_width):
        # 1、原图
        print('return img shape: ', im_fn.shape)
        pics_group = split_batch_to_pic_list(im_fn)  # batchsize切成图片列表
        pic = pics_group[batch_size_n]  # 取第batch_size_n张图片观察

        # 2、label图
        if use_binary_label:
            n_classes=1
        print('return label shape: ', seg_vec.shape)
        seg_vec = split_batch_to_pic_list(seg_vec)  # batchsize切成图片列表
        seg_vec = seg_vec[batch_size_n]  # 取第batch_size_n张图片观察
        seg_img = seg_vec_to_pic(seg_vec, pic.shape, colors, n_classes)


        # 5、显示img和label
        plt_imshow_two_pics(pic, seg_img)  # 用plt显示
        time.sleep(1)

def use_x_y_g_z_to_show(images_path , label_path , gray_path, batch_size,
    n_classes ,input_height,input_width , output_height,output_width ):
    batch_size_n = 0  #取第batch_size_n张图片观察
    plt.figure()
    colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in
              range(n_classes)]

    for im_fn, seg_vec, G_vec, file_name in X_Y_G_Z_iter(images_path , label_path , gray_path,batch_size,
    n_classes , input_height ,input_width, output_height,output_width):
        #1、原图
        print('return img shape: ',im_fn.shape)
        pics_group = split_batch_to_pic_list(im_fn)  #batchsize切成图片列表
        pic = pics_group[batch_size_n]  #取第batch_size_n张图片观察


        #2、label图(不用于之前的生成器，这里不再使用（h,w,class）标签，而是使用二值化（h,w,1）标签，
        # 所以seg_vec_to_pic最后的参数为1)
        print('return label shape: ',seg_vec.shape)
        seg_vec = split_batch_to_pic_list(seg_vec)  # batchsize切成图片列表
        seg_vec = seg_vec[batch_size_n]  # 取第batch_size_n张图片观察
        seg_img = seg_vec_to_pic(seg_vec,pic.shape,colors,1)

        #3、灰图
        print('return G_vec shape: ',G_vec.shape)
        G_pics_group = split_batch_to_pic_list(G_vec)  #batchsize切成图片列表
        G_pic = G_pics_group[batch_size_n]  #取第batch_size_n张图片观察
        G_pic = G_pic.reshape((input_width, input_height, -1))   #把向量还原
        plt_imshow_data(G_pic)  # 用plt显示
        time.sleep(1)

        #4、文件名
        print('return file name: ',file_name)

        #5、显示img和label
        plt_imshow_two_pics(pic,seg_img)  #用plt显示
        time.sleep(1)

def use_segcap_data_Generator_to_show(images_path , label_path , gray_path, batch_size,
    n_classes ,input_height,input_width , output_height,output_width ):
    batch_size_n = 0  #取第batch_size_n张图片观察
    plt.figure()
    colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in
              range(n_classes)]

    for seg_out,recon in segcap_data_Generator(images_path , label_path , gray_path,batch_size,
    n_classes , input_height ,input_width, output_height,output_width):
        x = seg_out[0]
        y = seg_out[1]
        g = recon[1]
        #1、原图
        print('return img shape: ',x.shape)
        pics_group = split_batch_to_pic_list(x)  #batchsize切成图片列表
        pic = pics_group[batch_size_n]  #取第batch_size_n张图片观察


        #2、label图(不用于之前的生成器，这里不再使用（h,w,class）标签，而是使用二值化（h,w,1）标签，
        # 所以seg_vec_to_pic最后的参数为1)
        print('return label shape: ',y.shape)
        y = split_batch_to_pic_list(y)  # batchsize切成图片列表
        y = y[batch_size_n]  # 取第batch_size_n张图片观察
        seg_img = seg_vec_to_pic(y,pic.shape,colors,1)

        #3、灰图
        print('return G_vec shape: ',g.shape)
        G_pics_group = split_batch_to_pic_list(g)  #batchsize切成图片列表
        G_pic = G_pics_group[batch_size_n]  #取第batch_size_n张图片观察
        G_pic = G_pic.reshape((input_width, input_height, -1))   #把向量还原
        plt_imshow_data(G_pic)  # 用plt显示
        time.sleep(1)



        #5、显示img和label
        plt_imshow_two_pics(pic,seg_img)  #用plt显示
        time.sleep(1)

print('train dataset')
# use_generator_to_show(images_path , label_path ,  batch_size,
#     n_classes ,input_height,input_width , output_height,output_width )



# use_x_y_g_z_to_show(images_path , label_path ,gray_path,  batch_size,
#     n_classes ,input_height,input_width , output_height,output_width )


use_segcap_data_Generator_to_show(images_path , label_path , gray_path, batch_size,
    n_classes ,input_height,input_width , output_height,output_width )