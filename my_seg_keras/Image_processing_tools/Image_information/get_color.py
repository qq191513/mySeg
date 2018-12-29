from PIL import Image

image = Image.open("/home/mo/work/seg_caps/Hand-Segmentation/GTEA_gaze_part/Resize/Images/0000022810carlos_pasta.jpg")
print(image,image.getcolors(),image.size)

image = Image.open("/home/mo/work/seg_caps/Hand-Segmentation/GTEA_gaze_part/Resize/Masks_1/0000022810carlos_pasta.png")
print(image,image.getcolors(),image.size)

# image = Image.open("/home/wdh/PycharmProjects/Semantic Segmentation/Pytorch-UNet-master-milesial_self/truth1.png")
# print(image)

import cv2
pic = cv2.imread("/home/mo/work/seg_caps/Hand-Segmentation/GTEA_gaze_part/Resize/Images/0000022810carlos_pasta.jpg")
# cv2.imshow('fuck', pic)
# cv2.waitKey(20000)
print(pic.shape)

dd = cv2.imread("/home/mo/work/seg_caps/Hand-Segmentation/GTEA_gaze_part/Resize/Masks_1/0000022810carlos_pasta.png")
# cv2.imshow('fuck', pic)
# cv2.waitKey(20000)
print(dd.shape)