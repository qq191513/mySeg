# coding = utf-8
from PIL import Image
import os

def convert(dir,width,height):
    file_list = os.listdir(dir)
    print(file_list)
    for filename in file_list:
        path = ''
        path = dir+filename
        im = Image.open(path)
        out = im.resize((width,height),Image.ANTIALIAS)
        #out = im.convert("P")
        print("%s has been resized!"%filename)
        out.save(path)

if __name__ == '__main__':
#   dir = input('please input the operate dir:')
    dir = "/home/mo/work/seg_caps/Hand-Segmentation/test_images/"
    convert(dir,1918,1280)
   #  convert(dir,256,256)