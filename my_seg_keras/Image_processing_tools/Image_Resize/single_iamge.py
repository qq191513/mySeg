# coding = utf-8  
from PIL import Image

def  convert(width,height):
    im = Image.open("/home/mo/work/seg_caps/Hand-Segmentation/test_images/my.jpg")
    out = im.resize((width, height),Image.ANTIALIAS)
    out.save("/home/mo/work/seg_caps/Hand-Segmentation/test_images/my1.jpg")
if __name__ == '__main__':
    convert(256,256)