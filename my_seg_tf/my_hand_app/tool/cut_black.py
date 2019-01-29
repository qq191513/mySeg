import cv2
import numpy as np
import time
from tool.visual_tool import plt_imshow_1_pics
def get_box(pics):
    image,pres,plot_pics, = pics
    if plot_pics:
        plt_imshow_1_pics(pres)
    #（1）用Sobel算子计算x，y方向上的梯度
    gradX = cv2.Sobel(pres, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    if plot_pics:
        plt_imshow_1_pics(gradX)

    gradY = cv2.Sobel(pres, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    if plot_pics:
        plt_imshow_1_pics(gradY)

    #（2）在x方向上减去y方向上的梯度，通过这个减法，我们留下具有高水平梯度和低垂直梯度的图像区域。
    gradient = cv2.subtract(gradX, gradY)
    if plot_pics:
        plt_imshow_1_pics(gradient)

    gradient = cv2.convertScaleAbs(gradient)
    if plot_pics:
        plt_imshow_1_pics(gradient)

    #（3）去除图像上的噪声。使用低通滤泼器平滑图像（9x9内核）,低通滤泼器作用是将每个像素替换为该像素周围像素的均值。
    blurred = cv2.blur(gradient, (9, 9))
    if plot_pics:
        plt_imshow_1_pics(blurred)

    #（4）对模糊图像二值化。梯度图像中不大于90的任何像素都设置为0（黑色）。 否则，像素设置为255（白色）。
    (_, thresh) = cv2.threshold(blurred, 50, 250, cv2.THRESH_BINARY)
    if plot_pics:
        plt_imshow_1_pics(thresh)

    #（5）分别执行4次形态学腐蚀与膨胀。去掉一些小的白色斑点
    closed = cv2.erode(thresh, None, iterations=2)
    if plot_pics:
        plt_imshow_1_pics(closed)
    # closed = cv2.dilate(closed, None, iterations=2)
    # plt_imshow_1_pics(closed)

    #（6）找出手势区域的轮廓，cv2.RETR_EXTERNAL表示只检测外轮廓,
    # 参数cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
    xxx = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #（7）主要求得包含点集最小面积的矩形，这个矩形是可以有偏转角度的，可以与图像的边界不平行。
    _a, cnts, _b = xxx
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    #（8）画出来看看
    cv2.drawContours(image, [box], -1, (0, 255, 0), 1)
    if plot_pics:
        plt_imshow_1_pics(image)


    #（9）红色矩形裁剪
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    if x1 < 0:
        x1 = 0
    if x2 < 0:
        x2 = 0
    if y1 < 0:
        y1 = 0
    if y2 < 0:
        y2 = 0

    height = y2 - y1
    width = x2 - x1
    if plot_pics:
        plt_imshow_1_pics(image)
        time.sleep(5) #休息5秒防止死机
    return x1,y1,height,width