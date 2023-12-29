import cv2 as cv
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn

def AddNoise(imarray, probility=0.05, method="salt_pepper"):  # 灰度图像
    #获取图片的长和宽
    height, width = imarray.shape[:2]

    for i in range(height):
        for j in range(width):
            if np.random.random(1) < probility:  # 随机加盐或者加椒
                if np.random.random(1) < 0.5:
                    imarray[i, j] = 0
                else:
                    imarray[i, j] = 255
    return imarray

def auto_median_filter(image, max_size):
    origen = 3                                                        # 初始窗口大小
    board = origen//2                                                 # 初始应扩充的边界
    # max_board = max_size//2                                         # 最大可扩充的边界
    copy = cv.copyMakeBorder(image, *[board]*4, borderType=cv.BORDER_DEFAULT)         # 扩充边界
    out_img = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            def sub_func(src, size):                         # 两个层次的子函数
                kernel = src[i:i+size, j:j+size]
                # print(kernel)
                z_med = np.median(kernel)
                z_max = np.max(kernel)
                z_min = np.min(kernel)
                if z_min < z_med < z_max:                                 # 层次A
                    if z_min < image[i][j] < z_max:                       # 层次B
                        return image[i][j]
                    else:
                        return z_med
                else:
                    next_size = cv.copyMakeBorder(src, *[1]*4, borderType=cv.BORDER_DEFAULT)   # 增尺寸
                    size = size+2                                        # 奇数的核找中值才准确
                    if size <= max_size:
                        return sub_func(next_size, size)     # 重复层次A
                    else:
                        return z_med
            out_img[i][j] = sub_func(copy, origen)
    return out_img



img = cv.imread("/home/disk1/xjb/code/python/project/aix/1.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# print(img.shape)
# plt.figure(figsize=(10, 10))
# plt.imshow(img)
# plt.title('彩色图')
# plt.show()
# img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
# plt.figure(figsize=(10, 10))
# plt.imshow(img, cmap='gray')
# plt.title('灰度图')
# print(img.shape)
# plt.show()
# img = AddNoise(img)
# print(img)
# plt.figure(figsize=(10, 10))
# plt.imshow(img, cmap='gray')
# plt.title('灰度图')
# plt.show()
img = auto_median_filter(img, 2)
plt.figure(figsize=(10, 10))
plt.imshow(img, cmap='gray')
plt.show()


