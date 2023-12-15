import os
import cv2
import pynvml
from copy import deepcopy
import numpy as np
from collections import defaultdict
from asm.generate_polygon import get_polygon
from config import HOST_UPLOADS

def sam_find_board_V2(masks):
    '''
    代码来源: 百度EISeg  https://github.com/PaddleCV-SIG/EISeg/blob/main/eiseg/util/polygon.py
    输入: 语义分割的分割图, 每个像素点的值是一个类别ID
    :return: 分割图的所有边缘点, 用于前端生成标注框
    '''
    res_cur = []
    boards = []
    for i in range(len(masks)):
        img_signal_cat = deepcopy(masks[i]['segmentation'])
        # print(img_signal_cat.shape)
        img_signal_cat = img_signal_cat.astype(int)
        cv2.imwrite('./ans.png', img_signal_cat * 255)
        res_all = get_polygon(np.array(img_signal_cat * 255, dtype=np.uint8))
        for res in res_all:
            for point in res:
                res_cur.append(list(point))
            boards.append(res_cur)
    return boards


def sam_find_board_V3(masks):
    '''
    代码来源:
    输入: 语义分割的分割图, 每个像素点的值是一个类别ID
    :return: 分割图的所有边缘点, 用于前端生成标注框
    '''
    res_cur = []
    boards = []
    for i in range(len(masks)):
        img_signal_cat = deepcopy(masks[i, :, :])
        img_signal_cat = img_signal_cat.squeeze(0)
        img_signal_cat = img_signal_cat.to(device = "cpu").numpy()
        print(img_signal_cat.shape)
        print(type(img_signal_cat))
        # print(img_signal_cat.shape)
        img_signal_cat = img_signal_cat.astype(int)
        cv2.imwrite('./ans.png', img_signal_cat * 255)
        res_all = get_polygon(np.array(img_signal_cat * 255, dtype=np.uint8))
        for res in res_all:
            for point in res:
                res_cur.append(list(point))
            boards.append(res_cur)
    return boards


def sam_find_board_V4(masks):
    '''
    代码来源:
    输入: 语义分割的分割图, 每个像素点的值是一个类别ID
    :return: 分割图的所有边缘点, 用于前端生成标注框
    '''
    res_cur = []
    boards = []
    img_signal_cat = deepcopy(masks)
    img_signal_cat = img_signal_cat.squeeze(0)
    print(img_signal_cat.shape)
    print(type(img_signal_cat))
    # print(img_signal_cat.shape)
    img_signal_cat = img_signal_cat.astype(int)
    cv2.imwrite('./ans.png', img_signal_cat * 255)
    res_all = get_polygon(np.array(img_signal_cat * 255, dtype=np.uint8))
    for res in res_all:
        for point in res:
            res_cur.append(list(point))
        boards.append(res_cur)
    return boards