import os
import cv2
import pynvml
from copy import deepcopy
import numpy as np
from collections import defaultdict
from asm.generate_polygon import get_polygon
from config import HOST_UPLOADS


def seek_gpu():
    choose_gid = 0
    max_free_memory = 0
    pynvml.nvmlInit()
    GPUCount = pynvml.nvmlDeviceGetCount()
    for gid in range(GPUCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gid)    # GPU的id
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if max_free_memory < meminfo.free / 1024**3:
            max_free_memory = meminfo.free / 1024**3
            choose_gid = gid
    print('choose GPU {}, Free Memory {:.2f} GB'.format(gid, max_free_memory))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(choose_gid)
    pynvml.nvmlShutdown()

def map_docker2host(img_path):
    img_path = img_path.replace('/uploads', HOST_UPLOADS)
    return '.'.join(img_path.split('.')[:-2])

def int2float(pt, img_w, img_h):
    # return (y, x)
    return min(pt[0], img_h) * 1.0, min(pt[1], img_w) * 1.0

def Board2Path(points, img_w, img_h):
    lens = len(points)
    resultPath = []
    for idx in range(0, lens, 2):
        if idx == 0:
            y, x = int2float(points[idx], img_w, img_h)
            resultPath.append(['M', x, y])
        else:
            y1, x1 = int2float(points[idx - 1], img_w, img_h)
            y2, x2 = int2float(points[idx], img_w, img_h)
            resultPath.append(['Q', x1, y1, x2, y2])
    return resultPath

def find_board_V2(img):
    '''
    代码来源: 百度EISeg  https://github.com/PaddleCV-SIG/EISeg/blob/main/eiseg/util/polygon.py
    输入: 语义分割的分割图, 每个像素点的值是一个类别ID
    :return: 分割图的所有边缘点, 用于前端生成标注框
    '''

    img_cats = set(img.flatten().tolist())
    boards = defaultdict(list)

    for cat in img_cats:
        if cat == 0:
            continue
        img_signal_cat = deepcopy(img)
        img_signal_cat[img_signal_cat != cat] = 0
        img_signal_cat[img_signal_cat == cat] = 1
        cv2.imwrite('./ans.png', img_signal_cat * 255)
        res_all = get_polygon(np.array(img_signal_cat * 255, dtype=np.uint8))
        for res in res_all:
            res_cur = []
            for point in res:
                res_cur.append(list(point))
            boards[cat].append(res_cur)
    return boards, img_cats
