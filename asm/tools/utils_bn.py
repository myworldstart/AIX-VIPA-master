from torchvision.transforms.functional import normalize
import torch.nn as nn
import json
import numpy as np
import PIL.Image
from PIL import ImageDraw
import os

def denormalize(tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean/std
    _std = 1/std
    return normalize(tensor, _mean, _std)

class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)

def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum

def fix_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def convert_tomask(res, mask_size):
    t = json.loads(res)
    mask = np.zeros((mask_size[1], mask_size[0]), dtype=np.uint8)
    # mask = np.zeros(( int(t[0]['height']) + 1, int(t[0]['width']) + 1 ), dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)

    for idx in range(len(t)):
        xy = []
        if (t[idx]['type'] == 'polygon' or t[idx]['type'] == 'rect'):
            for idx_points in range(len(t[idx]['actualPoints'])):
                xy.append((t[idx]['actualPoints'][idx_points][0], t[idx]['actualPoints'][idx_points][1]))
            draw.polygon(xy=xy, outline=1, fill=255)
        elif (t[idx]['type'] == 'path'):
            for idx_points in range(len(t[idx]['actualPoints'])):
                if (t[idx]['actualPoints'][idx_points][0] == 'M'):
                    xy.append((t[idx]['actualPoints'][idx_points][1], t[idx]['actualPoints'][idx_points][2]))
                elif (t[idx]['actualPoints'][idx_points][0] == 'L'):
                    xy.append((t[idx]['actualPoints'][idx_points][1], t[idx]['actualPoints'][idx_points][2]))
                    draw.polygon(xy=xy, outline=1, fill=255)
                    xy = []
                else:
                    xy.append((t[idx]['actualPoints'][idx_points][1], t[idx]['actualPoints'][idx_points][2]))
                    xy.append((t[idx]['actualPoints'][idx_points][3], t[idx]['actualPoints'][idx_points][4]))
        elif (t[idx]['type'] == 'circle'):
            for idx_points in range(len(t[idx]['actualPoints'])):
                draw.ellipse((t[idx]['actualPoints'][idx_points][0] - 3, t[idx]['actualPoints'][idx_points][1] - 3,
                              t[idx]['actualPoints'][idx_points][0] + 3, t[idx]['actualPoints'][idx_points][1] + 3),
                             outline=1, fill=255)
    return mask.convert('L')
