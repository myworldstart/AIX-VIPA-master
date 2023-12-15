import json
import io
import torch
import urllib.request
from enum import Enum
from time import time
from PIL import Image, ImageDraw
import cv2
from tqdm import tqdm
import numpy as np


path = '/nfs/jzj/ASM_SHOW/models/unet/unet_latest.pth'

state_dict = torch.load(path)

newstate = dict()

newstate['model_state'] = state_dict

torch.save(newstate, '/nfs/jzj/ASM_SHOW/models/unet/unet.pth')