import os
import sys
from asm.net import deeplab
from asm.net.PSPNet.pspnet import PSPNet
from asm.net.unet import UNet
from datetime import timedelta

SECRET_KEY = 'vipa-asm'
SESSION_TYPE = 'filesystem'
PERMANENT_SESSION_LIFETIME = timedelta(hours=1)        # session有效时间

# docker path map
# HOST_UPLOADS = '/disk3/xwx/apps/vipa-dataturks/client/build/uploads'
# HOST_UPLOADS = 'http://10.214.211.207:3030/uploads'
HOST_UPLOADS = 'http://10.214.211.207:3030/uploads'
# database
host = '10.214.211.207'
port = 3366
db = 'hope'
username, password = 'root', '123456'

# must +pymysql
SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://{}:{}@{}:{}/{}'.format(username, password, host, port, db)
SQLALCHEMY_COMMIT_TEARDOWN = True  # auto commit db changes
SQLALCHEMY_TRACK_MODIFICATIONS = False
SQLALCHEMY_POOL_RECYCLE = 1200

# model
ModelSet = {
    'deeplabv3_resnet50': {
        'model':  deeplab.modeling.deeplabv3_resnet50,
        'weight': '/nfs/jzj/ASM_SHOW/models/deeplabv3_resnet50/deeplabv3_resnet50_latest.pth',
        'args': {
            'num_classes': 21,
            'labels': ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                       'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
        },
        'type': 'segmentation',
        'source': 'Pascal VOC',
    },

    'deeplabv3_medical': {
        'model': deeplab.modeling.deeplabv3plus_resnet101,
        'weight': '/nfs/jzj/ASM_SHOW/models/deeplabv3_medical/deeplabv3_medical_latest.pth',
        'args': {
            'num_classes': 2,
            'labels': ['background', 'Positive'],
        },
        'type': 'segmentation',
        'source': '医学图像',
    },

    'pspnet_resnet50': {
        'model': PSPNet,
        'weight': '/nfs/jzj/ASM_SHOW/models/pspnet_resnet50/pspnet_resnet50_latest.pth',
        'args': {
            'num_classes': 21,
            'labels': ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                       'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
        },
        'type': 'segmentation',
        'source': 'Pascal VOC',
    },    
    
    'unet': {
        'model': UNet,
        'weight': '/nfs/jzj/ASM_SHOW/models/unet/unet_latest.pth',
        'args': {
            'num_classes': 2,
            'labels': ['background', 'Car'],
        },
        'type': 'segmentation',
        'source': '车辆数据集',
    },
    'SAM':{
        'weight': '/nfs/xjb/weights/sam_vit_h_4b8939.pth',
        'type': 'segmentation',
        'model_type': 'vit_h',
    }
}

SAVE_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
