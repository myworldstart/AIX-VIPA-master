import os
import sys
import io
from flask.globals import session
import torch
import urllib.request
import numpy as np
from PIL import Image
from app.db import db
from asm.tools.utils_bn import convert_tomask
import torch.utils.data as data

class Mvi(data.Dataset):
    def __init__(self, projectId, root='/disk3/xwx/apps/vipa-dataturks/client/build', transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.images = []
        self.masks = []

        sql = "select data, result from \
            (d_hits_result join d_hits on d_hits_result.hitId=d_hits.id and d_hits_result.model=d_hits.correctResult ) \
            where d_hits.projectId='{}' and d_hits_result.status='done'".format(projectId)

        res = db.session.execute(sql)
        db.session.close()
        maskDict = {}
        for row in res:
            data, result = row
            data = data[:-14]
            maskDict[data] = result
            self.images.append(self.root + data)
            self.masks.append(maskDict[data])

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        mask_size = image.size
        image = torch.tensor(np.array(image))
        image = image.permute(2, 0, 1)
        mask = self.masks[index]
        mask = convert_tomask(mask, mask_size)
        mask = np.array(mask)
        mask[mask > 0] = 1
        mask = torch.tensor(mask)

        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        mask[mask <= 0] = 0
        mask[mask > 0] = 255
        return mask

class Split(data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.transform = transform
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.transform:
            image, label = self.transform(image, label)
        return image, label

    def __len__(self):
        return len(self.images)