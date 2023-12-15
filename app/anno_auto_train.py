# -*- coding: utf-8 -*-
import os
import json
from flask import Blueprint, session, Flask, g
from torch import nn
from datetime import datetime
from app import app
from flask_sockets import Sockets
from config import SAVE_MODEL_DIR
from torch.utils.data import DataLoader
from app.db import db, valid_userId
from asm.predict import *
from asm.tools.metric import StreamSegMetrics
from asm.net import deeplab, unet
from asm.tools import scheduler, losses, ext_transforms, dataset
from asm.tools.utils import mkdir

bp = Blueprint('anno_train', __name__)
sockets = Sockets(app)
model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')  # __file__ 记录当前文件路径

model_map = {
    'deeplabv3_resnet50': deeplab.deeplabv3_resnet50,
    'deeplabv3plus_resnet50': deeplab.deeplabv3plus_resnet50,
    'deeplabv3_resnet101': deeplab.deeplabv3_resnet101,
    'deeplabv3plus_resnet101': deeplab.deeplabv3plus_resnet101,
    'deeplabv3_mobilenet': deeplab.deeplabv3_mobilenet,
    'deeplabv3plus_mobilenet': deeplab.deeplabv3plus_mobilenet,
    'UNet': unet.UNet,
}


def isValid(modelName, projectID, userId):
    if not modelName or not projectID or not userId:
        return False
    if modelName not in model_map.keys():
        return False
    return valid_userId(userId)


@sockets.route('/auto_label_train')
def auto_label_train(ws):
    print('Socket connect')
    response = json.loads(ws.receive())

    modelName, projectID, userId, config = response['modelName'], response['projectID'], response['userId'], response['config']

    if not isValid(modelName, projectID, userId):
        print(modelName, projectID, userId)
        ws.send('model and projectID or model not exists')
        return
        
    # try:    
    session[f'{modelName}_{projectID}_{userId}_train'] = str(datetime.now())
    sql = f"select name from d_projects where d_projects.id='{projectID}'"
    res = db.session.execute(sql).fetchone()
    datasetName = res[0]

    # Load model
    model = model_map[modelName](num_classes=int(config['num_classes']))

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * float(config['lr'])},
        {'params': model.classifier.parameters(), 'lr': float(config['lr'])},
    ],
        lr=float(config['lr']),
        momentum=0.9,
        weight_decay=1e-4,
    )

    # device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    model.to(device)
    model = nn.DataParallel(model)

    # Set up metrics
    metrics = StreamSegMetrics(int(config['num_classes']))

    # Set up scheduler
    scheduler_train = scheduler.PolyLR(optimizer, int(config['total_itrs']), power=0.9)

    # Set up dataset and dataloader
    trainDataset = dataset.Mvi(projectId=projectID)
    train_loader = DataLoader(trainDataset, batch_size=int(config['batch_size']), shuffle=True, num_workers=0, drop_last=True)
    # Set up criterion
    if config['loss_type'] == 'focal_loss':
        criterion = losses.FocalLoss(ignore_index=255, size_average=True)
    elif config['loss_type'] == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    cur_itrs = 0
    cur_epochs = 0
    model.train()
    while True:
        cur_epochs += 1
        for images, labels in train_loader:
            cur_itrs += 1
            images, _, _ = slide_img(images, windowWidth=500, windowHeight=500, stride=500)
            labels = torch.unsqueeze(labels, 1)          # [N, 1, H, W]
            labels, _, _ = slide_img(labels, windowWidth=500, windowHeight=500, stride=500)

            for index, image in enumerate(images):
                image = torch.cat(image, 0).to(device, dtype=torch.float32)
                label = torch.cat(labels[index], 0).to(device, dtype=torch.long)

                splitDataset = dataset.Split(images=image, labels=label)
                splitloader = DataLoader(splitDataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True)

                splitBar = tqdm(splitloader)
                for image, label in splitBar:
                    optimizer.zero_grad()
                    outputs = model(image)
                    loss = criterion(outputs, torch.squeeze(label))
                    splitBar.set_postfix(loss = loss.item())
                    loss.backward()
                    optimizer.step()

            if (cur_itrs) % int(config['save_iters']) == 0:
                mkdir(os.path.join(SAVE_MODEL_DIR, f'{modelName}_{datasetName}'))
                print('Model saved')
                torch.save({
                    "cur_itrs": cur_itrs,
                    "model_state": model.module.state_dict() if device.type == 'cuda' else model.state_dict() ,
                    "optimizer_state": optimizer.state_dict(),
                }, os.path.join(SAVE_MODEL_DIR, f'{modelName}', f'{modelName}_{datasetName}_latest.pth'))

            scheduler_train.step()

            if cur_itrs >= int(config['total_itrs']):
                session.pop(f'{modelName}_{projectID}_{userId}_train')
                ws.send('Train success')
                return
    # except Exception as e:
    #     session.pop(f'{modelName}_{projectID}_{userId}')
    #     print('Exception:', e)
    #     ws.send(f'{e}')