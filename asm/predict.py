import os
import torch
import numpy as np
from config import ModelSet
from torchvision.transforms import functional as F
from asm.utils import find_board_V2

model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')

def load_model(modelName):

    ckpt_path = ModelSet[modelName]['weight']
    print('load model ', modelName, 'pth ',ckpt_path)
    if(modelName != 'SAM'):
        model = ModelSet[modelName]['model'](num_classes=ModelSet[modelName]['args']['num_classes'])

    # load to gpu/cpu
    if torch.cuda.is_available():
        device = torch.device('cuda')
        state_dict = torch.load(ckpt_path)
        model.load_state_dict(state_dict['model_state'])
        model.to(device)
    else:
        device = torch.device('cpu')
        state_dict = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict['model_state'])
        model.to(device)

    print('load model done!')
    return model

def slide_img(img_tensor, windowWidth=1000, windowHeight=1000, stride=800, eval=False):
    _, channels, h, w = img_tensor.shape

    if h * w <= windowWidth * windowHeight:          
        if eval:                                     # 如果是测试阶段
            return img_tensor, 1, 1
        else:
            return [[img_tensor]], 1, 1
    
    imgs = []
    imgs_inner = []
    imgs_right = []
    imgs_bottom = []
    imgs_last = []
    st_x, st_y = 0, 0
    padding = windowWidth - stride
    cols = (w - padding) // stride if (w - padding) % stride == 0 else (w - padding) // stride + 1
    rows = (h - padding) // stride if (h - padding) % stride == 0 else (h - padding) // stride + 1

    while st_y + windowHeight < h:
        while st_x + windowWidth < w:
            if not eval:
                imgs_inner.append(img_tensor[:, :, st_y: st_y + windowHeight, st_x: st_x + windowWidth])
            else:
                imgs.append(img_tensor[:, :, st_y: st_y + windowHeight, st_x: st_x + windowWidth])
            st_x += stride
        if st_x < w and st_x + windowWidth >= w:
            if not eval:
                imgs_right.append(img_tensor[:, :, st_y: st_y + windowHeight, st_x: w])
            else:
                imgs.append(img_tensor[:, :, st_y: st_y + windowHeight, st_x: w])
        st_x = 0
        st_y += stride

    if st_y < h and st_y + windowHeight >= h:
        while st_x + windowWidth < w:
            if not eval:
                imgs_bottom.append(img_tensor[:, :, st_y: h, st_x: st_x + windowWidth])
            else:
                imgs.append(img_tensor[:, :, st_y: h, st_x: st_x + windowWidth])
            st_x += stride
        if st_x < w and st_x + windowWidth >= w:
            if not eval:
                imgs_last.append(img_tensor[:, :, st_y: h, st_x: w])
            else:
                imgs.append(img_tensor[:, :, st_y: h, st_x: w])

    if not eval:     # train step
        if len(imgs_inner):
            imgs.append(imgs_inner)
        if len(imgs_right):
            imgs.append(imgs_right)
        if len(imgs_bottom):
            imgs.append(imgs_bottom)
        if len(imgs_last):
            imgs.append(imgs_last)

    return imgs, rows, cols

@torch.no_grad()
def predict(model, img, device, stride=800, windowSize=1000):
    model.eval()
    h, w = img.size
    outputs = []
    padding = (windowSize - stride) // 2  

    img_tensor = F.to_tensor(img)                    
    img_tensor = torch.unsqueeze(img_tensor, 0)         # [1, C, H, W]
    imgs, rows, cols = slide_img(img_tensor, stride=stride, eval=True)
    
    if rows * cols == 1:
        img = imgs.to(device, dtype=torch.float32)
        detection = model(img)
        detection = torch.argmax(detection, 1)
        detection = torch.squeeze(detection).cpu().numpy()  # shape: [h, w]
    else:
        for index, img in enumerate(imgs):
            img = img.to(device, dtype=torch.float32)
            detection = model(img)
            detection = torch.argmax(detection, 1)
            detection = torch.squeeze(detection).cpu().numpy()  # shape: [h, w]
    
            if index % cols == 0:            
                if index == 0:                
                    detection = detection[0: -1 * padding, 0: -1 * padding]
                elif index == (rows - 1) * cols:   
                    detection = detection[padding:, 0: -1 * padding]
                else:
                    detection = detection[padding: -1 * padding, 0: -1 * padding]
            elif (index + 1) % cols == 0:       
                if index + 1 == cols:           
                    detection = detection[0: -1 * padding, padding:]
                elif index + 1 == cols * rows:  
                    detection = detection[padding:, padding:]
                else:
                    detection = detection[padding: -1 * padding, padding:]
            elif index < cols:                  
                detection = detection[0: -1 * padding, padding: -1 * padding]
            elif index >= (rows - 1) * cols:   
                detection = detection[padding:, padding: -1 * padding]
            else:
                detection = detection[padding:-1 * padding, padding:-1 * padding]
    
            outputs.append(detection)
       
        rowImg = []
        for i in range(rows):
            rowImg.append(np.hstack(outputs[i*cols: (i+1)*cols]))
        detection = np.vstack(rowImg)

    board, cat = find_board_V2(detection)

    return board, cat
