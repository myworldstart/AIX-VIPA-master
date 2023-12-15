from IPython.display import display, HTML
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append("/home/disk1/xjb/code/python/project/aix/asm/net")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    print(sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1])
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

image = cv2.imread('/home/disk1/xjb/code/python/project/aix/ww.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20,20))
plt.imshow(image)
plt.axis('off')
plt.show()


sam_checkpoint = "/nfs/xjb/weights/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda:1"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

masks = mask_generator.generate(image)

print(image.shape)
print(len(masks))
print(masks[0].keys())
print(masks[1]['segmentation'].shape)
plt.figure(figsize=(20,20))
print(image.shape[0])
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()
plt.close()