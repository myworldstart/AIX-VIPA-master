import matplotlib.pyplot as plt
import cv2
import sys

from asm.net.HQ_segment_anything import sam_model_registry


from asm.net.HQ_segment_anything import SamPredictor
import io
import json
import paddle
import urllib.request
from asm.predict import *
from flask import Blueprint, jsonify
from flask.globals import request
from asm.sam_utils import sam_find_board_V3, sam_find_board_V4

bp = Blueprint('HQ_sam_click', __name__)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    print(h, w)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


@paddle.no_grad()
@bp.route('/HQ_sam_seg', methods = ['POST'])
def click_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if('box' in request.json):
        img_path, input_boxes = request.json.get('img_path'), torch.tensor(request.json.get('box'), device=device)
    else:
        input_point, img_path = request.json.get('click_list'), request.json.get('img_path')
        input_boxes = None

    model_type = "vit_l"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=ModelSet['HQ_SAM']['weight'])
    sam.to(device=device)
    predictor = SamPredictor(sam)

    if img_path.startswith('http'):
        img_path = io.BytesIO(urllib.request.urlopen(img_path).read())
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image)
    hq_token_only = True
    if input_boxes != None:
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        masks, scores, logits = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
            hq_token_only=hq_token_only,
        )
        polygons = sam_find_board_V3(masks)


    else:
        point = []
        label = []
        for click in input_point:
            x = click['x']
            y = click['y']
            if(click['positive']):
                label.append(1)
            else:
                label.append(0)
            point.append([x, y])
        point = np.array(point)
        label = np.array(label)
        masks, scores, logits = predictor.predict(
            point_coords=point,
            point_labels=label,
            multimask_output=False,
            hq_token_only=hq_token_only,
        )
        polygons = sam_find_board_V4(masks)

    # print(len(polygons))
    # print(len(polygons[0]))
    # print(len(polygons[0][0]))
    for i in range(len(polygons)):
        for j in range(len(polygons[i])):
            x = polygons[i][j][0]
            y = polygons[i][j][1]
            image[y, x, :] = [0, 0, 0]

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    print(type(masks))
    print(masks.shape)
    for mask in masks:
        show_mask(mask, plt.gca(), random_color=True)
    plt.axis('off')
    plt.show()

    response = json.loads(json.dumps(polygons, default=lambda point: point.tolist()))
    return jsonify(response)
