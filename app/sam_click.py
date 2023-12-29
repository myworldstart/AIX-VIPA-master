import matplotlib.pyplot as plt
import cv2
import sys


from asm.net.segment_anything import sam_model_registry, SamPredictor
import io
import json
import paddle
import urllib.request
from asm.predict import *
from flask import Blueprint, jsonify
from flask.globals import request
from asm.sam_utils import sam_find_board_V3, sam_find_board_V4

bp = Blueprint('sam_click', __name__)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

@paddle.no_grad()
@bp.route('/sam_seg', methods = ['POST'])
def click_test():
    # input_point, img_path, input_label, input_box = request.json.get('click_list'), request.json.get('img_path'), request.json.get('label'), request.json.get('box')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if('box' in request.json):
        img_path, input_boxes = request.json.get('img_path'), torch.tensor(request.json.get('box'), device=device)
    else:
        # input_point, img_path, input_label = torch.tensor(request.json.get('click_list'), device=device), request.json.get('img_path'), torch.tensor(request.json.get('label'), device=device)
        input_point, img_path= request.json.get('click_list'), request.json.get('img_path')
        # input_point = input_point.squeeze(1)
        # input_label = input_label.squeeze(1)
        input_boxes = None

    if img_path.startswith('http'):
        img_path = io.BytesIO(urllib.request.urlopen(img_path).read())
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # print(input_label.shape)
    # print(input_point.shape)
    # print(input_boxes.shape)
    # input_boxes.to(device = "cuda:1")
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=ModelSet['SAM']['weight'])
    sam.to(device=device)
    predictor = SamPredictor(sam)
    print(image.shape)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('on')
    plt.show()
    predictor.set_image(image)
    if input_boxes != None:
        # image = cv2.resize(image, None, fx=0.5, fy=0.5)
        # input_boxes = input_boxes / 2
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        print(transformed_boxes)
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,)
        print(masks)  # output: (1, 600, 900)
        polygons = sam_find_board_V3(masks)
    #     ----------------------------------------------
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box in input_boxes:
            show_box(box.cpu().numpy(), plt.gca())
        plt.axis('off')
        plt.show()
    # ---------------------------------------------------------------
    else:
        # print(input_point.shape)
        # print(input_label.shape)
        # point = input_point.to("cpu").numpy()
        # label = input_label.to("cpu").numpy()
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
        masks, _, _ = predictor.predict(
            point_coords=point,
            point_labels=label,
            multimask_output=False,
        )
        # --------------------------------------------------------
        print(masks.shape)  # output: (1, 600, 900)
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(masks, plt.gca())
        # show_points(input_point, input_label, plt.gca())
        plt.axis('off')
        plt.show()
        print(masks)  # output: (1, 600, 900)
        # --------------------------------------------------------------
        polygons = sam_find_board_V4(masks)

    print(polygons)
    print(type(polygons))
    # for i in range(len(polygons)):
    #     for j in range(len(polygons[i])):
    #         y = polygons[i][j][0]
    #         x = polygons[i][j][1]
    #         image[x, y, :] = [0, 0, 0]
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # show_mask(masks, plt.gca())
    # show_points(input_point, input_label, plt.gca())
    # plt.axis('off')
    # plt.show()
    response = json.loads(json.dumps(polygons, default=lambda point: point.tolist()))
    return jsonify(response)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # print(len(masks))
    # a = input_point.detach().cpu().numpy()
    # b = input_label.detach().cpu().numpy()
    # for i in range(input_boxes.shape[0]):
    #     show_points(a[i], b[i], plt.gca())
    # for mask in masks:
    #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    # for box in input_boxes:
    #     print(box)
    #     show_box(box.cpu().numpy(), plt.gca())
    # plt.axis('off')
    # plt.show()


# if __name__ == '__main__':
#     a = [[[50, 60]]]
#     b = [[1]]
#     c = [[80, 20, 130, 100]]
#     click_test(img_path='/home/disk1/xjb/code/python/project/aix/ww.jpg',
#                input_point = np.array(a),
#                input_label = np.array(b),
#                )
#     # click_test(img_path='/home/disk1/xjb/code/python/project/aix/ww.jpg',
#     #            input_point= torch.tensor(a, device="cuda:1"),
#     #            input_label = torch.tensor(b, device="cuda:1"),
#     #            input_boxes=torch.tensor(c, device="cuda:1"))
#
