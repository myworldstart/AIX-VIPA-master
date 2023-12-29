import io
import json

import paddle
import sys

from PIL.Image import Image
from matplotlib import pyplot as plt

import urllib.request
from asm.predict import *
from flask import Blueprint, jsonify
from flask.globals import request
from eiseg.models import EISegModel
from eiseg.controller import InteractiveController

bp = Blueprint('eiseg_click', __name__)

control = InteractiveController(
    predictor_params={
        "brs_mode": "NoBRS",
        "zoom_in_params": {
            "skip_clicks": -1,
            "target_size": (400, 400),
            "expansion_ratio": 1.4,
        },
        "predictor_params": {"net_clicks_limit": None, "max_size": 800},
    },
    prob_thresh=0.5,
)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)



@bp.route('/eiseg', methods=['POST'])
@paddle.no_grad()
def click_test():
    control.model = EISegModel('/home/disk1/xjb/code/python/project/aix/weight/static_hrnet18_ocr64_cocolvis.pdmodel',
                               '/home/disk1/xjb/code/python/project/aix/weight/static_hrnet18_ocr64_cocolvis.pdiparams')
    # ------------------------------------------------------------------------------------------------------
    # control.reset_predictor()
    # ----------------------------------------------------------------------------------------
    clicks, img_path = request.json.get('click_list'), request.json.get('img_path')
    # img_path = map_docker2host(img_path)[:-14]

    print(control.model)
    if img_path.startswith('http'):
        img_path = io.BytesIO(urllib.request.urlopen(img_path).read())
        img = np.array(Image.open(img_path).convert('RGB'))
    else:
        img = np.array(Image.open(img_path).convert('RGB'))
    # img = img.astype(np.float32)
    x = paddle.to_tensor(1)
    # --------------------------------
    # img = np.random.random((3, 2, 2)) * 20
    # img = paddle.ones([2, 2])
    # img = img.astype("float32")

    # --------------------------------
    print(np.unique(img))
    print("now")
    control.setImage(img)
    # --------------------
    ss = []
    s = []
    # -------------------------------
    for click in clicks:
        # print("开始增加")
        control.addClick(x=click['x'], y=click['y'], is_positive=click['positive'])
        # print(1)
        # ---------------------------------------------------------------------------------
        # ss.append([click['x'], click['y']])
        # s.append(1)
    print(np.unique(control.current_object_prob))
    # --------------------------------------------------------------------------------------
    mask, polygons = control.finishObject()
    print(polygons)
    print(mask)
    print(type(polygons))
    print(img.shape)
    for i in range(len(polygons)):
        for j in range(len(polygons[i])):
            img[polygons[i][j][1], polygons[i][j][0], :] = [0, 0, 0]
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    # ss = np.array(ss)
    # s = np.array(s)
    # show_points(ss, s, plt.gca())
    plt.axis('on')
    plt.show()
    # cv2.imwrite('./mask.png', np.uint8(mask * 255))
    print('预测完成')
    response = json.loads(json.dumps(polygons, default=lambda point:point.tolist()))
    return jsonify(response)


# def click_test1(img_path, clicks):
#     control.model = EISegModel('/nfs/xjb/weights/static_hrnet18_ocr64_cocolvis.pdmodel', '/nfs/xjb/weights/static_hrnet18_ocr64_cocolvis.pdiparams')
#     # clicks, img_path = request.json.get('click_list'), request.json.get('img_path')
#     # img_path = map_docker2host(img_path)[:-14]
#
#     if img_path.startswith('http'):
#         img_path = io.BytesIO(urllib.request.urlopen(img_path).read())
#         img = np.array(Image.open(img_path).convert('RGB'))
#     else:
#         img = np.array(Image.open(img_path).convert('RGB'))
#
#     control.setImage(img)
#     for click in clicks:
#         control.addClick(x=click['x'], y=click['y'], is_positive=click['positive'])
#     mask, polygons = control.finishObject()
#     # cv2.imwrite('./mask.png', np.uint8(mask * 255))
#     print('预测完成')
#     response = json.loads(json.dumps(polygons, default=lambda point:point.tolist()))
#     return jsonify(response)
#
#
# if __name__ == '__main__':
#     click_test1(img_path='/home/disk1/xjb/code/python/project/aix/ww.jpg', clicks=[[25, 30]])