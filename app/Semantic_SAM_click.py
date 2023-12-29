import io
import json
import urllib.request

import cv2
import numpy as np
import torch
from PIL.Image import Image
from matplotlib import pyplot as plt

from asm.net.semantic_sam import prepare_image, plot_multi_results, build_semantic_sam
from asm.net.semantic_sam.tasks import SemanticSAMPredictor

from config import ModelSet
from flask import Blueprint, jsonify
from flask.globals import request
from asm.sam_utils import sam_find_board_V3, sam_find_board_V4

bp = Blueprint('Semantic_SAM_click', __name__)


@torch.no_grad()
@bp.route('/Semantic_SAM_click', methods=['POST'])
def Semantic_SAM_click():
    input_points, img_path = request.json.get('click_list'), request.json.get('img_path')
    print("收到请求")
    # 读取图片
    original_image, input_image, h, w = prepare_image(image_pth=img_path)
    print("图片读取成功")
    mask_generator = SemanticSAMPredictor(
        build_semantic_sam(model_type=ModelSet['Semantic_SAM']['model_type'], ckpt=ModelSet['Semantic_SAM']['weight']))
    print("mask_generator创建成功")
    points = []
    for point in input_points:
        x = point['x']
        y = point['y']
        x = x / w
        y = y / h
        points.append([x, y])
    # points = np.array(points)
    # points = [[0.5, 0.5]]
    # print(type(points))
    mask = mask_generator.predict_masks(original_image, input_image, point=points)

    polygons = sam_find_board_V4(mask)

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i in range(len(polygons)):
        for j in range(len(polygons[i])):
            x = polygons[i][j][0]
            y = polygons[i][j][1]
            image[y, x, :] = [0, 0, 0]
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()


    response = json.loads(json.dumps(polygons, default=lambda point: point.tolist()))
    return jsonify(response)
