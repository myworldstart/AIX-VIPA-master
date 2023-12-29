import io
import json
import urllib.request

import cv2
import numpy as np
import torch
from PIL.Image import Image
from matplotlib import pyplot as plt
from sqlalchemy import text
from tqdm import tqdm

from app.db import db, connect_redis, query_d_hits, parse_row_to_dict
from asm.predict import *
import urllib.request
from time import time
from asm.utils import map_docker2host, Board2Path
from datetime import datetime
from asm.sam_utils import sam_find_board_V2
from asm.net.semantic_sam import prepare_image, plot_multi_results, build_semantic_sam
from asm.net.semantic_sam.tasks import SemanticSAMPredictor
from asm.net.semantic_sam.tasks import SemanticSamAutomaticMaskGenerator
import hashlib
from config import ModelSet
from flask import Blueprint, jsonify
from flask.globals import request
from asm.sam_utils import sam_find_board_V3, sam_find_board_V4
from app.db import db, connect_redis, query_d_hits, parse_row_to_dict
from asm.predict import *
import urllib.request
from time import time
from asm.utils import map_docker2host, Board2Path
from datetime import datetime
from asm.sam_utils import sam_find_board_V2

bp = Blueprint('Semantic_SAM_click', __name__)

@torch.no_grad()
@bp.route('/Semantic_SAM_predict', methods=['POST'])
def Semantic_SAM_predict():
    try:
        print(request.json)
        modelName, datasetName, projectId, uid = request.json.get('modelName'), request.json.get(
            'datasetName'), request.json.get('projectId'), request.json.get('uid')
    except Exception as e:
        return jsonify({'code': 500, 'info': 'Parameter wrong, check your parameters is vaild all ?'})

    redisClient = connect_redis()
    if redisClient.exists(f'{datasetName}_{projectId}_{uid}'):
        return jsonify({'code': 500, 'info': 'Task running already!'})
    redisClient.set(f'{datasetName}_{projectId}_{uid}', 'True')

    TaskId = hashlib.md5((str(modelName) + str(projectId) + str(uid)).encode('utf-8')).hexdigest()

    mask_generator = SemanticSamAutomaticMaskGenerator(build_semantic_sam(model_type='<model_type>', ckpt='</your/ckpt/path>')) # model_type: 'L' / 'T', depends on your checkpint

    rows = query_d_hits(projectId=projectId)
    t1 = time()
    TotalIters = rows.rowcount
    rows_bar = tqdm(rows)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for _, row in enumerate(rows_bar):
        hit_dict = parse_row_to_dict(row)
        img_path = map_docker2host(hit_dict['data'])
        original_image, input_image, img_h, img_w = prepare_image(image_pth=img_path)
        masks = mask_generator.generate(input_image)

        board = sam_find_board_V2(masks)

        result = []
        if (len(board) > 0):
            for i in range(len(board)):
                box_info = {
                    "type": "asm_path",
                    "angle": 0,
                    "path": Board2Path(board[i], img_w, img_h),
                    "imageWidth": img_w,
                    "imageHeight": img_h,
                }
                result.append(box_info)
                # --------------------------------------
                # for j in range(len(board[i])):
                #     x = board[i][j][0]
                #     y = board[i][j][1]
                #     img[y][x] = [0, 0, 0]
                # ----------------------------------------

            sql = "insert into d_hits_result " \
                  "(`hitId`, `projectId`, `result`, `userId`, `timeTakenToLabelInSec`, `notes`, `created_timestamp`, `updated_timestamp`, `predLabel`, `model`, `status`) " \
                  "values ({},'{}','{}','{}',{},'{}','{}','{}','{}','{}','{}')".format(
                hit_dict['id'],  # int
                hit_dict['projectId'],  # str
                json.dumps(result),  # str
                uid,  # str
                0,
                'auto-label',  # str
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # str
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                {},
                modelName,
                'al'
            )
            sql = text(sql)
            db.session.execute(sql)
            db.session.commit()
        UsedTime = rows_bar.format_dict['elapsed']
        CurrentIters = rows_bar.format_dict['n'] + 1
        TimeLeft = (CurrentIters / UsedTime) * (TotalIters - CurrentIters)
        redisClient.set(name=f'{uid}_{TaskId}', value=json.dumps(
            {'TaskId': TaskId, 'DatasetName': datasetName, 'TimeLeft': TimeLeft,
             'Progress': CurrentIters / TotalIters * 100}))
        # cv2.imwrite('a.png', img)
    t2 = time()
    redisClient.delete(f'{datasetName}_{projectId}_{uid}')
    print('Cost time', t2 - t1)
    return jsonify({'code': 200, 'info': 'success'})

