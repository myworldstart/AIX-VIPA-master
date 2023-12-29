import cv2
import matplotlib.pyplot as plt
import io
import json
import time
import hashlib
from flask import Blueprint, jsonify
from flask.globals import request
from sqlalchemy import text
import sys

from tqdm import tqdm


from app.db import db, connect_redis, query_d_hits, parse_row_to_dict
from asm.predict import *
import urllib.request
from time import time
from asm.utils import map_docker2host, Board2Path
from datetime import datetime
from asm.sam_utils import sam_find_board_V2
import sys
sys.path.append("..")
from asm.net.segment_anything import sam_model_registry, SamAutomaticMaskGenerator

bp = Blueprint('sam_predict', __name__)

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

@torch.no_grad()
@bp.route('/sam_predict', methods=['POST'])
def sam_predict():
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
    sam = sam_model_registry['vit_h'](checkpoint=ModelSet[modelName]['weight'])
    rows = query_d_hits(projectId=projectId)
    # ------------------------------------------------------------
    # file = open("/home/disk1/xjb/code/python/project/aix/imga.txt", "r")
    # file_contents = file.readlines()
    # file.close()
    # rows = [line.strip() for line in file_contents]
    # print(rows[0])
    # ----------------------------------------------------------------------
    t1 = time()
    TotalIters = rows.rowcount
    rows_bar = tqdm(rows)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for _, row in enumerate(rows_bar):
        hit_dict = parse_row_to_dict(row)
        img_path = map_docker2host(hit_dict['data'])
        # -----------------------------
        # img_path = row
        # -----------------------------------------
        if 'http' in img_path:
            file = io.BytesIO(urllib.request.urlopen(img_path).read())
            img = cv2.imread(file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.imread('/home/xjb/code/SAM/ww.jpg')
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sam.to(device)
        img_w, img_h = img.shape[0:2]
        print(img.shape)
        mask_generator = SamAutomaticMaskGenerator(sam)
        masks = mask_generator.generate(img)
        print(masks[1])
        board = sam_find_board_V2(masks)
        print(type(board[0][0]))
        print(board[0][0])
        result = []
        ans = np.zeros((img_w, img_h, 3))
        # -----------------------------------
        # for i in range(len(masks)):
        #     ans[:, :, 0] += masks[i]['segmentation']*img[:, :, 0]
        #     ans[:, :, 1] += masks[i]['segmentation']*img[:, :, 1]
        #     ans[:, :, 2] += masks[i]['segmentation']*img[:, :, 2]
        # cv2.imwrite('aa.png', ans)
        # ------------------------------------
        if(len(board) > 0):
            for i in range(len(board)):
                box_info = {
                    "type":"asm_path",
                    "angle":0,
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
                #----------------------------------------

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
#
# if __name__ == '__main__':
#     sam_predict("SAM", 99999999, 132, 2)

