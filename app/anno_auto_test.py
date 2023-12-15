import io
import json
import time
import hashlib
from flask import Blueprint, jsonify
from flask.globals import request
from sqlalchemy import text

from app.db import connect_redis, query_d_hits, parse_row_to_dict
from asm.predict import *
import urllib.request
from time import time
from app import sam_predict
from asm.utils import map_docker2host, Board2Path
from datetime import datetime
from app.db import db, connect_redis, query_d_hits, parse_row_to_dict

bp = Blueprint('anno_auto_test', __name__)

@torch.no_grad()
@bp.route('/auto_label_test', methods=['POST'])
def auto_label_test():
    try:
        print(request.json)
        modelName, datasetName, projectId, uid = request.json.get('modelName'), request.json.get('datasetName'), request.json.get('projectId'), request.json.get('uid')
    except Exception as e:
        return jsonify( {'code': 500, 'info': 'Parameter wrong, check your parameters is vaild all ?'} )

    if modelName not in ModelSet.keys() or not projectId:
        return jsonify( {'code': 500, 'info': 'Model not exist and need projectId !'} )

    # if(modelName == 'SAM'):
    #     return sam_predict(modelName, datasetName, projectId, uid)

    labels = ModelSet[modelName]['args']['labels']

    redisClient = connect_redis()
    if redisClient.exists(f'{datasetName}_{projectId}_{uid}'):
        return jsonify( {'code': 500, 'info': 'Task running already!'} )

    redisClient.set(f'{datasetName}_{projectId}_{uid}', 'True')

    TaskId = hashlib.md5( ( str(modelName) + str(projectId) + str(uid) ).encode('utf-8') ).hexdigest()
    model = load_model(modelName=modelName)
    rows = query_d_hits(projectId=projectId)

    # try:
    t1 = time()
    TotalIters = rows.rowcount
    rows_bar = tqdm(rows)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for _, row in enumerate(rows_bar):
        hit_dict = parse_row_to_dict(row)
        img_path = map_docker2host(hit_dict['data'])
        if 'http' in img_path:
            file = io.BytesIO(urllib.request.urlopen(img_path).read())
            img = Image.open(file).convert('RGB')
        else:
            img = Image.open(img_path).convert('RGB')
        img_w, img_h = img.size
        
        boards, cats = predict(model, img, device)
        
        if len(boards.keys()) > 0:
            result = []
            for _, cat in enumerate(cats):
                if cat == 0:
                    continue
                for index in range(len(boards[cat])):
                    box_info = {
                        "label": [labels[cat - 1]],
                        "type": "asm_path",
                        "angle": 0,
                        "path": Board2Path(boards[cat][index], img_w, img_h),
                        "imageWidth": img_w,
                        "imageHeight": img_h
                    }
                    result.append(box_info)

            sql = "insert into d_hits_result " \
                    "(`hitId`, `projectId`, `result`, `userid`, `timeTakenToLabelInSec`, `notes`, `created_timestamp`, `updated_timestamp`, `predLabel`, `model`, `status`) " \
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
        redisClient.set( name=f'{uid}_{TaskId}', value=json.dumps({'TaskId':TaskId, 'DatasetName':datasetName, 'TimeLeft':TimeLeft, 'Progress': CurrentIters / TotalIters * 100 }) )

    t2 = time()
    redisClient.delete(f'{datasetName}_{projectId}_{uid}')
    print('Cost time', t2-t1)
    return jsonify( {'code': 200, 'info': 'success'} )

    # except Exception as e:
    #     print('Exception:', e)
    #     ws.send(f'Predict failed, {e}')


