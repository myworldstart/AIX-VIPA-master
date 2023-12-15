import io
import json
import urllib.request
from copy import deepcopy
from app.db import connect_redis
from flask import Blueprint, jsonify
from flask.globals import request
from config import ModelSet
from time import time

bp = Blueprint('data_info', __name__)

@bp.route('/getModelList', methods=['GET'])
def data_info():
    ## 请求该接口获取与训练模型列表以及预训练模型对应的训练参数
    models = deepcopy(ModelSet)
    for model in models:
        models[model]['model'] = None
    return jsonify(models)


@bp.route('/getTaskStatus', methods=['GET'])
def getTaskStatus():
    uid = request.args.get('uid')
    redisClient = connect_redis()
    items = redisClient.keys(f'{uid}_*')
    userTaskList = [json.loads( redisClient.get(item) ) for item in items]
    return jsonify(userTaskList)