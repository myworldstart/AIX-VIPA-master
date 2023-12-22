import datetime
import logging as rel_log
import os
import shutil
from datetime import timedelta
from flask import Blueprint, jsonify, request, send_from_directory, make_response
from .yolo_predict import Detector
import cv2

UPLOAD_FOLDER = r'./uploads'

ALLOWED_EXTENSIONS = set(['png', 'jpg'])
# app = Flask(__name__)
bp = Blueprint('yolo', __name__)
# bp.secret_key = 'secret!'
# bp.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# werkzeug_logger = rel_log.getLogger('werkzeug')
# werkzeug_logger.setLevel(rel_log.ERROR)

# 解决缓存刷新问题
# bp.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)

def pre_process(data_path):
    file_name = os.path.split(data_path)[1].split('.')[0]
    return data_path, file_name


def predict(dataset, model, ext):
    global img_y
    x = dataset[0].replace('\\', '/')
    file_name = dataset[1]
    print(x)
    print(file_name)
    x = cv2.imread(x)
    img_y, image_info = model.detect(x)
    cv2.imwrite('./tmp/draw/{}.{}'.format(file_name, ext), img_y)
    return image_info

def yolo_main(path, model, ext):
    image_data = pre_process(path)
    image_info = predict(image_data, model, ext)

    return image_data[1] + '.' + ext, image_info

# 添加header解决跨域
@bp.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
    return response

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@bp.route('/yolo_upload', methods=['GET', 'POST'])
def upload_file():
    file = request.files['file']
    print(datetime.datetime.now(), file.filename)
    if file and allowed_file(file.filename):
        src_path = os.path.join(UPLOAD_FOLDER, file.filename)
        # src_path = os.path.join(bp.config['UPLOAD_FOLDER'], file.filename)
        file.save(src_path)
        shutil.copy(src_path, './tmp/ct')
        image_path = os.path.join('./tmp/ct', file.filename)
        detector = Detector()
        pid, image_info = yolo_main(
            image_path, detector, file.filename.rsplit('.', 1)[1])
        
        return jsonify({'status': 1,
                        'image_url': 'http://10.214.211.207:3030/tmp/ct/' + pid,
                        'draw_url': 'http://10.214.211.207:3030/tmp/draw/' + pid,
                        'image_info': image_info})

    return jsonify({'status': 0})

@bp.route("/yolo_download", methods=['GET'])
def download_file():
    # 需要知道2个参数, 第1个参数是本地目录的path, 第2个参数是文件名(带扩展名)
    return send_from_directory('data', 'testfile.zip', as_attachment=True)

# show photo
@bp.route('/tmp/<path:file>', methods=['GET'])
def show_photo(file):
    if request.method == 'GET':
        if not file is None:
            image_data = open(f'tmp/{file}', "rb").read()
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response