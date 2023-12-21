import torch
from flask import Flask, render_template
from flask_cors import CORS
from asm.utils import seek_gpu


app = Flask(__name__)

app.config.from_object('config')
CORS(app, supports_credentials=True)

# load index page here!
@app.route('/')
def index():
    # do this when start the web
    return render_template('index.html')


# todo: can't move to top of this file?
from . import anno_auto_test, anno_auto_train, data_info, eiseg_click, anno_auto_tradition, sam_click, sam_predict, yolo_detect

# register blueprint here
app.register_blueprint(anno_auto_test.bp)
app.register_blueprint(anno_auto_train.bp)
app.register_blueprint(data_info.bp)
app.register_blueprint(eiseg_click.bp)
app.register_blueprint(anno_auto_tradition.bp)
app.register_blueprint(sam_click.bp)
app.register_blueprint(sam_predict.bp)
app.register_blueprint(yolo_detect.bp)

# load models
seek_gpu()
