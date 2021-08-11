# -*- coding:utf-8 -*-
# Author: lianhai
# Data: 2021-08-04

import base64
import json
import time
import traceback
from flask import Flask, request
from tools.infer.predict_with_layout import predict_mul as predict_mul
from tools.utils.logging import get_logger
logger = get_logger(name='runserver', log_file='./logs/ocr.log')

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route('/ocr_hs', methods=['POST'])
def ocr_hs():
    try:
        req = json.loads(request.get_data(as_text=True))

        if "image" not in req:
            logger.error('缺少参数')
        content = str(req['image']).strip()
        if len(content) == 0:
            logger.error('参数为空')

        img_str = str(req['image']).strip()
        img_data = base64.b64decode(img_str)
        result = predict_mul(img_data, str(int(time.time() * 1000000000)))
        return result
    except:
        logger.error(traceback.format_exc())
        result = {
            "results_num": 0,
            "log_id": 0,
            "img_direction": 0,
            "layouts_num": 0,
            "results": [],
            "layouts": [],
            "msg": 'error'
        }
        return result

@app.route('/ocr_img', methods=['POST'])
def ocr_img():
    try:
        is_show = 'F'
        if 'is_show' in request.form:
            is_show = str(request.form['is_show']).strip()
        img = request.files['pic']
        img_bin = img.stream.read()
        result = predict_mul(img_bin, str(int(time.time() * 1000000000)),is_visualize=True if is_show=='T' else False)
        return result
    except:
        logger.error(traceback.format_exc())
        result = {
            "results_num": 0,
            "log_id": 0,
            "img_direction": 0,
            "layouts_num": 0,
            "results": [],
            "layouts": [],
            "msg": 'error'
        }
        return result

@app.route('/')
def hello_world():
    return 'hello world'

if __name__ == '__main__':
    # app.config['JSON_AS_ASCII'] = False
    app.run(host='0.0.0.0', port=12307, debug=False, threaded=False, use_reloader=False)
