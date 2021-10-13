import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

# from ocr_det.db_onnx_class import DBInference
from ocr_det.db_a_onnx_class import DBInference
from ocr_rec.crnn_onnx_class import CrnnInference
from ocr_angle_cls.angle_onnx_class import  AngleClsInference
from layout.yolox_onnx_class import YoloxInference
from onnx_inference.inference_tools import *

import cv2
import copy
import numpy as np
import time
import datetime
import yaml
import requests

from PIL import Image

# import tools.infer.predict_cls as predict_cls
# import tools.infer.predict_angle as predict_angle
from loguru import logger

if not os.path.exists(os.path.join(__dir__, 'logs')):
    os.mkdir(os.path.join(__dir__, 'logs'))

logger.add(os.path.join(__dir__, 'logs', f'onnx_ocr_{datetime.date.today()}.log'))
config_yaml = os.path.join(__dir__, '../', 'config/Config.yaml')

with open(config_yaml, mode='r') as fr:
    cfg = yaml.load(fr)

print('* configs: ', cfg)

class TextSystem(object):
    def __init__(self, config):

        self.config = config
        self.drop_score = config['inference']['drop_score']

        if config['inference']['type'] == 'onnx':

            self.text_detector = DBInference(config['model_path']['det'], eval(config['inference']['input_size']))
            self.text_recognizer = CrnnInference(config['model_path']['rec'], config['file_path']['crnn_chars'])
            if config['inference']['use_layout']:
                self.loyout_detector = YoloxInference(config['model_path']['layout'], eval(config['inference']['input_size']), score_thr=0.75)
            if config['inference']['use_formula']:
                raise NotImplementedError('Formula inference not supported.')
            if config['inference']['use_table_struct']:
                raise NotImplementedError('Table struct inference not supported.')
            if config['inference']['use_angle_cls']:
                self.text_angle_classifier = AngleClsInference(config['model_path']['angle_cls'], eval(config['angle_cls']['input_size']))
            if config['inference']['use_pic_angle_cls']:
                # self.angle_classifier = predict_angle.AngleClassifier(args)
                raise NotImplementedError('Pic_angle_cls inference not supported.')

        else:
            raise NotImplementedError('Type {} not supported.'.format(config['inference']['type']))

        if self.config['visual_save']['visual_flag']:
            if not os.path.exists(self.config['visual_save']['visual_path']):
                os.makedirs(self.config['visual_save']['visual_path'])

            self.visual_save_path = self.config['visual_save']['visual_path']
        else:
            self.visual_save_path = None


    def print_draw_crop_rec_res(self, img_crop_list, rec_res, timestamp):
        bbox_num = len(img_crop_list)
        # timestamp = str(time.time()).replace('.', '')
        if not os.path.exists(os.path.join(self.config['visual_save']['visual_path'], timestamp, 'crop')):
            os.makedirs(os.path.join(self.config['visual_save']['visual_path'], timestamp, 'crop'))

        crop_save_path = os.path.join(self.config['visual_save']['visual_path'], timestamp, 'crop')

        for bno in range(bbox_num):
            cv2.imwrite(os.path.join(crop_save_path, "img_crop_%d.jpg" % bno), img_crop_list[bno])
            logger.info(bno, rec_res[bno])

    def run(self, img, visual_name='visual_img.jpg'):
        img_angle = img.copy()
        if self.config['inference']['use_angle_cls']:
            start = time.time()
            ac_res = self.text_angle_classifier.inference(img_angle)
            rot_n = self.config['angle_cls']['angel_label_list'].index(ac_res[0])
            img_angle = np.rot90(img_angle, -rot_n).copy()
            stop = time.time()
            logger.info('大角度检测耗时: {}'.format(stop-start))
            logger.info("angle : {}".format(ac_res[0]))

            start = time.time()
            img_angle, tiny_angle_0 = self.text_detector.inference_angle(img_angle, None, visual_name)
            img_angle, tiny_angle_1 = self.text_detector.inference_angle(img_angle, self.visual_save_path, visual_name,  enhance=True)
            angle = tiny_angle_0+tiny_angle_1
            stop = time.time()
            logger.info('小角度角度检测耗时: {}'.format(stop-start))
            logger.info("Tiny angle : {}".format(angle))
        else:
            angle = 0

        # 版面分析
        th = MyThread(self.loyout_detector.inference, args=([img_angle, self.config['visual_save']['visual_path'], 'layout_'+visual_name]))
        th.start()

        # adjust = contrast_brightness_image(img_angle, 1.5, 10)
        start = time.time()
        dt_boxes = self.text_detector.inference(img_angle, self.visual_save_path, visual_name)
        stop = time.time()

        logger.info('文字检测耗时: {}'.format(stop-start))
        logger.info("dt_boxes num : {}".format(len(dt_boxes)))
        if dt_boxes is None:
            return None, None

        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)
        width_list = []
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            tmp_box = tmp_box.astype(np.float32)
            img_crop = get_rotate_crop_image(img_angle, tmp_box)
            img_crop_list.append(img_crop)
            width_list.append(img_crop.shape[1] / float(img_crop.shape[0]))
            
        indices = np.argsort(np.array(width_list))
        # print(indices)
        img_crop_list = [img_crop_list[ind] for ind in indices]

        # # crnn total batch inference
        # start = time.time()
        # rec_res = self.text_recognizer.inference_batch(img_crop_list)
        # stop = time.time()
        # logger.info('text rec time-total batch: {}'.format(stop-start))
        #
        # # crnn single inference
        # # rec_res = []
        # start = time.time()
        # for img_ in img_crop_list:
        #     rec_re = self.text_recognizer.inference(img_)
        #     # rec_res.append(rec_re)
        # stop = time.time()
        # logger.info('text rec time-single: {}'.format(stop-start))

        ## crnn small_batch inference
        rec_res = []
        small_batch = []
        start = time.time()
        for i, img_ in enumerate(img_crop_list):
            small_batch.append(img_)
            if len(small_batch) == int(self.config['crnn']['inference_batch']) or i == len(img_crop_list)-1:
                # try:
                rec_small = self.text_recognizer.inference_batch(small_batch)
                small_batch = []
                for rec_ in rec_small:
                    rec_res.append(rec_)
                # except:
                #     small_batch = []

        rec_res_ = [0] * len(indices)
        for i,j in enumerate(indices):
            rec_res_[j] = rec_res[i]
            
        rec_res = rec_res_   

        stop = time.time()
        logger.info('文字识别耗时: {}'.format(stop-start))

        logger.info("rec_res num  : {}".format(len(rec_res)))
        # self.print_draw_crop_rec_res(img_crop_list, rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            text, score = rec_reuslt
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)

        th.join()
        analysis = th.get_result()
        # print('....', analysis)

        analysis_re = []
        for i in range(analysis[0].shape[0]):
            tmp_ar = []
            tmp_ar.append(int(analysis[2][i]))
            tmp_ar.append(analysis[1][i])
            tmp_ar.append(analysis[0][i].tolist())
            analysis_re.append(tmp_ar)

        return filter_boxes, filter_rec_res, analysis_re, img_angle, angle   # , angle_label_index

    def warm(self):
        logger.info("Warming...... Please Wait")
        for _ in range(5):
            for i in range(32):
                warm_rec_data = np.zeros((32, (i+1)*32, 3), dtype=np.uint8)
                _ = self.text_recognizer.inference_batch([warm_rec_data]*2)

            warm_det_data = np.zeros((960,960, 3), dtype=np.uint8)
            _ = self.text_detector.inference(warm_det_data, visual_save=None)

            warm_layout_data = np.zeros((960,960, 3), dtype=np.uint8)
            _ = self.loyout_detector.inference(warm_layout_data, visual_save=None)
        logger.info("Warm Done")

def postprocess( boxes, rec_res, analysis, img, angle, type_list = ['text', 'title', 'formula', 'table', 'figure']):
    start_time = time.time()
    img = img.copy()
    txts = [rec_res[i][0] for i in range(len(rec_res))]
    scores = [rec_res[i][1] for i in range(len(rec_res))]
    layouts = []
    for box in analysis:
        layouts.append({"layout": type_list[int(box[0])],
                    "layout_location": [{"x": int(box[2][0]), "y": int(box[2][1])},
                                        {"x": int(box[2][2]), "y": int(box[2][1])},
                                        {"x": int(box[2][2]), "y": int(box[2][3])},
                                        {"x": int(box[2][0]), "y": int(box[2][3])}],
                    "layout_idx": []})

    i_thresh = 0.5
    r = []
    for i in range(len(txts)):
        box = boxes[i]
        txt = txts[i]
        scor = scores[i]

        box_type = "unknown"
        box_index = -1

        for l, layout in enumerate(layouts):
            # box:[top, left, bottom, right]
            anbox = [layout["layout_location"][0]["y"],
                     layout["layout_location"][0]["x"],
                     layout["layout_location"][2]["y"],
                     layout["layout_location"][2]["x"]]
            dtbox = [box[0][1], box[0][0], box[3][1], box[1][0]]
            ios_rate = ios(dtbox, anbox)
            if ios_rate > i_thresh:
                layout["layout_idx"].append(i)
                box_index = l
                box_type = layout["layout"]
                break

        single_data = dict()
        single_data["words_type"] = box_type
        single_data["block_index"] = box_index
        single_data["score"] = float(scor)
        single_data["words"] = {
            "words_location": {
                "top": int(box[0][1]),
                "left": int(box[0][0]),
                "width": int(box[1][0] - box[0][0]),
                "height": int(box[3][1] - box[0][1]),
            },
            "word": txt,
        }
        r.append(single_data)

    layouts_new = [lout for lout in layouts if not (
                (lout['layout'] in ['text', 'title'] and len(lout['layout_idx']) == 0) or (
                    lout['layout'] == 'table' and len(lout['layout_idx']) < 3))]

    logger.info("后处理耗时: %.3fs" % (time.time() - start_time))

    result = {
        "results_num": len(boxes),
        "img_direction": angle,
        "layouts_num": len(layouts_new),
        "results": r,
        "layouts": layouts_new
    }

    return result

if __name__=='__main__':

    text_sys = TextSystem(cfg)
    text_sys.warm()

    root_dir = '/home/shaoran/company_work/starsee/ocr_solution/onnx_inference/ocr_zp'
    start = time.time()
    for path in os.listdir(root_dir):
        image = cv2.imread(os.path.join(root_dir, path))
        filter_boxes, filter_rec_res, analysis, img_angle, angle = text_sys.run(image, visual_name=path)
        result = postprocess( filter_boxes, filter_rec_res, analysis, img_angle, angle, type_list = ['text', 'title', 'formula', 'table', 'figure'])
    stop = time.time()
    print('per image spend time: ', (stop-start)/len(os.listdir(root_dir)))
    fps = len(os.listdir(root_dir))/(stop-start)
    print('fps: ', fps)
    # print(filter_rec_res)


