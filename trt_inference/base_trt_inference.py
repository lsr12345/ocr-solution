import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from ocr_det.db_trt_class import DBInference_trt
from ocr_rec.crnn_trt_class import CrnnInference_trt
from ocr_angle_cls.angle_trt_class import  AngleClsInference_trt
from layout.yolox_trt_class import YoloxInference_trt
from trt_inference.inference_tools import *

import cv2
import copy
import numpy as np
import time
import datetime
import yaml
# import requests

# from PIL import Image

from loguru import logger

if not os.path.exists(os.path.join(__dir__, 'logs')):
    os.mkdir(os.path.join(__dir__, 'logs'))

logger.add(os.path.join(__dir__, 'logs', f'onnx_ocr_{datetime.date.today()}.log'))

config_yaml = os.path.join(__dir__, '../', 'config/Config_trt.yaml')
# config_yaml = os.path.join(__dir__, '../', 'config/Config_trt_docker.yaml')

with open(config_yaml, mode='r') as fr:
    cfg = yaml.load(fr)

print('* configs: ', cfg)

class TextSystem(object):
    def __init__(self, config):

        self.config = config
        self.drop_score = config['inference']['drop_score']

        if config['inference']['type'] == 'tensorrt':

            self.text_detector = DBInference_trt(config['model_path']['det'], eval(config['inference']['input_size']))
            self.text_recognizer = CrnnInference_trt(config['model_path']['rec'], config['file_path']['crnn_chars'])
            if config['inference']['use_layout']:
                self.loyout_detector = YoloxInference_trt(config['model_path']['layout'], eval(config['inference']['input_size']))
            if config['inference']['use_formula']:
                raise NotImplementedError('Formula inference not supported.')
            if config['inference']['use_table_struct']:
                raise NotImplementedError('Table struct inference not supported.')
            if config['inference']['use_angle_cls']:
                self.text_angle_classifier = AngleClsInference_trt(config['model_path']['angle_cls'], eval(config['angle_cls']['input_size']))
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
            logger.info('角度检测耗时: {}'.format(stop-start))
            logger.info("angle : {}".format(ac_res[0]))

        # 版面分析
        th = MyThread(self.loyout_detector.inference, args=(img_angle))
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

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            tmp_box = tmp_box.astype(np.float32)
            img_crop = get_rotate_crop_image(img_angle, tmp_box)
            img_crop_list.append(img_crop)

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
                try:
                    if len(small_batch) == 1:
                        zeros_pad = np.zeros_like(small_batch[0])
                        small_batch.append(zeros_pad)
                        rec_small = self.text_recognizer.inference_batch(small_batch)
                        rec_res.append(rec_small[0])
                        small_batch = []
                    else:
                        rec_small = self.text_recognizer.inference_batch(small_batch)
                        small_batch = []
                        for rec_ in rec_small:
                            rec_res.append(rec_)                        
                except:
                    small_batch = []
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

        return filter_boxes, filter_rec_res, analysis, img_angle   # , angle_label_index

if __name__=='__main__':

    text_sys = TextSystem(cfg)

    image = cv2.imread('/home/shaoran/company_work/starsee/ocr_solution/onnx_inference/demo/2.jpg')
    # image = cv2.imread('/data/company_work/starsee/ocr_solution/onnx_inference/demo/2.jpg')
    
    for _ in range(10):
        _, _, _, _ = text_sys.run(image)

    start = time.time()
    for _ in range(50):
        filter_boxes, filter_rec_res, analysis, img_angle = text_sys.run(image)
    stop = time.time()
    print('per image spend time: ', (stop-start)/50)
    fps = 50/(stop-start)
    print('fps: ', fps)
    print(filter_rec_res)
