import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import copy
import numpy as np
import time
import requests
import threading
from PIL import Image
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
import tools.infer.predict_angle as predict_angle
import tools.infer.predict_layout as predict_layout
from tools.utils.logging import get_logger
from tools.infer.utility import draw_ocr_box_txt

logger = get_logger(name='predict', log_file='./logs/ocr.log')

args = utility.parse_args()

def contrast_brightness_image(src1, a, g):
    h, w, ch = src1.shape  # 获取shape的数值，height和width、通道

    # 新建全零图片数组src2,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
    src2 = np.zeros([h, w, ch], src1.dtype)
    dst = cv2.addWeighted(src1, a, src2, 1 - a, g)  # addWeighted函数说明如下
    return dst

class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        start_time = time.time()
        self.result = self.func(self.args)
        logger.info("版面分析请求耗时: %.3fs" % (time.time() - start_time))

    def get_result(self):
        try:
            return self.result  # 如果子线程不使用join方法，此处可能会报没有self.result的错误
        except Exception:
            return None


class TextSystem(object):
    def __init__(self, args):
        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)
        self.use_pic_angle_cls = args.use_pic_angle_cls
        if self.use_pic_angle_cls:
            self.angle_classifier = predict_angle.AngleClassifier(args)
        self.loyout_detector = predict_layout.YoloxInference(args)

    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def print_draw_crop_rec_res(self, img_crop_list, rec_res):
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite("./output/img_crop_%d.jpg" % bno, img_crop_list[bno])
            logger.info(bno, rec_res[bno])

    def __call__(self, img):
        img_angle = img.copy()
        angle_label_index = 0
        if args.use_pic_angle_cls:
            _, cls_angle, elapse = self.angle_classifier([img])

            logger.info("angle : {}, elapse : {}".format(
                len(cls_angle[0][0]), elapse))
            angle_label_index = args.angel_label_list.index(cls_angle[0][0])
            img_angle = img.copy()
            if angle_label_index != 0:
                img_angle = np.rot90(img_angle, -angle_label_index).copy()

        # 版面分析
        th = MyThread(self.loyout_detector, args=(img_angle))
        th.start()

        adjust = contrast_brightness_image(img_angle, 1.5, 10)
        dt_boxes, elapse = self.text_detector(adjust)
        logger.info("dt_boxes num : {}, elapse : {}".format(
            len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(img_angle, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            logger.info("cls num  : {}, elapse : {}".format(
                len(img_crop_list), elapse))
        rec_res, elapse = self.text_recognizer(img_crop_list)
        logger.info("rec_res num  : {}, elapse : {}".format(
            len(rec_res), elapse))
        # self.print_draw_crop_rec_res(img_crop_list, rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            text, score = rec_reuslt
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)

        th.join()
        analysis = th.get_result()

        return filter_boxes, filter_rec_res, analysis, img_angle, angle_label_index


text_sys = TextSystem(args)


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


def iou(box1, box2):
    '''
    两个框（二维）的 iou 计算
    注意：边框以左上为原点
    box:[top, left, bottom, right]
    '''
    in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
    in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])
    inter = 0 if in_h < 0 or in_w < 0 else in_h * in_w
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
            (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    iou = inter / union
    return iou


def ios(box1, box2):
    '''
    两个框（二维）的 ios 计算，交集与box1的比值:
    注意：边框以左上为原点
    box:[top, left, bottom, right]
    '''
    in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
    in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])
    inter = 0 if in_h < 0 or in_w < 0 else in_h * in_w
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    ios_rate = inter / box1_area
    return ios_rate


def doc_analyse_maskrcnn(img):
    # 版面分析
    start_time = time.time()
    url = "http://192.168.0.166:8888/server_route_seg_"
    payload = {'file': ''}
    response = requests.request("POST", url, headers={}, data=payload, files=[('file', img)])
    logger.info("版面分析请求耗时: %.3fs" % (time.time() - start_time))
    if response.json()['msg'] == 'success':
        return eval(response.json()["data"])
    else:
        return []


def predict_mul(img_bin, log_id, is_visualize=True):
    img_name = log_id + "_n.png"
    img = cv2.imdecode(np.fromstring(img_bin, dtype=np.uint8), flags=1)

    starttime = time.time()
    dt_boxes, rec_res, analysis, img_angle, angle = text_sys(img)
    img = img_angle.copy()
    elapse = time.time() - starttime
    logger.info("Predict time of %s: %.3fs" % (img_name, elapse))

    txts = [rec_res[i][0] for i in range(len(rec_res))]
    scores = [rec_res[i][1] for i in range(len(rec_res))]

    type_list = ["body", "head", "formula", "form", "pic"]
    layouts = [{"layout": type_list[int(analysis[2][index])],
                "layout_location": [{"x": int(box[0]), "y": int(box[1])},
                                    {"x": int(box[0]), "y": int(box[3])},
                                    {"x": int(box[2]), "y": int(box[3])},
                                    {"x": int(box[2]), "y": int(box[1])}],
                "layout_idx": []}
               for index, box in enumerate(analysis[0])]

    ''' 
    # type_list = ["标题", "正文", "图片", "表格"]
    # box [1.0, 35.46666717529297, 60.287498474121094, 57.63333511352539, 163.6374969482422, 720.0]
    # [layout, ymin, xmin, ymax, xmax, area]
    # layout_location: leftup, rightup, rightdown, leftdown
    # ogger.info('analysis', analysis)
    # logger.info('analysis', analysis[0])
    type_list = ['title', 'text', 'figure', 'table']
    layouts = [{"layout": type_list[int(box[0]) - 1],
                "layout_location": [{"x": int(box[2]), "y": int(box[1])},
                                    {"x": int(box[4]), "y": int(box[1])},
                                    {"x": int(box[4]), "y": int(box[3])},
                                    {"x": int(box[2]), "y": int(box[3])}],
                "layout_idx": []}
               for box in analysis]'''

    same_thresh = 0.5
    r = []
    for i in range(len(txts)):
        box = dt_boxes[i]
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
            if ios_rate > same_thresh:
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
                (lout['layout'] in ['body', 'head'] and len(lout['layout_idx']) == 0) or (
                    lout['layout'] == 'form' and len(lout['layout_idx']) < 3))]

    logger.info("总耗时: %.3fs" % (time.time() - starttime))

    # result = {
    #     "results_num": len(dt_boxes),
    #     "log_id": int(log_id),
    #     "img_direction": 0,
    #     "layouts_num": len(analysis),
    #     "results": r,
    #     "layouts": layouts
    # }
    result = {
        "results_num": len(dt_boxes),
        "log_id": int(log_id),
        "img_direction": angle,
        "layouts_num": len(layouts_new),
        "results": r,
        "layouts": layouts_new
    }
    # if is_visualize:
    #     font_path = args.vis_font_path
    #     drop_score = args.drop_score
    #     image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #     boxes = dt_boxes
    #
    #     draw_img = draw_ocr_box_txt(
    #         image,
    #         boxes,
    #         txts,
    #         scores,
    #         drop_score=drop_score,
    #         font_path=font_path)
    #     draw_img_save = "./inference_results/"
    #     if not os.path.exists(draw_img_save):
    #         os.makedirs(draw_img_save)
    #     cv2.imwrite(
    #         os.path.join(draw_img_save, os.path.basename(img_name)),
    #         draw_img[:, :, ::-1])
    #     logger.info("The visualized image saved in {}".format(
    #         os.path.join(draw_img_save, os.path.basename(img_name))))

    if is_visualize:
        tv = time.time()
        type_list = ["body", "head", "formula", "form", "pic"]
        colcor = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
        for b, lout in enumerate(layouts_new):
            cv2.rectangle(img, (int(lout['layout_location'][0]['x']), int(lout['layout_location'][0]['y'])),
                          (int(lout['layout_location'][2]['x']), int(lout['layout_location'][2]['y'])),
                          colcor[int(type_list.index(lout['layout']))], 4)

        font_path = args.vis_font_path
        drop_score = args.drop_score
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        boxes = dt_boxes

        draw_img = draw_ocr_box_txt(
            image,
            boxes,
            txts,
            scores,
            drop_score=drop_score,
            font_path=font_path)
        draw_img_save = "./inference_results/"
        if not os.path.exists(draw_img_save):
            os.makedirs(draw_img_save)
        cv2.imwrite(
            os.path.join(draw_img_save, os.path.basename('1_' + img_name)),
            draw_img[:, :, ::-1])
        logger.info("The visualized image saved in {}".format(
            os.path.join(draw_img_save, os.path.basename(img_name))))

        logger.info("visualize time: %.3fs" % (time.time() - tv))
    logger.info("\n")
    return result
