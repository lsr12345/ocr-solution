'''
# Author: Shaoran Lu
# Date: 2021/08/11
# Email: lushaoran92@gmail.com
# Description: OCR文本检测的推理类，infernence传入图片路径，返回值 boxes

example:
    db_inference = DBInference(onnx_model)
    boxes = db_inference.inference(image_path)
'''

import os
import cv2
import time
import numpy as np
from shapely.geometry import Polygon
import pyclipper
import onnxruntime

class DBPostProcess(object):
    """
    The post process for Differentiable Binarization (DB).
    """

    def __init__(self,
                 thresh=0.3,
                 box_thresh=0.7,
                 max_candidates=1000,
                 unclip_ratio=2.0,
                 use_dilation=False,
                 score_mode="fast",
                 **kwargs):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.score_mode = score_mode
        assert score_mode in [
            "slow", "fast"
        ], "Score mode must be in [slow, fast] but got: {}".format(score_mode)

        self.dilation_kernel = None if not use_dilation else np.array(
            [[1, 1], [1, 1]])

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        '''

        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            if self.score_mode == "fast":
                score = self.box_score_fast(pred, points.reshape(-1, 2))
            else:
                score = self.box_score_slow(pred, contour)
            if self.box_thresh > score:
                continue

            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(np.int16))
            scores.append(score)
        return np.array(boxes, dtype=np.int16), scores

    def unclip(self, box):
        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        '''
        box_score_fast: use bbox mean score as the mean score
        '''
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def box_score_slow(self, bitmap, contour):
        '''
        box_score_slow: use polyon mean score as the mean score
        '''
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))

        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self, pred, shape_list):
#         pred = outs_dict['maps']
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh

        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            if self.dilation_kernel is not None:
                mask = cv2.dilate(
                    np.array(segmentation[batch_index]).astype(np.uint8),
                    self.dilation_kernel)
            else:
                mask = segmentation[batch_index]
            boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask,
                                                   src_w, src_h)

            boxes_batch.append({'points': boxes})
        return boxes_batch
    
    
class DBInference():

    def __init__(self, model_path, input_size=(960, 960), thresh=0.2, box_thresh=0.5, max_candidates=800, unclip_ratio=2.0):
        self.model_path = model_path
        self.input_size = input_size
        self.postprocess = DBPostProcess(thresh=thresh, box_thresh=box_thresh, max_candidates=max_candidates, unclip_ratio=unclip_ratio)
        
        self.session_db = onnxruntime.InferenceSession(self.model_path)

        inputs = {self.session_db.get_inputs()[0].name: np.ones((1,3,input_size[0],input_size[1])).astype(np.float32)}
        output = self.session_db.run(None, inputs)

    def resize_image(self, img, resize_shape):
        resize_h, resize_w = resize_shape
        ori_h, ori_w = img.shape[:2]  # (h, w, c)
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        # return img, np.array([ori_h, ori_w])
        return img, [ratio_h, ratio_w]

    def DBPreprocess(self, img, shape=(640,640)):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        shape_ = (1, 1, 3)
        scale = 1./255.
        mean = np.array(mean).reshape(shape_).astype('float32')
        std = np.array(std).reshape(shape_).astype('float32')

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        mask_img, [ratio_h, ratio_w] = self.resize_image(img, shape) 
        mask_img = (mask_img.astype('float32') * scale - mean) / std
        mask_img = mask_img.transpose((2, 0, 1))
        return mask_img, ratio_h, ratio_w
    
    def draw_det_res(self, dt_boxes, visual_save, img, img_name):
        if len(dt_boxes) > 0:
            src_im = img
            for box in dt_boxes:
                box = box.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
            if not os.path.exists(visual_save):
                os.makedirs(visual_save)
            save_path = os.path.join(visual_save, img_name)
            cv2.imwrite(save_path, src_im)

    def inference(self, image, visual_save=None, visual_name='visual_img_db.jpg'):
        # image = cv2.imread(img_path)
        img_pre,ratio_h,ratio_w = self.DBPreprocess(image, shape=self.input_size)
        img_pre = np.expand_dims(img_pre, axis=0)
        ort_inputs = {self.session_db.get_inputs()[0].name: img_pre}
        output = self.session_db.run(None, ort_inputs)
        
        output_numpy = np.array(output)
        
        shape_list = [[image.shape[0], image.shape[1], ratio_h, ratio_w]]
        res = self.postprocess(output_numpy[0], shape_list)
        
        dt_boxes = res[0]['points']

        if len(dt_boxes) < 1:
            print('Dets ob nums == 0')
            return None

        if visual_save is not None:
            if not os.path.exists(visual_save):
                os.makedirs(visual_save)
            self.draw_det_res(dt_boxes, visual_save, image, visual_name)

        return dt_boxes

if __name__=='__main__':
    onnx_model = '/home/shaoran/company_work/starsee/ocr_solution/models/det_inference_free_dim.onnx'
    # demo_image = './ocr_demo/layout/010005.jpg'
    # visual_dir = './visual_results'

    input_size = (1920, 1920)

    db_inference = DBInference(onnx_model, input_size, thresh=0.1, box_thresh=0.1)
    # print('onnx model initial done')
    # for _ in range(20):
    #     _ = db_inference.inference(demo_image, visual_save=None)
        
    # start = time.time()
    # for _ in range(100):
    #     res = db_inference.inference(demo_image, visual_save=None)
    # stop = time.time()
    # print('per image spend time: ', (stop-start)/100)
    # fps = 100/(stop-start)
    # print('fps: ', fps)

    root_dir = '/home/shaoran/company_work/starsee/ocr_solution/onnx_inference/海研究院'
    visual_path = '/home/shaoran/company_work/starsee/ocr_solution/visual_results'
    start = time.time()
    for path in os.listdir(root_dir):
        image = cv2.imread(os.path.join(root_dir, path))
        _ = db_inference.inference(image, visual_path, path)
    
    # print(res)