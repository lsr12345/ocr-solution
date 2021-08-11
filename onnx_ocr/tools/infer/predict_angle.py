import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import copy
import numpy as np
import math
import time
import traceback

import tools.infer.utility as utility
from tools.postprocess import build_post_process
from tools.utils.logging import get_logger
from tools.utils.utility import get_image_file_list, check_and_read_gif
from onnxruntime import InferenceSession

logger = get_logger()


class AngleClassifier(object):
    def __init__(self, args):
        self.angle_image_shape = [int(v) for v in args.angle_image_shape.split(",")]
        self.angle_batch_num = args.angle_batch_num
        self.angle_thresh = args.angle_thresh
        postprocess_params = {
            'name': 'ClsPostProcess',
            "label_list": args.angel_label_list,
        }
        self.postprocess_op = build_post_process(postprocess_params)
        # self.predictor, self.input_tensor, self.output_tensors = \
        #     utility.create_predictor(args, 'angle', logger)
        self.model = InferenceSession(args['angle_model_dir'])

    def resize_ori_ratio_img(self, img):
        imgC, imgH, imgW = self.angle_image_shape
        h = img.shape[0]
        w = img.shape[1]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
            resized_h = int(imgH / ratio)
        else:
            resized_w = int(math.ceil(imgH * ratio))
            resized_h = imgH
        resized_image = cv2.resize(img, (resized_w, resized_h))
        resized_image = resized_image.astype('float32')
        if self.angle_image_shape == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, 0:resized_h, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, img_list):
        img_list = copy.deepcopy(img_list)
        img_num = len(img_list)

        cls_res = [['', 0.0]] * img_num
        batch_num = self.angle_batch_num
        elapse = 0
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_ori_ratio_img(img_list[ino])
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()
            starttime = time.time()
            # self.input_tensor.copy_from_cpu(norm_img_batch)
            # self.predictor.run()
            output_tensors = self.model.run(output_names=None, input_feed={'x': norm_img_batch})
            # prob_out = self.output_tensors[0].copy_to_cpu()
            # self.predictor.try_shrink_memory()
            prob_out = output_tensors[0]
            cls_result = self.postprocess_op(prob_out)
            elapse += time.time() - starttime
            for rno in range(len(cls_result)):
                label, score = cls_result[rno]
                cls_res[beg_img_no + rno] = [label, score]
        return img_list, cls_res, elapse


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    text_classifier = AngleClassifier(args)
    valid_image_file_list = []
    img_list = []
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        valid_image_file_list.append(image_file)
        img_list.append(img)
    try:
        img_list, cls_res, predict_time = text_classifier(img_list)
        print(predict_time)
    except:
        logger.info(traceback.format_exc())
        exit()
    for ino in range(len(img_list)):
        logger.info("Predicts of {}:{}".format(valid_image_file_list[ino],
                                               cls_res[ino]))
    logger.info("Total predict time for {} images, cost: {:.3f}".format(
        len(img_list), predict_time))


if __name__ == "__main__":
    args = utility.parse_args()
    args.image_dir = '../../train_data/angle/test_temp'
    # args.angle_model_dir = '../../inference/angle/'
    args.angle_model_dir = '../../onnx_models/angle_dy.onnx'
    main(args)
