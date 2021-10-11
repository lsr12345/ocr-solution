'''
# Author: Shaoran Lu
# Date: 2021/08/11
# Email: lushaoran92@gmail.com
# Description: OCR文本角度分类推理基类，infernence传入图片，返回值类别得分和类别

example:
    angle_inference = AngleClsInference(onnx_model)
    res = db_inference.inference(image)
'''

import os
import cv2
import time
import math
import numpy as np
import onnxruntime


class AngleClsInference():

    def __init__(self, model_path, input_size=(720, 720), angle_thresh=0.9, angel_label_list=['0', '90', '180', '270']):
        self.model_path = model_path
        self.input_size = input_size
        self.angle_thresh = angle_thresh
        self.angel_label_list = angel_label_list
        
        self.session_angle = onnxruntime.InferenceSession(self.model_path)

        inputs = {self.session_angle.get_inputs()[0].name: np.ones((1,3,input_size[0],input_size[1])).astype(np.float32)}
        output = self.session_angle.run(None, inputs)

    def AnglePreprocess(self, img):
        imgC, imgH, imgW = 3, self.input_size[0], self.input_size[1]
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
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, 0:resized_h, 0:resized_w] = resized_image
        return padding_im

    def AnglePostprocess(self, pred):
        pred_idx = pred.argmax(axis=0)
        label = self.angel_label_list[pred_idx]
        score = pred[pred_idx]
        return label, score

    def inference(self, image):
        img_pre = self.AnglePreprocess(image)
        img_pre = np.expand_dims(img_pre, axis=0)
        ort_inputs = {self.session_angle.get_inputs()[0].name: img_pre}
        output = self.session_angle.run(None, ort_inputs)
        output_numpy = np.array(output[0][0])
        # print(output_numpy.shape)
        label, score = self.AnglePostprocess(output_numpy)
        return [label, score]

    def AnglePostprocess_batch(self, preds):
        preds_idx = preds.argmax(axis=1)
        res_batch = [[self.angel_label_list[pred_idx],preds[i][pred_idx].tolist()] for i, pred_idx in enumerate(preds_idx)]
        return res_batch

    def inference_batch(self, images, batch_size=2):
        img_pre_batch = []
        ress = []
        for i, image in enumerate(images):
            img_pre = self.AnglePreprocess(image)
            img_pre = np.expand_dims(img_pre, axis=0)
            img_pre_batch.append(img_pre)
            if len(img_pre_batch) == batch_size or i == len(images)-1:
                img_pre_batch = np.concatenate(img_pre_batch, axis=0)
                ort_inputs = {self.session_angle.get_inputs()[0].name: img_pre_batch}
                outputs = self.session_angle.run(None, ort_inputs)
                outputs_numpy = np.array(outputs[0])
                res_batch = self.AnglePostprocess_batch(outputs_numpy)
                ress += res_batch
                img_pre_batch = []
        return ress

if __name__=='__main__':
    onnx_model = './model_onnx/angle_cls.onnx'
    demo_image_path = './ocr_demo/layout/010005.jpg'
    demo_image = cv2.imread(demo_image_path)

    input_size = (720, 720)

    angle_inference = AngleClsInference(onnx_model, input_size)
    print('onnx model initial done')
    for _ in range(20):
        _ = angle_inference.inference(demo_image)
        
    start = time.time()
    for _ in range(100):
        res = angle_inference.inference(demo_image)
    stop = time.time()
    print('per image spend time: ', (stop-start)/100)
    fps = 100/(stop-start)
    print('fps: ', fps)
    
    res = angle_inference.inference(demo_image)
    print(res)
    print('Test batch inference..')
    demo_images_list = [demo_image for _ in range(9)]
    ress = angle_inference.inference_batch(demo_images_list)
    print(ress)
    