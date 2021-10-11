'''
# Author: Shaoran Lu
# Date: 2021/08/11
# Email: lushaoran92@gmail.com
# Description: OCR文本识别推理类，infernence传入图片，返回值 文字识别List

example:
    crnn_inference = CrnnInference(onnx_model, char_path)
    res = crnn_inference.inference(image)
'''

import os
import cv2
import time
import numpy as np
import onnxruntime

class CrnnInference():

    def __init__(self, model_path, char_path):
        self.model_path = model_path
        self.char_path = char_path
        self.session_crnn = onnxruntime.InferenceSession(self.model_path)
        
        with open(char_path, 'r', encoding='UTF-8') as f:
            ff = f.readlines()
            char_list = []
            for i, char in enumerate(ff):
                char = char.strip()
                char_list.append(char)
        self.id2char = {i:j for i, j in enumerate(char_list)}

        inputs = {self.session_crnn.get_inputs()[0].name: np.ones((1, 3, 32, 100)).astype(np.float32)}
        output = self.session_crnn.run(None, inputs)

    def preprocess(self, img):
        h,w,c = img.shape

        scale = h / 32
        w_ = w / scale
        w_ = int(w_/32)*32

        img = cv2.resize(img, (w_, 32))
        img = img.astype('float32')
        img = img.transpose((2, 0, 1)) / 255
        img -= 0.5
        img /= 0.5
        return img

    def decode(self, index, id2char):
        res = []
        for i, idx in enumerate(index):
            if idx == 6624 or idx ==0:
                continue
            elif index[i-1] == index[i]:
                continue
            res.append(id2char[idx-1])
        return res

    def inference(self, image, beamserach=False):
        img = self.preprocess(image)
        img = np.expand_dims(img, axis=0)
        ort_inputs = {self.session_crnn.get_inputs()[0].name: img}
        output = np.array(self.session_crnn.run(None, ort_inputs)[0])[0]
        res_ids = np.argmax(output, axis=-1)
        pred_prob = np.max(output, axis=-1)
        if not beamserach:
            res = self.decode(res_ids, self.id2char)
            return [res, np.mean(pred_prob)]
        else:
            return None

    def batch_norm(self, image_lists, w):
        norm_preprocess_lists = []
        for img in image_lists:
            mask_img = np.full((3, 32, w), fill_value=1.0)
            mask_img[:, :img.shape[1], :img.shape[2]] = img
            mask_img = np.expand_dims(mask_img, axis=0)
            norm_preprocess_lists.append(mask_img.astype('float32'))
        return norm_preprocess_lists

    def inference_batch(self, image_lists, beamserach=False):
        max_w = 0
        preprocess_lists = []

        for img in image_lists:
            img = self.preprocess(img)
            if img.shape[2] > max_w:
                max_w = img.shape[2]
            preprocess_lists.append(img)

        norm_preprocess_lists = self.batch_norm(preprocess_lists, max_w)
        img_batch = np.concatenate(norm_preprocess_lists)
        ort_inputs = {self.session_crnn.get_inputs()[0].name: img_batch}
        outputs = np.array(self.session_crnn.run(None, ort_inputs))
        res_ids = np.argmax(outputs, axis=-1)
        preds_prob = np.max(outputs, axis=-1)
        res_lists = []
        if not beamserach:
            for res, prob in zip(res_ids[0], preds_prob[0]):
                res_lists.append([self.decode(res, self.id2char), np.mean(prob)])
            return res_lists
        else:
            return None


if __name__=='__main__':
    onnx_model = './model_onnx/rec_inference_free_dim.onnx'
    demo_image_path = './ocr_demo/rec/rec_0.png'
    char_path = './ocr_demo/ppocr_keys_v1.txt'

    crnn_inference = CrnnInference(onnx_model, char_path)
    print('onnx model initial done')
    # for path in os.listdir(demo_image_dir):
    #     demo_image = cv2.imread(os.path.join(demo_image_dir, path))
    #     res = crnn_inference.inference(demo_image)
    #     print(res)
    demo_image = cv2.imread(demo_image_path)
    
    for _ in range(20):
        _ = crnn_inference.inference(demo_image)
        
    start = time.time()
    for _ in range(100):
        res = crnn_inference.inference(demo_image)
    stop = time.time()
    print('per image spend time: ', (stop-start)/100)
    fps = 100/(stop-start)
    print('fps: ', fps)
    print(res)
    