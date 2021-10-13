'''
# Author: Shaoran Lu
# Date: 2021/08/11
# Email: lushaoran92@gmail.com
# Description: OCR文本识别的tensorrt推理类，infernence传入图片，返回值 文字识别List

example:
    crnn_inference = CrnnInference(onnx_model, char_path)
    res = crnn_inference.inference(image)
'''
# coding: utf-8

# In[1]:
import cv2
import time
import os
import math

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class HostDeviceMem(object):
    """Simple helper data class that's a little nicer to use than a 2-tuple."""
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return 'Host:\n' + str(self.host) + '\nDevice:\n' + str(self.device)

    def __repr__(self):
        return self.__str__()


class CrnnInference_trt():

    def __init__(self, model_path, char_path):
        # self.cfx = cuda.Device(0).make_context()
        self.model_path = model_path
        self.char_path = char_path
        with open(char_path, 'r', encoding='UTF-8') as f:
            ff = f.readlines()
            char_list = []
            for i, char in enumerate(ff):
                char = char.strip()
                char_list.append(char)
        self.id2char = {i:j for i, j in enumerate(char_list)}
        
        trt_logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(trt_logger, '')
        
        with open(model_path, 'rb') as f, trt.Runtime(trt_logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        assert self.engine.get_binding_dtype('input') == trt.tensorrt.DataType.FLOAT
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings, self.inputs, self.outputs = self.allocate_buffers(self.engine, input_shape=(1,3,32,1024), output_shape=(1, 256, 6625))
        
    def allocate_buffers(self, engine, input_shape=None, output_shape=None):
        inputs = []
        outputs = []
        bindings = []
        for binding in engine:
            if engine.binding_is_input(binding):
                if input_shape:
                    size = trt.volume(input_shape) * engine.max_batch_size
                else:
                    size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
                    print(engine.get_binding_shape(binding))
                dtype = trt.nptype(engine.get_binding_dtype(binding))
                # Allocate host and device buffers
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                # Append the device buffer to device bindings.
                bindings.append(int(device_mem))
                # Append to the appropriate list.
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                if output_shape:
                    size = trt.volume(output_shape) * engine.max_batch_size
                else:
                    size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
                    print(engine.get_binding_shape(binding))
                dtype = trt.nptype(engine.get_binding_dtype(binding))
                # Allocate host and device buffers
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                # Append the device buffer to device bindings.
                bindings.append(int(device_mem))
                # Append to the appropriate list.
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return bindings, inputs, outputs
    
    def do_inference(self, context, bindings, inputs, outputs, stream, input_shape):
        context.set_binding_shape(0, input_shape)
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]
    
    def preprocess(self, img):
        h,w,c = img.shape

        scale = h / 32
        w_ = w / scale
        w_ = math.ceil(w_/32)*32
        if w_ > 1024:
            print(w_)
            w_ = 1024
            print('long')
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
        # self.cfx.push()
        img = self.preprocess(image)
        input_shape = (1, img.shape[0], img.shape[1], img.shape[2])
        img = np.expand_dims(img, axis=0)
        
        self.inputs[0].host = np.ascontiguousarray(img)
        trt_outputs = self.do_inference(context=self.context,
                                        bindings=self.bindings,
                                        inputs=self.inputs,
                                        outputs=self.outputs,
                                        stream=self.stream,
                                        input_shape=input_shape)
        
        output = trt_outputs[0].reshape((-1, 6625))
        # self.cfx.pop()
        res_ids = np.argmax(output[:input_shape[-1]//4], axis=-1) 

        if not beamserach:
            res = self.decode(res_ids, self.id2char)
            return res
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
        # self.cfx.push()
        max_w = 0
        preprocess_lists = []

        for img in image_lists:
            img = self.preprocess(img)
            if img.shape[2] > max_w:
                max_w = img.shape[2]
            preprocess_lists.append(img)

        norm_preprocess_lists = self.batch_norm(preprocess_lists, max_w)
        img_batch = np.concatenate(norm_preprocess_lists)
        input_shape = (img_batch.shape[0], img_batch.shape[1], img_batch.shape[2], img_batch.shape[3])
        self.inputs[0].host = np.ascontiguousarray(img_batch)
        trt_outputs = self.do_inference(context=self.context,
                                        bindings=self.bindings,
                                        inputs=self.inputs,
                                        outputs=self.outputs,
                                        stream=self.stream,
                                        input_shape=input_shape)
        # self.cfx.pop()
        size_ = max_w//4*6625
        outputs_ = [trt_outputs[0][size_*i:size_*(i+1)].reshape(1, -1, 6625) for i in range(img_batch.shape[0])]
        
        outputs = np.concatenate(outputs_)
        res_ids = np.argmax(outputs, axis=-1) 
        preds_prob = np.max(outputs, axis=-1) 

        res_lists = []
        if not beamserach:
            for i in range(len(preprocess_lists)):
                res_lists.append([self.decode(res_ids[i], self.id2char), np.mean(preds_prob[i])])
            return res_lists
        else:
            return None
    
if __name__=='__main__':
    trt_model = './trt_model/ppocr_rec_free_dim.trt'
    demo_image_path = './ocr_demo/rec/rec_0.png'
    char_path = './ocr_demo/ppocr_keys_v1.txt'

    crnn_inference_trt = CrnnInference_trt(trt_model, char_path)
    print('trt model initial done')
    
    demo_image = cv2.imread(demo_image_path)
    print('demo_image.shape', demo_image.shape)
    
    for _ in range(20):
        _ = crnn_inference_trt.inference(demo_image)
    start = time.time()
    for _ in range(100):
        res = crnn_inference_trt.inference(demo_image)
    stop = time.time()
    print('per image spend time: ', (stop-start)/100)
    fps = 100/(stop-start)
    print('fps: ', fps)
    print(res)

