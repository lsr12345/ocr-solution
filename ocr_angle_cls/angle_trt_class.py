'''
# Author: Shaoran Lu
# Date: 2021/08/11
# Email: lushaoran92@gmail.com
# Description: OCR文本角度分类的tensorrt推理类，infernence传入图片，返回值 角度分类类别

example:
    angle_inference = AngleClsInference_trt(onnx_model, char_path)
    res = angle_inference.inference(image)
'''
# coding: utf-8

# In[1]:
import cv2
import time
import os

import numpy as np
import math
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


class AngleClsInference_trt():

    def __init__(self, model_path, input_size=(720, 720), angle_thresh=0.9, angel_label_list=['0', '90', '180', '270']):
        # self.cfx = cuda.Device(0).make_context()
        self.model_path = model_path
        self.input_size = input_size
        self.angel_label_list = angel_label_list
        
        trt_logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(trt_logger, '')
        
        with open(model_path, 'rb') as f, trt.Runtime(trt_logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        assert self.engine.get_binding_dtype('input') == trt.tensorrt.DataType.FLOAT
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings, self.inputs, self.outputs = self.allocate_buffers(self.engine, input_shape=(1,3,self.input_size[0],self.input_size[1]), output_shape=(1, 1, len(self.angel_label_list)))
        
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
        # self.cfx.push()
        img = self.AnglePreprocess(image)
        input_shape = (1, img.shape[0], img.shape[1], img.shape[2])
        img = np.expand_dims(img, axis=0)
        
        self.inputs[0].host = np.ascontiguousarray(img)
        trt_outputs = self.do_inference(context=self.context,
                                        bindings=self.bindings,
                                        inputs=self.inputs,
                                        outputs=self.outputs,
                                        stream=self.stream,
                                        input_shape=input_shape)
        # self.cfx.pop()
        output = trt_outputs[0].reshape((4))
        output_numpy = np.array(output)
        label, score = self.AnglePostprocess(output_numpy)
        return [label, score]
    
if __name__=='__main__':
    trt_model = './trt_model/angle_cls.trt'
    demo_image_path = './ocr_demo/layout/2.jpg'

    angle_inference_trt = AngleClsInference_trt(trt_model)
    print('trt model initial done')
    
    demo_image = cv2.imread(demo_image_path)
    print('demo_image.shape', demo_image.shape)
    
    for _ in range(20):
        _ = angle_inference_trt.inference(demo_image)
    start = time.time()
    for _ in range(100):
        res = angle_inference_trt.inference(demo_image)
    stop = time.time()
    print('per image spend time: ', (stop-start)/100)
    fps = 100/(stop-start)
    print('fps: ', fps)
    print(res)

