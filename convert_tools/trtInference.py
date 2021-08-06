# Author: Shaoran Lu
# Date: 2021/08/6
# Email: lushaoran92@gmail.com
# Description:
# coding: utf-8

# In[1]:


import tensorrt as trt
import numpy as np
import cv2
import os


# In[ ]:


'''
trt推理代码
'''

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return 'Host:\n' + str(self.host) + '\nDevice:\n' + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine
def allocate_buffers(engine, input_shape=None, output_shape=None):
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

def do_inference(context, bindings, inputs, outputs, stream, input_shape):
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

input_shape = (640, 640)

trt_logger = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(trt_logger, '')

engine_path = './trt_model/det.trt'
with open(engine_path, 'rb') as f, trt.Runtime(trt_logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()
stream = cuda.Stream()

bindings, inputs, outputs = allocate_buffers(engine, input_shape=(1, 3,input_shape[0], input_shape[1]), output_shape=(1, 1, input_shape[0],input_shape[1]))

img = cv2.imread('./demo.jpg')
img_resized = cv2.resize(img)
img_resized = img_resized.transpose((2, 0, 1))
img_resized = img_resized.astype('float32')
input_shape = (1, img_resized.shape[0], img_resized.shape[1], img_resized.shape[2])
img_resized = np.expand_dims(img_resized, axis=0)


inputs[0].host = np.ascontiguousarray(img_resized)
trt_outputs = do_inference_v2(context=context,
                                bindings=bindings,
                                inputs=inputs,
                                outputs=outputs,
                                stream=stream,
                                input_shape=input_shape)

