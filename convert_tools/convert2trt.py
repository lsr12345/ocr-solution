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


'''
onnx转trt实例代码
'''

BATCH_SIZE = 1

def load_onnx(onnx_file_path):
    with open(onnx_file_path, 'rb') as f:
        return f.read()
    
def set_net_batch(network, batch_size):
    shape = list(network.get_input(0).shape)
    shape[0] = batch_size
    network.get_input(0).shape = shape
    return network

def build_engine(onnx_file_path, width, height, verbose=False):
    onnx_data = load_onnx(onnx_file_path)

    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
    EXPLICIT_BATCH = [1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)]
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(*EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        if not parser.parse(onnx_data):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
        network = set_net_batch(network, BATCH_SIZE)

        builder.max_batch_size = BATCH_SIZE
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        config.set_flag(trt.BuilderFlag.FP16)
        profile = builder.create_optimization_profile()
        profile.set_shape(
            network.get_input(0).name, 
            (BATCH_SIZE, 3, height, width),      # min shape
            (BATCH_SIZE, 3, height*2, width*2),  # opt shape
            (BATCH_SIZE, 3, height*4, width*4))  # max shape
        config.add_optimization_profile(profile)
        engine = builder.build_engine(network, config)

        return engine
    
Onnx_Model_Path = './model_onnx/det.onnx'
engine_path = './trt_model/det.trt'
engine = build_engine(Onnx_Model_Path, width=320, height=320)

with open(engine_path, "wb") as f:
    f.write(engine.serialize())

