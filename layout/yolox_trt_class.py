'''
# Author: Shaoran Lu
# Date: 2021/08/11
# Email: lushaoran92@gmail.com
# Description: 版面分析tensorrt的推理类，infernence传入图片路径，返回值 [boxes, scores, cls_inds】
#              classes: ("body", "head", "formula", "form", "pic")

example:
    yolox_inference_trt = YoloxInference_trt(onnx_model)
    res = yolox_inference_trt.inference(image_path)
'''
# coding: utf-8

# In[1]:


import cv2
import time
import os

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

class YoloxInference_trt():

    def __init__(self, model_path, input_size=(960, 960), nms_thr=0.45, score_thr=0.1, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

        # self.cfx = cuda.Device(0).make_context()

        self.model_path = model_path
        self.input_size = input_size
        self.nms_thr = nms_thr
        self.score_thr = score_thr
        self.mean = mean
        self.std = std
        
        trt_logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(trt_logger, '')
        
        with open(model_path, 'rb') as f, trt.Runtime(trt_logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        assert self.engine.get_binding_dtype('input') == trt.tensorrt.DataType.FLOAT
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings, self.inputs, self.outputs = self.allocate_buffers(self.engine, input_shape=(1, 3, input_size[0], input_size[1]), output_shape=(1, 1, 18900, 10))
        
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
        
    def nms(self, boxes, scores, nms_thr):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]

        return keep

    def multiclass_nms(self, boxes, scores, nms_thr, score_thr):
        final_dets = []
        num_classes = scores.shape[1]
        for cls_ind in range(num_classes):
            cls_scores = scores[:, cls_ind]
            valid_score_mask = cls_scores > score_thr
            if valid_score_mask.sum() == 0:
                continue
            else:
                valid_scores = cls_scores[valid_score_mask]
                valid_boxes = boxes[valid_score_mask]
                keep = self.nms(valid_boxes, valid_scores, nms_thr)
                if len(keep) > 0:
                    cls_inds = np.ones((len(keep), 1)) * cls_ind
                    dets = np.concatenate(
                        [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                    )
                    final_dets.append(dets)
        if len(final_dets) == 0:
            return None
        return np.concatenate(final_dets, 0)

    def vis(self, img, boxes, scores, cls_ids, conf=0.5, class_names=None):

        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            if score < conf:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])

            color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
            text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
            txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX

            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

            txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(
                img,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                txt_bk_color,
                -1
            )
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

        return img

    def preprocess(self, image, swap=(2, 0, 1)):
        if len(image.shape) == 3:
            padded_img = np.ones((self.input_size[0], self.input_size[1], 3)) * 114.0
        else:
            padded_img = np.ones(self.input_size) * 114.0
        img = np.array(image)
        r = min(self.input_size[0] / img.shape[0], self.input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img[:, :, ::-1]
        padded_img /= 255.0
        if self.mean is not None:
            padded_img -= self.mean
        if self.std is not None:
            padded_img /= self.std
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def postprocess(self, outputs, ratio, origin_img, p6=False, visual=False):
        grids = []
        expanded_strides = []

        if not p6:
            strides = [8, 16, 32]

        else:
            strides = [8, 16, 32, 64]

        hsizes = [self.input_size[0] // stride for stride in strides]
        wsizes = [self.input_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)

        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        outputs = outputs[0]

        boxes = outputs[:, :4]
        scores = outputs[:, 4:5] * outputs[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio

        dets = self.multiclass_nms(boxes_xyxy, scores, nms_thr=self.nms_thr, score_thr=self.score_thr)
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            if visual:
                vis_img = self.vis(origin_img, final_boxes, final_scores, final_cls_inds,
                                   conf=self.score_thr, class_names=("body", "head", "formula", "form", "pic"))
                return [vis_img, final_boxes, final_scores, final_cls_inds]
            else:
                return [None, final_boxes, final_scores, final_cls_inds]
        else:
            return [None]

    def inference(self, image, visual_save=None, visual_name='visual_img_yolox.jpg'):
        # image = cv2.imread(img_path)
        # self.cfx.push()
        img, ratio = self.preprocess(image)
        input_shape = (1, img.shape[0], img.shape[1], img.shape[2])
        img = np.expand_dims(img, axis=0)

        self.inputs[0].host = np.ascontiguousarray(img)
        trt_outputs = self.do_inference(context=self.context,
                                        bindings=self.bindings,
                                        inputs=self.inputs,
                                        outputs=self.outputs,
                                        stream=self.stream,
                                        input_shape=input_shape)
        
        outputs_ = trt_outputs[0].reshape(1, -1, 10)

        # self.cfx.pop()

        res = self.postprocess(outputs_, ratio, origin_img=image, p6=False, visual=False if visual_save is None else True)

        if len(res) == 1:
            print('Dets ob nums == 0')
            return None

        if visual_save is not None:
            if not os.path.exists(visual_save):
                os.makedirs(visual_save)
            cv2.imwrite(os.path.join(visual_save, visual_name), res[0])

        return res[1:]

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)


if __name__=='__main__':
    onnx_model = '/home/shaoran/company_work/starsee/ocr_solution/models/yolox_l_layout.trt'
    demo_image = '/home/shaoran/company_work/starsee/ocr_solution/onnx_inference/demo/3.jpg'
    visual_dir = '/home/shaoran/company_work/starsee/ocr_solution/visual_results'

    demo_image = cv2.imread(demo_image)
    input_size = (960, 960)

    yolox_inference_trt = YoloxInference_trt(onnx_model, input_size, nms_thr=0.45,
                                     score_thr=0.75, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    print('trt model initial done')
    for _ in range(1):
        _ = yolox_inference_trt.inference(demo_image, visual_save=visual_dir)
    start = time.time()
    for _ in range(1):
        res = yolox_inference_trt.inference(demo_image, visual_save=visual_dir)
    stop = time.time()
    print('per image spend time: ', (stop-start)/100)
    print(res)