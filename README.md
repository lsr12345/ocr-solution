## 简介  
OCR解决方案推理部分，通过配置Config.yaml文件选择Pipline中嵌入的算法模块。  
## 启动
onnx_inference/base_onnx_inference.py  
## 配置  
* use_layout：是否启用版面分析  
* use_angle_cls：是否启用角度纠正
* input_size：设置输入图片大小
* crnn_chars：文字识别字符集
* model_path：各算法模型路径（绝对路径）
* visual_flag：是否保存中间可视结果
* visual_path：可是结果存放位置
* crnn.inference_batch：文字识别batch
* angle_cls.input_size：角度分类图片大小  
## 结构
  * config：配置文件及字符集
  * convert_tools：onnx转tensorrt脚本以及tensorrt的python预测代码示例
  * correction：纠错模块（暂无）
  * formula：公式识别训练代码
  * layout：版面分析模块
  * ocr_angle_cls：角度分类模块
  * ocr_det：文件检测模块
  * ocr_rec：文字识别模块
  * onnx_infernece：推理入口
  * table_rec：表格重建模块（暂无）
  * trt_inference：tensorrt推理入口（暂无）

## 模型效果说明  
* 公式识别模型在10W自动生成的数据训练完成，在1W数据上验证准确率为80+%
* 版面分析模型在7500张图训练完成，在300张数据map为86

