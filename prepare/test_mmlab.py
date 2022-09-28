#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/13 下午4:51
# @Author : wangyangyang
from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import cv2
import numpy as np

config_file = "/home/yang/Documents/workplace_pix2pix/pspnet_r50-d8_512x1024_40k_cityscapes.py"
checkpoint_file = "/home/yang/Documents/workplace_pix2pix/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth"


# 通过配置文件和模型权重文件构建模型
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# 对单张图片进行推理并展示结果
# img = './test/test1.png'  # or img = mmcv.imread(img), which will only load it once
img = '/media/yang/sys1/train_data/style2_model/output1/47out1.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_segmentor(model, img)
# 在新窗口中可视化推理结果
# model.show_result(img, result, show=True)

# 或将可视化结果存储在文件中
# 你可以修改 opacity 在(0,1]之间的取值来改变绘制好的分割图的透明度
model.show_result(img, result, out_file='/media/yang/sys1/train_data/style2_model/output1/47out2.jpg', opacity=0.5)
file_path = "./result00.jpg"
image = cv2.imread(file_path)
B, G, R = cv2.split(image)
labelId_image = "color00.png"
result = np.array(result)
cv2.imwrite(labelId_image, result[0])
