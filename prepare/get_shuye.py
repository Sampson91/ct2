#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/27 下午3:01
# @Author : wangyangyang
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.draw import disk

img = cv2.imread('/media/yang/sys1/train_data/style2_model/output/47.jpg')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_blur = cv2.medianBlur(img, 71)
img_diff = cv2.bitwise_and(img_blur, img)
# img_diff = cv2.adaptiveThreshold(img_diff, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)

ret,thresh1 = cv2.threshold(img_diff,127,155,cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17))
# kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(35,35))
opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
plt.imshow(thresh1, cmap='gray')
plt.show()
