#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/22 下午2:11
# @Author : wangyangyang


import cv2
import numpy as np
import matplotlib.pyplot as plt
"""
角点检测 goodFeaturesToTrack()
"""
img = cv2.imread("/home/yang/Documents/AI_test_demo/grayscale_graph.png")
img = cv2.imread("/home/yang/Documents/AI_test_demo/blank.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray, 5, 0.1, 50)
corners = np.int0(corners)

for corner in corners:
    x, y = corner.ravel()
    cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

# 缩小图像
height, width = img.shape[:2]
size = (int(width * 0.4), int(height * 0.4))
img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

# cv2.imshow("img", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()