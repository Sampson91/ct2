#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/20 下午4:26
# @Author : wangyangyang
import matplotlib.pyplot as plt
import cv2
import numpy as np

# img = cv2.imread("/home/yang/Documents/AI_test_demo/finnal_output.png")
img = cv2.imread("/home/yang/Documents/AI_test_demo/blank.png")
# img = cv2.imread("/home/yang/Documents/AI_test_demo/ref_47.jpg")
# img = cv2.imread("/home/yang/Documents/AI_test_demo/download.png")

img = cv2.GaussianBlur(img, (3, 3), 0)
edges = cv2.Canny(img, 90, 120, apertureSize=3)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 60)
result = img.copy()
lines = lines.reshape((lines.shape[0],lines.shape[2]))
# 经验参数
minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 70, minLineLength, maxLineGap)

for line in lines:
    for x1, y1, x2, y2  in line:
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

plt.imshow(edges)
plt.show()
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
# cv2.waitKey(0)
# cv2.destroyAllWindows()