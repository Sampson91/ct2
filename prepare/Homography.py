#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/23 下午1:30
# @Author : wangyangyang
import cv2
import matplotlib.pyplot as plt
import numpy as np

im_src = cv2.imread('/home/yang/Documents/AI_test_demo/grayscale_graph.png')
gray = cv2.cvtColor(im_src, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray, 20, 0.1, 50)
corners_src = np.int0(corners)
# Four corners of the book in source image.
# 书的四个角在源图像
# pts_src = np.array([[141, 131], [480, 159], [493, 630], [64, 601]])

# Read destination image.
# 读取目标图像
im_dst = cv2.imread('/home/yang/Documents/AI_test_demo/blank.png')
gray = cv2.cvtColor(im_dst, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray, 20, 0.1, 50)
corners_dst = np.int0(corners)
# Four corners of the book in destination image.书的四个角在目的图像
# pts_dst = np.array([[318, 256], [534, 372], [316, 670], [73, 473]])

# Calculate Homography
# 计算Homography矩阵
# h, status = cv2.findHomography(pts_src, pts_dst)
h, status = cv2.findHomography(corners_src, corners_dst)

# Warp source image to destination based on homography
# 基于单线图实现经源图像到目标图像
im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))

# Display images
plt.figure(figsize=(16, 16))
plt.subplot(1, 3, 1)
plt.imshow(im_src)
plt.xticks([])
plt.yticks([])
plt.subplot(1, 3, 2)
plt.imshow(im_dst)
plt.xticks([])
plt.yticks([])
plt.subplot(1, 3, 3)
plt.imshow(im_out)
plt.xticks([])
plt.yticks([])
plt.show()