#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/27 下午1:28
# @Author : wangyangyang
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.draw import disk


def fft_in_np(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    rows, cols = img.shape
    row_half, col_half = rows // 2, cols // 2
    mask = np.ones((rows, cols))
    mask = mask * 1.5
    rr, cc = disk((row_half, col_half), 10)
    mask[rr, cc] = 1
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    cv2.imwrite("/media/yang/sys1/train_data/style2_model/output/47out.jpg",img_back)

    plt.subplot(131), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(img_back, cmap='gray')
    plt.title('img_back'), plt.xticks([]), plt.yticks([])
    plt.show()
#
#
img = cv2.imread('/media/yang/sys1/train_data/style2_model/output/47.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
fft_in_np(img)


#  自适应直方图均衡
# clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
# cl1 = clahe.apply(img)
# cv2.imwrite("/media/yang/sys1/train_data/style2_model/output/47out1.jpg", cl1)
