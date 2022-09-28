#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/8/25 下午5:13
# @Author : wangyangyang
import os
import numpy as np
import cv2


def gaussian(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    b = np.array([[2, 4,  5,  2,  2],
               [4, 9,  12, 9,  4],
               [5, 12, 15, 12, 5],
               [4, 9,  12, 9,  4],
               [2, 4,  5,  4,  2]]) / 156
    kernel = np.zeros(im.shape)
    kernel[:b.shape[0], :b.shape[1]] = b

    fim = np.fft.fft2(im)
    fkernel = np.fft.fft2(kernel)
    fil_im = np.fft.ifft2(fim * fkernel)

    return abs(fil_im).astype(int)


def process_image(path, file, save_path):
    total_path = os.path.join(path, file)
    img = cv2.imread(total_path)
    num_bilateral = 7
    img_color = img
    for i in range(num_bilateral):
        img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)
    img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    # img_cartoon = cv2.bitwise_and(img_color, img_edge)
    img_edge =255 - cv2.Canny(img_gray, 120, 150)
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    img_cartoon = cv2.bitwise_and(img_color, img_edge)
    # img_cartoon = img_edge
    saving_path = os.path.join(save_path, file)
    img_cartoon = cv2.medianBlur(img_cartoon, 3)
    img_cartoon = gaussian(img_cartoon)
    img_cartoon = cv2.blur(img_cartoon, (5, 5))
    cv2.imwrite(saving_path, img_cartoon)


if __name__ == "__main__":
    # path = "/media/yang/sys/kitt2_data100/image/"
    # out_path = "/media/yang/sys/kitt2_data100/cartoon_image/"

    path = "/media/yang/sys1/train_data/left_jz"
    out_path = "/media/yang/sys1/train_data/left_jz_gray_carton1"
    files = os.listdir(path)
    for file in files:
        process_image(path, file, out_path)
