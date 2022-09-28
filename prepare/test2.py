#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/22 下午2:15
# @Author : wangyangyang

# -*- coding: utf-8 -*-
import cv2

# 查找棋盘格 角点
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

# 棋盘格参数
corners_vertical = 60  # 纵向角点个数;
corners_horizontal = 70  # 横向角点个数;
pattern_size = (corners_vertical, corners_horizontal)


def find_corners_sb(img):
    """
    查找棋盘格角点函数 SB升级款
    :param img: 处理原图
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 查找棋盘格角点;
    ret, corners = cv2.findChessboardCornersSB(gray, pattern_size, cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
    if ret:
        # 精细查找角点
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # 显示角点
        cv2.drawChessboardCorners(img, pattern_size, corners2, ret)


def find_corners(img):
    """
    查找棋盘格角点函数
    :param img: 处理原图
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点;
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH +
                                             cv2.CALIB_CB_FAST_CHECK +
                                             cv2.CALIB_CB_FILTER_QUADS)
    if ret:
        # 精细查找角点
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # 显示角点
        cv2.drawChessboardCorners(img, pattern_size, corners2, ret)


def main():
    # 1.创建显示窗口
    cv2.namedWindow("img", 0)
    cv2.resizeWindow("img", 1075, 900)

    file_path = ("/home/yang/Documents/AI_test_demo/download.png")
    img_src = cv2.imread(file_path)

    if img_src is not None:
        # 执行查找角点算法
        find_corners_sb(img_src)
        # find_corners(img_src)

        # 显示图片
        cv2.imshow("img", img_src)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
