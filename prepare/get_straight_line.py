#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/22 下午4:05
# @Author : wangyangyang
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt


def GetAngle(line1, line2):
    """
    计算两条线段之间的夹角
    :param line1:
    :param line2:
    :return angle:
    """
    # dx1 = line1.Point1.X - line1.Point2.X
    # dy1 = line1.Point1.Y - line1.Point2.Y
    # dx2 = line2.Point1.X - line2.Point2.X
    # dy2 = line2.Point1.Y - line2.Point2.Y
    dx1 = line1[0] - line1[2]
    dy1 = line1[1] - line1[3]
    dx2 = line2[0] - line2[2]
    dy2 = line2[1] - line2[3]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180 / math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180 / math.pi)
    # print(angle2)
    if angle1 * angle2 >= 0:
        insideAngle = abs(angle1 - angle2)
    else:
        insideAngle = abs(angle1) + abs(angle2)
        if insideAngle > 180:
            insideAngle = 360 - insideAngle
    insideAngle = insideAngle % 180
    return insideAngle


def intersectionLines(a1, a2, b1, b2):
    return (
        ((a1[0] * a2[1] - a1[1] * a2[0]) * (b1[0] - b2[0]) -
         (b1[0] * b2[1] - b1[1] * b2[0]) * (a1[0] - a2[0])) /
        ((a1[0] - a2[0]) * (b1[1] - b2[1]) -
         (a1[1] - a2[1]) * (b1[0] - b2[0])),
        ((a1[0] * a2[1] - a1[1] * a2[0]) * (b1[1] - b2[1]) -
         (b1[0] * b2[1] - b1[1] * b2[0]) * (a1[1] - a2[1])) /
        ((a1[0] - a2[0]) * (b1[1] - b2[1]) -
         (a1[1] - a2[1]) * (b1[0] - b2[0]))
    )


def p_to_l(point, l_p1, l_p2):
    if l_p1[0] - l_p2[0] == 0:
        crop = (l_p1[0], point[1])
        # print("crop",crop)
    elif l_p1[1] - l_p2[1] == 0:
        crop = (point[0], l_p1[1])
    else:
        k_l = (l_p1[1] - l_p2[1]) / (l_p1[0] - l_p2[0])
        # print(k_l)
        k = -1 / k_l
        # print(k)
        b = point[1] - k * point[0]
        # print(b)
        x = l_p1[0]
        # print(x)
        y = k * x + b
        point2 = (x, y)
        # print(point)
        # print(point2)
        # print(l_p1,l_p2)
        crop = intersectionLines(point, point2, l_p1, l_p2)
    return crop


def p_to_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def to_overlap_pts(a1, a2, b1, b2):
    l1_p1, l1_p2, l2_p1, l2_p2 = (0., 0.), (0., 0.), (0., 0.), (0., 0.)
    if a1[1] > a2[1]:
        l1_hight = a1
        l1_low = a2
    else:
        l1_hight = a2
        l1_low = a1
    if b1[1] > b2[1]:
        l2_hight = b1
        l2_low = b2
    else:
        l2_hight = b2
        l2_low = b1


    point1_state = 0
    point2_state = 0
    # 直线没有重合部分l1
    l1_l = p_to_l(l1_low, l2_low, l2_hight)

    l1_h = p_to_l(l1_hight, l2_low, l2_hight)
    # print("l1",l1_h)
    # print("l2_low",l2_low)
    # print("l2_hight",l2_hight)
    list = [l2_low[0], l2_hight[0]]
    l = min(list)
    r = max(list)
    if l <= l1_h[0] <= r:
        point1_state = 1
    if l <= l1_l[0] <= r:
        point2_state = 1
    print("state", point1_state, point2_state)
    if point1_state == 0 and point2_state == 0:
        print("框选错误，请重新框选")
    else:
        l1_b = 0
        l1_t = 0
        l2_b = 0
        l2_t = 0
        # 左边直线
        if l2_low[0] < l2_hight[0]:
            l = l2_low[0]
            r = l2_hight[0]
        else:
            r = l2_low[0]
            l = l2_hight[0]

        cro_p1 = p_to_l(l1_low, l2_low, l2_hight)
        if l <= cro_p1[0] <= r:
            p1 = l1_low
            p2 = cro_p1
            cro_p1 = (cro_p1[0], cro_p1[1])
            l1_b = 1
        else:
            pass
        cro_p2 = p_to_l(l1_hight, l2_low, l2_hight)
        if l <= cro_p2[0] <= r:

            p1 = l1_hight
            p2 = cro_p2
            cro_p2 = (cro_p2[0], cro_p2[1])
            l1_t = 1
        else:
            pass

        # 右边直线
        if l1_low[0] < l1_hight[0]:
            l = l1_low[0]
            r = l1_hight[0]
        else:
            r = l1_low[0]
            l = l1_hight[0]
        cro_p3 = p_to_l(l2_low, l1_low, l1_hight)
        if l <= cro_p3[0] <= r:
            p1 = l2_low
            p2 = cro_p3
            cro_p3 = (cro_p3[0], cro_p3[1])
            l2_b = 1
        else:
            pass
        cro_p4 = p_to_l(l2_hight, l1_low, l1_hight)
        if l <= cro_p4[0] <= r:
            p1 = l2_hight
            p2 = cro_p4
            cro_p4 = (cro_p4[0], cro_p4[1])
            l2_t = 1
        else:
            pass
        # 右直线投影到左直线
        if l1_b + l1_t == 0 and l2_b + l2_t == 2:
            l1_p1 = cro_p3
            l1_p2 = cro_p4
            l2_p1 = l2_low
            l2_p2 = l2_hight
        # 左直线投影到右直线
        elif l1_b + l1_t == 2 and l2_b + l2_t == 0:
            l1_p1 = l1_low
            l1_p2 = l1_hight
            l2_p1 = cro_p1
            l2_p2 = cro_p2
        # 右直线投影到作直线，左直线一个点投影到右直线
        elif l1_b + l1_t == 1 and l2_b + l2_t == 2:
            # 左直线上点
            if l1_t == 1:
                l2_p1 = l2_low
                l2_p2 = cro_p2
            # 左直线下
            else:
                l2_p1 = cro_p1
                l2_p2 = l2_hight
            l1_p1 = cro_p3
            l1_p2 = cro_p4
        # 左直线投影到右直线，右直线一个点投影到左直线
        elif l1_b + l1_t == 2 and l2_b + l2_t == 1:
            # 右直线上点
            if l1_t == 1:
                l1_p1 = l1_low
                l1_p2 = cro_p4
            # 右直线下点
            else:
                l1_p1 = cro_p3
                l1_p2 = l1_hight
            l2_p1 = cro_p1
            l2_p2 = cro_p2
        elif l1_b == 1 and l2_t == 1:
            l1_p1 = l1_low
            l1_p2 = cro_p4
            l2_p1 = cro_p1
            l2_p2 = l2_hight
        elif l1_t == 1 and l2_b == 1:
            l1_p1 = cro_p3
            l1_p2 = l1_hight
            l2_p1 = l2_low
            l2_p2 = cro_p2

    return l1_p1, l1_p2, l2_p1, l2_p2


def coincidence_part(a1, a2, b1, b2):
    # img=cv2.imread("1030_1_undistorted.png")
    l1_p1, l1_p2, l2_p1, l2_p2 = (0., 0.), (0., 0.), (0., 0.), (0., 0.)
    if a1[1] > a2[1]:
        l1_hight = a1
        l1_low = a2
    else:
        l1_hight = a2
        l1_low = a1
    if b1[1] > b2[1]:
        l2_hight = b1
        l2_low = b2
    else:
        l2_hight = b2
        l2_low = b1
    # 直线没有重合部分l1

    if l1_hight[1] <= l2_low[1] or l1_low[1] >= l2_hight[1]:
        print("框选错误，请重新框选")
    else:
        l1_b = 0
        l1_t = 0
        l2_b = 0
        l2_t = 0
        # 左边直线
        if l2_low[0] < l2_hight[0]:
            l = l2_low[0]
            r = l2_hight[0]
        else:
            r = l2_low[0]
            l = l2_hight[0]

        cro_p1 = p_to_l(l1_low, l2_low, l2_hight)
        if l <= cro_p1[0] <= r:
            p1 = l1_low
            p2 = cro_p1
            cro_p1 = (cro_p1[0], cro_p1[1])
            l1_b = 1
        else:
            pass
        cro_p2 = p_to_l(l1_hight, l2_low, l2_hight)
        if l <= cro_p2[0] <= r:

            p1 = l1_hight
            p2 = cro_p2
            cro_p2 = (cro_p2[0], cro_p2[1])
            l1_t = 1
        else:
            pass

        # 右边直线
        if l1_low[0] < l1_hight[0]:
            l = l1_low[0]
            r = l1_hight[0]
        else:
            r = l1_low[0]
            l = l1_hight[0]
        cro_p3 = p_to_l(l2_low, l1_low, l1_hight)
        if l <= cro_p3[0] <= r:
            p1 = l2_low
            p2 = cro_p3
            cro_p3 = (cro_p3[0], cro_p3[1])
            l2_b = 1
        else:
            pass
        cro_p4 = p_to_l(l2_hight, l1_low, l1_hight)
        if l <= cro_p4[0] <= r:
            p1 = l2_hight
            p2 = cro_p4
            cro_p4 = (cro_p4[0], cro_p4[1])
            l2_t = 1
        else:
            pass
        # 右直线投影到左直线
        if l1_b + l1_t == 0 and l2_b + l2_t == 2:
            l1_p1 = cro_p3
            l1_p2 = cro_p4
            l2_p1 = l2_low
            l2_p2 = l2_hight
        # 左直线投影到右直线
        elif l1_b + l1_t == 2 and l2_b + l2_t == 0:
            l1_p1 = l1_low
            l1_p2 = l1_hight
            l2_p1 = cro_p1
            l2_p2 = cro_p2
        # 右直线投影到作直线，左直线一个点投影到右直线
        elif l1_b + l1_t == 1 and l2_b + l2_t == 2:
            # 左直线上点
            if l1_t == 1:
                l2_p1 = l2_low
                l2_p2 = cro_p2
            # 左直线下
            else:
                l2_p1 = cro_p1
                l2_p2 = l2_hight
            l1_p1 = cro_p3
            l1_p2 = cro_p4
        # 左直线投影到右直线，右直线一个点投影到左直线
        elif l1_b + l1_t == 2 and l2_b + l2_t == 1:
            # 右直线上点
            if l1_t == 1:
                l1_p1 = l1_low
                l1_p2 = cro_p4
            # 右直线下点
            else:
                l1_p1 = cro_p3
                l1_p2 = l1_hight
            l2_p1 = cro_p1
            l2_p2 = cro_p2
        elif l1_b == 1 and l2_t == 1:
            l1_p1 = l1_low
            l1_p2 = cro_p4
            l2_p1 = cro_p1
            l2_p2 = l2_hight
        elif l1_t == 1 and l2_b == 1:
            l1_p1 = cro_p3
            l1_p2 = l1_hight
            l2_p1 = l2_low
            l2_p2 = cro_p2

        # print(l1_b, l1_t, l2_b, l2_t)
        # print(l1_p1,l1_p2,l2_p1,l2_p2)
        try:
            l1_c = ((l1_p1[0] + l1_p2[0]) / 2, (l1_p1[1] + l1_p2[1]) / 2)
        except:
            print(l1_p1)
        l1_to_l2 = p_to_l(l1_c, l2_p1, l2_p2)
        l1_to_l2 = (l1_to_l2[0], l1_to_l2[1])

        l2_c = ((l2_p1[0] + l2_p2[0]) / 2, (l2_p1[1] + l2_p2[1]) / 2)
        # print(l2_c,l1_p1,l1_p2)

        l2_to_l1 = p_to_l(l2_c, l1_p1, l1_p2)
        l2_to_l1 = (int(l2_to_l1[0]), int(l2_to_l1[1]))

        l1 = p_to_distance(l1_c, l1_to_l2)
        l2 = p_to_distance(l2_c, l2_to_l1)
        # print(l1,l2)
        distance = (l1 + l2) / 2

        # print(distance)
        return distance

# coincidence_part((200,100),(200,600),(800,100),(800,621))
# coincidence_part((100,100),(600,100),(200,400),(500,400))
img1 = cv2.imread("/home/yang/Documents/AI_test_demo/download.png")
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(gray, 250, 0.1, 50)
corners = np.int0(corners)

cor_list = []
for corner in corners:
    x, y = corner.ravel()
    cor_list.append((x, y))

n = len(cor_list)
cor_line_list = []
for i in range(n):
    for j in range(n):
        if i != j:
            cor_line_list.append((cor_list[i][0],cor_list[i][1], cor_list[j][0],cor_list[j][1]))


img = cv2.GaussianBlur(img1, (3, 3), 0)
edges = cv2.Canny(img, 90, 120, apertureSize=3)

minLineLength = 100
maxLineGap = 15
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength, maxLineGap)
huf_line = []
for line in lines:
    for x1, y1, x2, y2 in line:
        huf_line.append((x1, y1, x2, y2))


# decide angle and distant
n_line = len(huf_line)
# for i in range(n_line):
#     for j in cor_line_list:
#         angle = GetAngle(huf_line[i], j)
#
#         if angle == 0:
#             dist = coincidence_part((huf_line[i][0], huf_line[i][1]), (huf_line[i][2], huf_line[i][3]), (j[0], j[1]),
#                                     (j[2], j[3]))
#             if dist is None:
#                 dist = -1
#             if 0<dist < 1:
#                 cv2.line(img1, (j[0], j[1]), (j[2], j[3]), (0, 255, 0), 2)

for corner in corners:
    x, y = corner.ravel()
    cv2.circle(img1, (x, y), 5, (0, 0, 255), -1)

for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(img1, (x1, y1), (x2, y2), (0, 255, 0), 2)

img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
