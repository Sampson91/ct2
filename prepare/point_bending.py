import cv2
import numpy as np
np.set_printoptions(threshold=np.inf)
img = cv2.imread("/home/yang/Documents/AI_test_demo/download.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray,150,160,cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# contours[2]输出第二个物体的顶点坐标
print(str(contours[2].reshape(-1)).replace('\n',',').replace(' ',',').replace(',,',',').replace(',,',',').replace('[,','['))
# 将轮廓画到原图，-1表示全部绘制

for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True) #取1%為\epsilon 值
        (x, y, w, h) = cv2.boundingRect(approx)
        # if len(approx) == 2:
                # cv2.drawContours(img, [c], -1, (0, 255, 255), 3)
                # (x, y, w, h) = cv2.boundingRect(approx)
cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyWindow("img")
