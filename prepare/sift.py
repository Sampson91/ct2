import cv2
import numpy as np
import matplotlib.pyplot as plt


def sift_func(img_path1, img_path2):
    img_1 = cv2.imread(img_path1)

    img_2 = cv2.imread(img_path2)
    gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    # SIFT特征计算
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=0,contrastThreshold=0.01, edgeThreshold=120,sigma=2.2)
    psd_kp1, psd_des1 = sift.detectAndCompute(gray_1, None)
    psd_kp2, psd_des2 = sift.detectAndCompute(gray_2, None)

    # Flann特征匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=1500)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(psd_des1, psd_des2, k=2)
    goodMatch = []
    for m, n in matches:
        # goodMatch是经过筛选的优质配对，如果2个配对中第一匹配的距离小于第二匹配的距离的1/2，
        # 基本可以说明这个第一配对是两幅图像中独特的，不重复的特征点,可以保留。
        if m.distance < 0.99 * n.distance:
            goodMatch.append(n)

    # 增加一个维度
    goodMatch = np.expand_dims(goodMatch, 1)
    print(len(goodMatch))
    img_out = cv2.drawMatchesKnn(img_1, psd_kp1,img_2, psd_kp2, goodMatch, None, flags=2)
    img_src = []
    img_blank = []
    goodMatch_bast = []
    best_dist = 3
    for good in goodMatch:
        kp1_id = good[0].queryIdx
        kp2_id = good[0].trainIdx
        src = psd_kp1[kp1_id].pt
        blank = psd_kp2[kp2_id].pt
        if abs(src[1] - blank[1]) > best_dist:
            continue
        goodMatch_bast.append(good)
        img_src.append(psd_kp1[kp1_id].pt)
        img_blank.append(psd_kp2[kp2_id].pt)
    img_src = np.float32(img_src)
    img_blank = np.float32(img_blank)

    # jisuan benzhi juzheng
    # Cam_Mat = np.array([[1,0,0],[0,1,0],[0,0,1]])
    # D = cv2.findEssentialMat(img_src,img_blank,Cam_Mat,cv2.RANSAC,0.99,1)
    # D = D[0]
    # R = np.array([[0,0],[0,0]])
    # T = np.array([0,0])
    # rec_p = cv2.recoverPose(D,img_src,img_blank,Cam_Mat)
    #
    # res_out = cv2.warpAffine(img_1,rec_p[1][:2],(img_2.shape[1], img_2.shape[0]))
    points1 = np.float32([[30, 30], [100, 40], [40, 100]])
    M = cv2.getAffineTransform(img_src[:4],img_blank[:4])
    Affine_img = cv2.warpAffine(img_1,M,(img_2.shape[1], img_2.shape[0]))

    h, status = cv2.findHomography(img_src, img_blank)
    img_out = cv2.drawMatchesKnn(img_1, psd_kp1,img_2, psd_kp2, goodMatch_bast, None, flags=2)
    im_out = cv2.warpPerspective(img_1, h, (img_2.shape[1], img_2.shape[0]))
    return Affine_img, Affine_img


if __name__ == '__main__':
    img_path1 = '/home/yang/Documents/AI_test_demo/grayscale_graph.png'
    img_path2 = '/home/yang/Documents/AI_test_demo/blank.png'
    im_src = cv2.imread(img_path1)
    im_dst = cv2.imread(img_path2)
    img_out,im_out = sift_func(img_path1, img_path2)

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

    cv2.imshow('image', img_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()