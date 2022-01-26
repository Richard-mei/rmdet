import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse


def SIFT_(image1, image2, save_path=None):
    sift = cv2.SIFT_create()
    img1 = cv2.imread(image1)
    kp1, des1 = sift.detectAndCompute(img1, None)  # des是描述子
    img2 = cv2.imread(image2)
    kp2, des2 = sift.detectAndCompute(img2, None)  # des是描述子
    img3 = cv2.drawKeypoints(img1, kp1, img1, color=(255, 0, 255))  # 画出特征点，并显示为红色圆圈
    img4 = cv2.drawKeypoints(img2, kp2, img2, color=(255, 0, 255))  # 画出特征点，并显示为红色圆圈
    hmerge = np.hstack((img3, img4))  # 水平拼接
    cv2.imshow("point", hmerge)  # 拼接显示为gray

    # BFMatcher解决匹配
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # 调整ratio
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)

    """
    计算仿射变换矩阵
    """
    img5 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=2)
    # img5 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    if save_path is not None:
        cv2.imwrite(f'{save_path}/BF_SIFTmatch.png', img5)
    while True:
        cv2.imshow("BF_SIFTmatch", img5)
        key = cv2.waitKey(0)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break


def ORB_(image1, image2, save_path=None):
    orb = cv2.ORB_create()
    img1 = cv2.imread(image1)
    kp1, des1 = orb.detectAndCompute(img1, None)  # des是描述子
    img2 = cv2.imread(image2)
    kp2, des2 = orb.detectAndCompute(img2, None)  # des是描述子
    img3 = cv2.drawKeypoints(img1, kp1, img1, color=(255, 0, 255))  # 画出特征点，并显示为红色圆圈
    img4 = cv2.drawKeypoints(img2, kp2, img2, color=(255, 0, 255))  # 画出特征点，并显示为红色圆圈
    hmerge = np.hstack((img3, img4))  # 水平拼接
    cv2.imshow("point", hmerge)  # 拼接显示为gray

    # BFMatcher解决匹配
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # 调整ratio
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)

    """
    计算仿射变换矩阵
    """
    img5 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=2)
    # img5 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    if save_path is not None:
        cv2.imwrite(f'{save_path}/BF_ORBmatch.png', img5)
    while True:
        cv2.imshow("BF_ORBmatch", img5)
        key = cv2.waitKey(0)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img1', type=str or list, help='img path.')
    parser.add_argument('--img2', type=str or list, help='img path.')
    parser.add_argument('--save_path', type=str, default=None, help='save path of result.')

    opt = parser.parse_args()

    SIFT_(opt.img1, opt.img2, opt.save_path)
    ORB_(opt.img1, opt.img2, opt.save_path)
