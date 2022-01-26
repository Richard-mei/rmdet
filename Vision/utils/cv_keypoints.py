import cv2
import numpy
import matplotlib.pyplot as plt
import argparse


def akaze_(img, save_path=None):
    img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
    # cv2.imshow('Input Image', img)
    # cv2.waitKey(0)

    # 检测
    akaze = cv2.AKAZE_create()
    keypoints = akaze.detect(img, None)

    # 显示
    # 必须要先初始化img2
    img2 = img.copy()
    img2 = cv2.drawKeypoints(img, keypoints, img2, color=(255, 0, 0))
    # cv2.imshow('Detected AKAZE keypoints', img2)
    # cv2.waitKey(0)
    while True:
        plt.subplot(211)
        plt.imshow(img)
        plt.subplot(212)
        plt.imshow(img2)
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()

        key = cv2.waitKey(0)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
        break


def brisk_(img, save_path=None):
    img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
    # cv2.imshow('Input Image', img)

    brisk = cv2.BRISK_create()
    keypoints = brisk.detect(img, None)

    # 必须要先初始化img2
    img2 = img.copy()
    img2 = cv2.drawKeypoints(img, keypoints, img2, color=(255, 0, 0))
    # cv2.imshow('Detected BRISK keypoints', img2)
    while True:
        plt.subplot(211)
        plt.imshow(img)
        plt.subplot(212)
        plt.imshow(img2)
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()

        key = cv2.waitKey(0)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
        break


def kaze_(img, save_path=None):
    img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
    # cv2.imshow('Input Image', img)
    # cv2.waitKey(0)

    # 检测
    kaze = cv2.KAZE_create()
    keypoints = kaze.detect(img, None)

    # 显示
    # 必须要先初始化img2
    img2 = img.copy()
    img2 = cv2.drawKeypoints(img, keypoints, img2, color=(255, 0, 0))
    # cv2.imshow('Detected KAZE keypoints', img2)
    # cv2.waitKey(0)
    while True:
        plt.subplot(211)
        plt.imshow(img)
        plt.subplot(212)
        plt.imshow(img2)
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()

        key = cv2.waitKey(0)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
        break


def orb_(img, save_path=None):
    img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
    # cv2.imshow('Input Image', img)
    # cv2.waitKey(0)

    # 检测
    orb = cv2.ORB_create()
    keypoints = orb.detect(img, None)

    # 显示
    # 必须要先初始化img2
    img2 = img.copy()
    img2 = cv2.drawKeypoints(img, keypoints, img2, color=(255, 0, 0))
    # cv2.imshow('Detected ORB keypoints', img2)
    # cv2.waitKey(0)
    while True:
        plt.subplot(211)
        plt.imshow(img)
        plt.subplot(212)
        plt.imshow(img2)
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()

        key = cv2.waitKey(0)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
        break


def fast_(img, save_path=None):
    img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
    # cv2.imshow('Input Image', img)
    # cv2.waitKey(0)

    # 2018-03-17 Amusi: OpenCV3.x FeatureDetector写法有变化
    # OpenCV2.x
    # fast = cv2.FastFeatureDetector()
    # keypoints = fast.detect(img, None)

    # OpenCV3.x
    # 注意有_create()后缀
    fast = cv2.FastFeatureDetector_create()
    keypoints = fast.detect(img, None)

    # 必须要先初始化img2
    img2 = img.copy()
    img2 = cv2.drawKeypoints(img, keypoints, img2, color=(0, 255, 0))
    # cv2.imshow('Detected FAST keypoints', img2)
    # cv2.waitKey(0)
    while True:
        plt.subplot(211)
        plt.imshow(img)
        plt.subplot(212)
        plt.imshow(img2)
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
        break


def sift_(img, save_path=None):
    img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
    # cv2.imshow('Input Image', img)
    # cv2.waitKey(0)

    # 2018-03-17 Amusi: OpenCV3.x FeatureDetector写法有变化
    # OpenCV2.x
    # fast = cv2.FastFeatureDetector()
    # keypoints = fast.detect(img, None)

    # OpenCV3.x
    # 注意有_create()后缀
    sift = cv2.SIFT_create()
    keypoints = sift.detect(img, None)

    # 必须要先初始化img2
    img2 = img.copy()
    img2 = cv2.drawKeypoints(img, keypoints, img2, color=(0, 255, 0))
    # cv2.imshow('Detected FAST keypoints', img2)
    # cv2.waitKey(0)
    while True:
        plt.subplot(211)
        plt.imshow(img)
        plt.subplot(212)
        plt.imshow(img2)
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
        break


def all_k(image, save_path=None):

    for i, img in enumerate(image if isinstance(image, list) else [image]):
        img = cv2.imread(img)
        # img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        # cv2.imshow('Input Image', img)

        brisk = cv2.BRISK_create()
        kpt_brisk = brisk.detect(img, None)

        akaze = cv2.AKAZE_create()
        kpt_akaze = akaze.detect(img, None)

        fast = cv2.FastFeatureDetector_create()
        kpt_fast = fast.detect(img, None)

        orb = cv2.ORB_create()
        kpt_orb = orb.detect(img, None)

        kaze = cv2.KAZE_create()
        kpt_kaze = kaze.detect(img, None)

        sift = cv2.SIFT_create()
        kpt_sift = sift.detect(img, None)

        # sift = cv2.xfeatures2d.SIFT_create()
        # kpt_sift, _ = sift.detectAndCompute(img, None)

        # 必须要先初始化img2
        img1 = img.copy()
        img2 = img.copy()
        img3 = img.copy()
        img4 = img.copy()
        img5 = img.copy()
        img6 = img.copy()
        img1 = cv2.drawKeypoints(img, kpt_brisk, img1, color=(0, 0, 255))
        img2 = cv2.drawKeypoints(img, kpt_akaze, img2, color=(0, 0, 255))
        img3 = cv2.drawKeypoints(img, kpt_fast, img3, color=(0, 0, 255))
        img4 = cv2.drawKeypoints(img, kpt_orb, img4, color=(0, 0, 255))
        img5 = cv2.drawKeypoints(img, kpt_kaze, img5, color=(0, 0, 255))
        img6= cv2.drawKeypoints(img, kpt_sift, img6, color=(0, 0, 255))
        cv2.imwrite(f'kpt_brisk_{i}.png', img1)
        cv2.imwrite(f'kpt_akaze_{i}.png', img2)
        cv2.imwrite(f'kpt_fast_{i}.png', img3)
        cv2.imwrite(f'kpt_orb_{i}.png', img4)
        cv2.imwrite(f'kpt_kaze_{i}.png', img5)
        cv2.imwrite(f'kpt_sift_{i}.png', img6)
        # while True:
        #     plt.figure(figsize=(6, 8))
        #     plt.subplot(161)
        #     plt.imshow(img)
        #     # plt.imsave('ori.png', img)
        #     plt.subplot(162)
        #     plt.xlabel('brisk')
        #     plt.imshow(img1)
        #     plt.subplot(163)
        #     plt.xlabel('akaze')
        #     plt.imshow(img2)
        #     plt.subplot(164)
        #     plt.xlabel('fast')
        #     plt.imshow(img3)
        #     plt.subplot(165)
        #     plt.xlabel('orb')
        #     plt.imshow(img4)
        #     plt.subplot(166)
        #     plt.xlabel('kaze')
        #     plt.imshow(img5)
        #     if save_path is not None:
        #         plt.savefig(save_path)
        #     plt.show()
        #     key = cv2.waitKey(0)
        #     if key & 0xFF == ord('q') or key == 27:
        #         cv2.destroyAllWindows()
        #     break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, nargs='+', help='img path.')
    parser.add_argument('--save_path', type=str, default=None, help='save path of result.')

    opt = parser.parse_args()

    # akaze_(opt.img, opt.save_path)
    # brisk_(opt.img, opt.save_path)
    # kaze_(opt.img, opt.save_path)
    # orb_(opt.img, opt.save_path)
    # fast_(opt.img, opt.save_path)
    # sift_(opt.img, opt.save_path)
    all_k(opt.img, opt.save_path)
