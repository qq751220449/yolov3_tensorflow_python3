# coding: utf-8

import numpy as np
import random
import cv2


def random_translate(img, bboxes, p=0.5):
    if random.random() < p:
        h_img, w_img, _ = img.shape    # 这里已经被裁切了,因此输出不一定是(960,1280,3)了
        # print("crop")
        # print(img.shape)
        # 得到可以包含所有bbox的最大bbox
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w_img - max_bbox[2]
        max_d_trans = h_img - max_bbox[3]

        tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
        ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

        M = np.array([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, M, (w_img, h_img))    # 随机平移

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty
    return img, bboxes          # 包括黑边


def random_crop(img, bboxes, p=0.5):
    if random.random() < p:
        # print(img.shape)
        h_img, w_img, _ = img.shape         # (960.1280,3)   # 有时会输出,有时不会输出
        # 得到可以包含所有bbox的最大bbox
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
        max_l_trans = max_bbox[0]               # xmin方向最多可以裁切的大小
        max_u_trans = max_bbox[1]               # ymin方向最多可以裁切的大小
        max_r_trans = w_img - max_bbox[2]       # xmax方向最多可以裁切的大小
        max_d_trans = h_img - max_bbox[3]       # ymax方向最多可以裁切的大小
        # 为了保证BBox框不会被裁切

        crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))       # 找到裁切的位置
        crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
        crop_xmax = max(w_img, int(max_bbox[2] + random.uniform(0, max_r_trans)))
        crop_ymax = max(h_img, int(max_bbox[3] + random.uniform(0, max_d_trans)))

        img = img[crop_ymin : crop_ymax, crop_xmin : crop_xmax]     # 根据上述找到的裁切位置进行裁切

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin           # 调整BBox的数值
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
    return img, bboxes


def random_horizontal_flip(img, bboxes, p=0.5):   # 随机水平翻转
    if random.random() < p:
        _, w_img, _ = img.shape
        img = img[:, ::-1, :]
        bboxes[:, [0, 2]] = w_img - bboxes[:, [2, 0]]
    return img, bboxes