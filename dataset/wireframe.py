#!/usr/bin/env python
"""Process Huang's wireframe dataset for L-CNN network
Usage:
    dataset/wireframe.py <src> <dst>
    dataset/wireframe.py (-h | --help )

Examples:
    python dataset/wireframe.py /datadir/wireframe Good/wireframe

Arguments:
    <src>                Original Good directory of Huang's wireframe dataset
    <dst>                Directory of the output

Options:
   -h --help             Show this screen.
"""

import os
import sys
import json
from itertools import combinations

import cv2
import numpy as np
import skimage.draw
import matplotlib.pyplot as plt
from docopt import docopt
from scipy.ndimage import zoom
from wireframe_line import prepare_rotation, coor_rot90

try:
    sys.path.append("")
    sys.path.append("../data")
    from FClip.utils import parmap
except Exception:
    raise


def inrange(v, shape):
    return 0 <= v[0] < shape[0] and 0 <= v[1] < shape[1]


def to_int(x):
    return tuple(map(int, x))

'''
    方法小结：
        wireframe数据集包含一组images和lines
        将每个lines对应的两个端点坐标编号后放入junc，并用jids记录每个点在junc中对应的索引
        通过junc产生连接点热图jmap，所对应的坐标点设为1
        将float类型的junc转化为int，并计算junc中点与其像素中心的偏移量，存入joff
        用lines画出线段热图lmap，同时这些gt线段的端点在junc中的索引放入Lpos，坐标放入lpos
        junc中的点两两组合，其中不是gt线段的线段作为负样本，线段的端点在junc中的索引放入Lneg，坐标放入lneg
'''
def save_heatmap(prefix, image, lines):
    im_rescale = (512, 512)
    heatmap_scale = (128, 128)

    fy, fx = heatmap_scale[1] / image.shape[0], heatmap_scale[0] / image.shape[1]
    jmap = np.zeros((1,) + heatmap_scale, dtype=np.float32)  # (1, 128, 128)
    joff = np.zeros((1, 2) + heatmap_scale, dtype=np.float32)  # (1, 2, 128, 128)
    lmap = np.zeros(heatmap_scale, dtype=np.float32)  # (128, 128)

    # the coordinate of lines can not equal to 128 (less than 128).
    lines[:, :, 0] = np.clip(lines[:, :, 0] * fx, 0, heatmap_scale[0] - 1e-4)
    lines[:, :, 1] = np.clip(lines[:, :, 1] * fy, 0, heatmap_scale[1] - 1e-4)
    lines = lines[:, :, ::-1]  # change position of x and y

    junc = []
    jids = {}   # jids中用jun的值作为索引，jun在junc中的索引作为值

    # collecting junction endpoints (jun) and number them in dictionary (junc, jids).
    # 收集连接点端点(jun)并在字典(junc, jids)中编号。
    def jid(jun):
        '''
            eg:
            jun = np.array([2,3])           # jun=[2 3]    shape=(2,)
            jun = tuple(jun[:2])            # jun=(2, 3)
            jids[jun] = len(junc)           # jids={(2, 3): 0}
            jun = jun + (0,)                # jun=(2, 3, 0)
            junc.append(np.array(jun))      # junc=[array([2, 3, 0])]
        '''
        jun = tuple(jun[:2])
        if jun in jids:
            return jids[jun]
        jids[jun] = len(junc)
        junc.append(np.array(jun + (0,)))
        return len(junc) - 1

    lnid = []                           # nid -> Lpos
    lpos, lneg = [], []
    # drawing the heat map of line.
    # 绘制线段热图
    for v0, v1 in lines:
        lnid.append((jid(v0), jid(v1)))
        lpos.append([junc[jid(v0)], junc[jid(v1)]])

        # 将v0，v1转化为两点坐标的元组
        vint0, vint1 = to_int(v0), to_int(v1)
        jmap[0][vint0] = 1                      # 将v0对应的jmap坐标值设为1
        jmap[0][vint1] = 1                      # 将v1对应的jmap坐标值设为1
        # def line_aa(r0, c0, r1, c1)生成抗锯齿线像素坐标       img[rr, cc] = value，即该像素点的平均像素密度
        rr, cc, value = skimage.draw.line_aa(*to_int(v0), *to_int(v1))
        # 这里将lmap[rr, cc]与新的value对比，如果变大，则更新变大值
        lmap[rr, cc] = np.maximum(lmap[rr, cc], value)

    for v in junc:
        vint = to_int(v[:2])
        # 这里减0.5是因为，我们说的像素坐标是像素左上角的整形坐标，实际计算偏移量是看点到像素中心(x+0.5,y+0.5)的坐标偏移，所以要减去0.5
        # 可以看成是 joff[0, :, vint[0], vint[1]] = v[:2] - (vint + 0.5)
        joff[0, :, vint[0], vint[1]] = v[:2] - vint - 0.5

    llmap = zoom(lmap, [0.5, 0.5])  # down sampler lmap     lmap[128,128] -> llmap[64,64]
    lineset = set([frozenset(l) for l in lnid])
    # 将连接点集做排列组合两两匹配
    for i0, i1 in combinations(range(len(junc)), 2):
        if frozenset([i0, i1]) not in lineset:
            v0, v1 = junc[i0], junc[i1]
            vint0, vint1 = to_int(v0[:2] / 2), to_int(v1[:2] / 2)
            rr, cc, value = skimage.draw.line_aa(*vint0, *vint1)
            lneg.append([v0, v1, i0, i1, np.average(np.minimum(value, llmap[rr, cc]))])  # ?why minimum  hardness score?

    assert len(lneg) != 0
    lneg.sort(key=lambda l: -l[-1])

    junc = np.array(junc, dtype=np.float32)
    Lpos = np.array(lnid, dtype=np.int)
    Lneg = np.array([l[2:4] for l in lneg][:4000], dtype=np.int)
    lpos = np.array(lpos, dtype=np.float32)
    lneg = np.array([l[:2] for l in lneg[:2000]], dtype=np.float32)

    image = cv2.resize(image, im_rescale)

    '''
        For junc, lpos, and lneg that stores the junction coordinates, the last dimension is (y, x, t), 
        where t represents the type of that junction.  In the wireframe dataset, t is always zero.
        对于存储连接点坐标的junc、lpos和lneg，最后一个维度是(y, x, t)，其中t表示该结点的类型。在线框数据集中，t总是为零。
    '''
    '''
    np.savez_compressed(
        f"{prefix}_label.npz",
        aspect_ratio=image.shape[1] / image.shape[0],
        jmap=jmap,  # [J, H, W]    Junction heat map
        joff=joff,  # [J, 2, H, W] Junction offset within each pixel
        lmap=lmap,  # [H, W]       Line heat map with anti-aliasing
        junc=junc,  # [Na, 3]      Junction coordinate
        Lpos=Lpos,  # [M, 2]       Positive lines represented with junction indices
        Lneg=Lneg,  # [M, 2]       Negative lines represented with junction indices
        lpos=lpos,  # [Np, 2, 3]   Positive lines represented with junction coordinates
        lneg=lneg,  # [Nn, 2, 3]   Negative lines represented with junction coordinates
    )
    '''
    np.savez_compressed(
        f"{prefix}_label.npz",
        aspect_ratio=image.shape[1] / image.shape[0],
        jmap=jmap,  # [J, H, W]    (1,128,128)连接点热图,0/1
        joff=joff,  # [J, 2, H, W] (1,2,128,128)每个像素内的连接点偏移量,-0.5到0.5
        lmap=lmap,  # [H, W]       (128,128)抗锯齿后的线段热图，0-1
        junc=junc,  # [Na, 3]      (201,3)连接点坐标
        Lpos=Lpos,  # [M, 2]       (176,2)用连接点指数表示的正样本线段
        Lneg=Lneg,  # [M, 2]       (4000,2)用连接点指数表示的负样本线段
        lpos=lpos,  # [Np, 2, 3]   (176,2,3)用连接点坐标表示的正样本线段
        lneg=lneg,  # [Nn, 2, 3]   (2000,2,3)用连接点坐标表示的负样本线段
    )


def main():
    args = docopt(__doc__)
    data_root = args["<src>"]
    data_output = args["<dst>"]

    os.makedirs(data_output, exist_ok=True)
    for batch in ["train", "valid"]:  # "train", "valid"
        anno_file = os.path.join(data_root, f"{batch}.json")

        with open(anno_file, "r") as f:
            dataset = json.load(f)

        def handle(data):
            im = cv2.imread(os.path.join(data_root, "images", data["filename"]))
            prefix = data["filename"].split(".")[0]                         # prefix：数据名
            lines = np.array(data["lines"]).reshape(-1, 2, 2)               # lines：[n,2,2]
            os.makedirs(os.path.join(data_output, batch), exist_ok=True)
            path = os.path.join(data_output, batch, prefix)

            lines0 = lines.copy()
            save_heatmap(f"{path}_0", im[::, ::], lines0)

            # 如果是训练集则进行数据增广
            if batch != "valid":
                # 水平翻转数据增广
                lines1 = lines.copy()
                lines1[:, :, 0] = im.shape[1] - lines1[:, :, 0]
                im1 = im[::, ::-1]
                save_heatmap(f"{path}_1", im1, lines1)

                # 垂直翻转数据增广
                lines2 = lines.copy()
                lines2[:, :, 1] = im.shape[0] - lines2[:, :, 1]
                im2 = im[::-1, ::]
                save_heatmap(f"{path}_2", im2, lines2)

                # 水平+垂直翻转数据增广
                lines3 = lines.copy()
                lines3[:, :, 0] = im.shape[1] - lines3[:, :, 0]
                lines3[:, :, 1] = im.shape[0] - lines3[:, :, 1]
                im3 = im[::-1, ::-1]
                save_heatmap(f"{path}_3", im3, lines3)

                # 逆时针旋转数据增广
                im4, lines4 = prepare_rotation(im, lines.copy())
                lines4 = coor_rot90(lines4.reshape((-1, 2)), (im4.shape[1] / 2, im4.shape[0] / 2),
                                    1)  # rot90 on anticlockwise
                im4 = np.rot90(im4, k=1)  # rot90 on anticlockwise
                save_heatmap(f"{path}_4", im4, lines4.reshape((-1, 2, 2)))

                # 顺时针旋转数据增广
                im5, lines5 = prepare_rotation(im, lines.copy())
                lines5 = coor_rot90(lines5.reshape((-1, 2)), (im5.shape[1] / 2, im5.shape[0] / 2),
                                    3)  # rot90 on clockwise
                im5 = np.rot90(im5, k=-1)  # rot90 on clockwise
                save_heatmap(f"{path}_5", im5, lines5.reshape((-1, 2, 2)))

            print("Finishing", os.path.join(data_output, batch, prefix))

        # multiprocessing the function of handle with augment 'dataset'.
        # 对'dataset'函数进行多线程处理。
        parmap(handle, dataset, 16)


if __name__ == "__main__":
    main()

