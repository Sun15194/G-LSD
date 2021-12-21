#!/usr/bin/env python
"""Process Huang's wireframe dataset for L-CNN network
Usage:
    dataset/wireframe_line.py <src> <dst>
    dataset/wireframe_line.py (-h | --help )

Examples:
    dataset/wireframe.py /home/shu-usv005/Documents/Dataset/LETR/wireframe/wireframe_raw data/wireframe
    dataset/wireframe_line.py /home/shu-usv005/Documents/Dataset/LETR/wireframe/wireframe_raw data/wireframe

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
import seaborn as sns

sns.set()
try:
    sys.path.append("")
    sys.path.append("../data")
    from glsd.utils import parmap
except Exception:
    raise


def inrange(v, shape):
    return 0 <= v[0] < shape[0] and 0 <= v[1] < shape[1]


def to_int(x):
    return tuple(map(int, x))


def save_heatmap(prefix, image, lines):
    im_rescale = (512, 512)
    heatmap_scale = (128, 128)

    fy, fx = heatmap_scale[1] / image.shape[0], heatmap_scale[0] / image.shape[1]

    lcmap = np.zeros(heatmap_scale, dtype=np.float32)  # (128, 128)
    lcoff = np.zeros((2,) + heatmap_scale, dtype=np.float32)  # (2, 128, 128)
    lleng = np.zeros((2,) + heatmap_scale, dtype=np.float32)  # (2, 128, 128)
    angle = np.zeros(heatmap_scale, dtype=np.float32)  # (128, 128)
    xmap = np.zeros(heatmap_scale, dtype=np.float32)
    ymap = np.zeros(heatmap_scale, dtype=np.float32)

    # the coordinate of lines can not equal to 128 (less than 128).
    lines[:, :, 0] = np.clip(lines[:, :, 0] * fx, 0, heatmap_scale[0] - 1e-4)
    lines[:, :, 1] = np.clip(lines[:, :, 1] * fy, 0, heatmap_scale[1] - 1e-4)
    # lines = lines[:, :, ::-1]  # change position of x and y --> (r, c)

    # center = (v0 + v1) / 2
    # leng = np.sqrt(np.sum((v0 - v1) ** 2)) / 2
    # if leng < 3:
    #     target = [center]
    # elif leng >= 3 and leng < 10:
    #     k = [1, 2, 3]
    #     target = v0 * k / 5 + v1 * (5-k) / 5
    # else:
    #     k = range(10)
    #     target = v0 * k / 10 + v1 * (10-k) / 10
    #
    # for v in target:
    #     vint = to_int(v)

    for v0, v1 in lines:
        if v0[0] > v1[0]:
            temp = v1
            v1 = v0
            v0 =temp
        vint0, vint1 = to_int(v0), to_int(v1)
        cc, rr = skimage.draw.line(*vint0, *vint1)
        lcmap[rr, cc] = 1

        x1, y1, x2, y2 = v0[0], v0[1], v1[0], v1[1]
        v = (v0 + v1) / 2

        if (x2 - x1) <= 1e-4:
            xmap[rr, cc] = x1
            ymap[rr, cc] = cc + 0.5
        else:
            k = (y2 - y1) / (x2 - x1)
            b = y1 - k * x1
            ax = k / (k * k + 1)
            bx = 1 / (k * k + 1)
            cx = -1 * k *b / (k * k + 1)
            ay = k * k / (k * k + 1)
            by = k / (k * k + 1)
            cy = b / (k * k + 1)

            xmap[rr, cc] = ax * (cc + 0.5) + bx * (rr + 0.5) + cx
            ymap[rr, cc] = ay * (cc + 0.5) + by * (rr + 0.5) + cy

        lcoff[0][rr, cc] = xmap[rr, cc] - rr - 0.5
        lcoff[1][rr, cc] = ymap[rr, cc] - cc - 0.5

        lleng[0][rr, cc] = np.power(np.power((xmap[rr, cc] - v0[0]), 2) + np.power((ymap[rr, cc] - v0[1]), 2), 0.5)
        lleng[1][rr, cc] = np.power(np.power((xmap[rr, cc] - v1[0]), 2) + np.power((ymap[rr, cc] - v1[1]), 2), 0.5)

        # the angle under the image coordinate system (r, c)                    图像坐标系下的角度(r, c)
        # theta means the component along the c direction on the unit vector    θ表示单位向量上沿c方向的分量
        if np.sqrt(np.sum((v0 - v) ** 2)) <= 1e-4:
            continue

        # 此处的angle的实质为cosθ
        angle[rr, cc] = np.sum((v0 - v) * np.array([0., 1.])) / np.sqrt(np.sum((v0 - v) ** 2))  # theta

    image = cv2.resize(image, im_rescale)

    # fig = plt.figure()
    # sns_plot = sns.heatmap(lcmap)
    # plt.show()

    np.savez_compressed(
        f"{prefix}_line.npz",
        # aspect_ratio=image.shape[1] / image.shape[0],
        lcmap=lcmap,  # [128, 128], value=0/1
        lcoff=lcoff,  # [2, 128, 128]
        lleng=lleng,  # [2, 128, 128]
        angle=angle,  # [128, 128]
    )
    cv2.imwrite(f"{prefix}.png", image)

    # leng = np.sqrt(np.sum((v0 - v1) ** 2))
    # v = (v0 + v1) / 2
    # vint = to_int(v)
    # lcmap[vint] = 1
    # lcoff[:, vint[0], vint[1]] = v - vint - 0.5
    # leng = np.sqrt(np.sum((v0 - v1) ** 2)) / 2  # 两点之间距离计算公式
    # lleng[0][vint] = leng / 2
    # lleng[1][vint] = leng / 2

    # 令vv是v在x轴方向坐标那个端点
    # if v0[0] <= v[0]:
    #     vv = v0
    # else:
    #     vv = v1

    # the angle under the image coordinate system (r, c)                    图像坐标系下的角度(r, c)
    # theta means the component along the c direction on the unit vector    θ表示单位向量上沿c方向的分量
    # if np.sqrt(np.sum((vv - v) ** 2)) <= 1e-4:
    #     continue
    # 此处的angle的实质为cosθ
    # angle[vint] = np.sum((vv - v) * np.array([0., 1.])) / np.sqrt(np.sum((vv - v) ** 2))  # theta

    # the junction coordinate(image coordinate system) of line can be recovered by follows:
    # direction = [-sqrt(1-theta^2), theta]
    # (-sqrt(1-theta^2) means the component along the r direction on the unit vector, it always negative.)
    # center = coordinate(lcmap) + offset + 0.5
    # J = center (+-) direction * lleng  (+-) means two end points
    '''
        线的连接点坐标(图像坐标系)可以通过以下方法恢复:
        direction = [-sqrt(1-theta^2), theta]
        其中-sqrt(1-theta^2)表示单位向量上r方向的分量，它总是负的。
        center = coordinate(lcmap) + offset + 0.5
        J = center ± direction * lleng
        ±表示两个端点
    '''


'''
    在图像上逆时针旋转坐标90度

    (x, y) --> (p-q+y, p+q-x) 代表点(x,y)沿着图片中心点(p,q)顺时针旋转90度
    但是，y方向是逆的，不是上而是下。
    所以它等于逆时针旋转坐标。

    coordinares: [n, 2];    center：(p, q)旋转中心。
    坐标和中心应该遵循(x, y)顺序，而不是(h, w)。
        
    k是逆时针旋转90度的次数，旋转一次为逆时针90度，旋转三次为顺时针90度
'''
def coor_rot90(coordinates, center, k):
    # !!!rotate the coordinates 90 degree anticlockwise on image!!!!

    # (x, y) --> (p-q+y, p+q-x) means point (x,y) rotate 90 degree clockwise along center (p,q)
    # but, the y direction of coordinates is inverse, not up but down.
    # so it equals to rotate the coordinate anticlockwise.

    # coordinares: [n, 2]; center: (p, q) rotation center.
    # coordinates and center should follow the (x, y) order, not (h, w).
    new_coor = coordinates.copy()
    p, q = center
    for i in range(k):
        x = p - q + new_coor[:, 1:2]
        y = p + q - new_coor[:, 0:1]
        new_coor = np.concatenate([x, y], 1)
    return new_coor


def prepare_rotation(image, lines):
    heatmap_scale = (512, 512)

    fy, fx = heatmap_scale[1] / image.shape[0], heatmap_scale[0] / image.shape[1]

    # the coordinate of lines can not equal to 128 (less than 128).
    lines[:, :, 0] = np.clip(lines[:, :, 0] * fx, 0, heatmap_scale[0] - 1e-4)
    lines[:, :, 1] = np.clip(lines[:, :, 1] * fy, 0, heatmap_scale[1] - 1e-4)

    im = cv2.resize(image, heatmap_scale)

    return im, lines


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
            prefix = data["filename"].split(".")[0]
            lines = np.array(data["lines"]).reshape(-1, 2, 2)
            os.makedirs(os.path.join(data_output, batch), exist_ok=True)
            path = os.path.join(data_output, batch, prefix)

            lines0 = lines.copy()
            save_heatmap(f"{path}_0", im[::, ::], lines0)

            if batch != "valid":
                lines1 = lines.copy()
                lines1[:, :, 0] = im.shape[1] - lines1[:, :, 0]
                im1 = im[::, ::-1]
                save_heatmap(f"{path}_1", im1, lines1)

                lines2 = lines.copy()
                lines2[:, :, 1] = im.shape[0] - lines2[:, :, 1]
                im2 = im[::-1, ::]
                save_heatmap(f"{path}_2", im2, lines2)

                lines3 = lines.copy()
                lines3[:, :, 0] = im.shape[1] - lines3[:, :, 0]
                lines3[:, :, 1] = im.shape[0] - lines3[:, :, 1]
                im3 = im[::-1, ::-1]
                save_heatmap(f"{path}_3", im3, lines3)

                im4, lines4 = prepare_rotation(im, lines.copy())
                lines4 = coor_rot90(lines4.reshape((-1, 2)), (im4.shape[1] / 2, im4.shape[0] / 2), 1)  # rot90 on anticlockwise
                im4 = np.rot90(im4, k=1)  # rot90 on anticlockwise
                save_heatmap(f"{path}_4", im4, lines4.reshape((-1, 2, 2)))

                im5, lines5 = prepare_rotation(im, lines.copy())
                lines5 = coor_rot90(lines5.reshape((-1, 2)), (im5.shape[1] / 2, im5.shape[0] / 2), 3)  # rot90 on clockwise
                im5 = np.rot90(im5, k=-1)  # rot90 on clockwise
                save_heatmap(f"{path}_5", im5, lines5.reshape((-1, 2, 2)))

                # linesf = lines.copy()
                # linesf[:, :, 0] = im.shape[1] - linesf[:, :, 0]
                # im1 = im[::, ::-1]
                # im6, lines6 = prepare_rotation(im1, linesf.copy())
                # lines6 = coor_rot90(lines6.reshape((-1, 2)), (im6.shape[1] / 2, im6.shape[0] / 2), 1)  # rot90 on anticlockwise
                # im6 = np.rot90(im6, k=1)  # rot90 on anticlockwise
                # save_heatmap(f"{path}_6", im6, lines6.reshape((-1, 2, 2)))
                #
                # im7, lines7 = prepare_rotation(im1, linesf.copy())
                # lines7 = coor_rot90(lines7.reshape((-1, 2)), (im7.shape[1] / 2, im7.shape[0] / 2), 3)  # rot90 on clockwise
                # im7 = np.rot90(im7, k=-1)  # rot90 on clockwise
                # save_heatmap(f"{path}_7", im7, lines7.reshape((-1, 2, 2)))

                # exit()

            print("Finishing", os.path.join(data_output, batch, prefix))

        # handle(dataset[0])
        # multiprocessing the function of handle with augment 'dataset'.
        parmap(handle, dataset, 16)

if __name__ == "__main__":
    main()

