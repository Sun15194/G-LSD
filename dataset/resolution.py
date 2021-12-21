import numpy as np
import cv2
import skimage.draw

class ResizeResolution():

    @staticmethod
    def resize(lpos, image, resolu, ang_type="radian"):
        """resize the heatmap"""

        if resolu < 128:
            image_ = cv2.resize(image, (resolu * 4, resolu * 4))
        elif resolu == 128:
            image_ = image
        else:
            raise ValueError("not support!")

        lcmap, lcoff, lleng, angle = ResizeResolution.resolution_glsd(lpos, resolu, ang_type)

        return image_, lcmap, lcoff, lleng, angle

    @staticmethod
    def resolution_glsd(lpos, resolu, ang_type="radian"):
        heatmap_scale = (resolu, resolu)
        scale = resolu / 128

        lines = lpos * scale

        lcmap = np.zeros(heatmap_scale, dtype=np.float32)  # (resolu, resolu)
        lcoff = np.zeros((2,) + heatmap_scale, dtype=np.float32)  # (2, resolu, resolu)
        lleng = np.zeros((2,) + heatmap_scale, dtype=np.float32)  # (2, resolu, resolu)
        angle = np.zeros(heatmap_scale, dtype=np.float32)  # (resolu, resolu)
        xmap = np.zeros(heatmap_scale, dtype=np.float32)
        ymap = np.zeros(heatmap_scale, dtype=np.float32)

        lines[:, :, 0] = np.clip(lines[:, :, 0], 0, heatmap_scale[0] - 1e-4)
        lines[:, :, 1] = np.clip(lines[:, :, 1], 0, heatmap_scale[1] - 1e-4)

        for v0, v1 in lines:
            if v0[0] > v1[0]:
                temp = v1
                v1 = v0
                v0 = temp
            vint0, vint1 = tuple(map(int, v0)), tuple(map(int, v1))
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
                cx = -1 * k * b / (k * k + 1)
                ay = k * k / (k * k + 1)
                by = k / (k * k + 1)
                cy = b / (k * k + 1)

                xmap[rr, cc] = ax * (cc + 0.5) + bx * (rr + 0.5) + cx
                ymap[rr, cc] = ay * (cc + 0.5) + by * (rr + 0.5) + cy

            lcoff[0][rr, cc] = xmap[rr, cc] - rr - 0.5
            lcoff[1][rr, cc] = ymap[rr, cc] - cc - 0.5

            lleng[0][rr, cc] = np.power(np.power((xmap[rr, cc] - v0[0]), 2) + np.power((ymap[rr, cc] - v0[1]), 2), 0.5)
            lleng[1][rr, cc] = np.power(np.power((xmap[rr, cc] - v1[0]), 2) + np.power((ymap[rr, cc] - v1[1]), 2), 0.5)

            if np.sqrt(np.sum((v0 - v) ** 2)) <= 1e-4:
                continue

            angle[rr, cc] = np.sum((v0 - v) * np.array([0., 1.])) / np.sqrt(np.sum((v0 - v) ** 2))  # theta

        lleng_ = np.clip(lleng, 0, 128 * scale - 1e-4) / (128 * scale)

        if ang_type == "cosine":
            angle = (angle + 1) * lcmap / 2
        elif ang_type == "radian":
            angle = lcmap * np.arccos(angle) / np.pi
        else:
            raise NotImplementedError

        return lcmap, lcoff, lleng_, angle
