import numpy as np
import cv2


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

        lcmap, lcoff, lleng, angle = ResizeResolution.resolution_fclip(lpos, resolu, ang_type)

        return image_, lcmap, lcoff, lleng, angle

    @staticmethod
    def resolution_fclip(lpos, resolu, ang_type="radian"):
        heatmap_scale = (resolu, resolu)
        scale = resolu / 128

        lines = lpos * scale

        lcmap = np.zeros(heatmap_scale, dtype=np.float32)  # (resolu, resolu)
        lcoff = np.zeros((2,) + heatmap_scale, dtype=np.float32)  # (2, resolu, resolu)
        lleng = np.zeros(heatmap_scale, dtype=np.float32)  # (resolu, resolu)
        angle = np.zeros(heatmap_scale, dtype=np.float32)  # (resolu, resolu)

        lines[:, :, 0] = np.clip(lines[:, :, 0], 0, heatmap_scale[0] - 1e-4)
        lines[:, :, 1] = np.clip(lines[:, :, 1], 0, heatmap_scale[1] - 1e-4)

        for v0, v1 in lines:
            v = (v0 + v1) / 2
            vint = tuple(map(int, v))
            lcmap[vint] = 1
            lcoff[:, vint[0], vint[1]] = v - vint - 0.5
            lleng[vint] = np.sqrt(np.sum((v0 - v1) ** 2)) / 2  # L

            if v0[0] <= v[0]:
                vv = v0
            else:
                vv = v1
            # the angle under the image coordinate system (r, c)
            # theta means the component along the c direction on the unit vector
            if np.sqrt(np.sum((vv - v) ** 2)) <= 1e-4:
                continue
            angle[vint] = np.sum((vv - v) * np.array([0., 1.])) / np.sqrt(np.sum((vv - v) ** 2))  # theta

        lleng_ = np.clip(lleng, 0, 64 * scale - 1e-4) / (64 * scale)

        if ang_type == "cosine":
            angle = (angle + 1) * lcmap / 2
        elif ang_type == "radian":
            angle = lcmap * np.arccos(angle) / np.pi
        else:
            raise NotImplementedError

        return lcmap, lcoff, lleng_, angle
