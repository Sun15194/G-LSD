import numpy as np


class WireframeHuangKun():
    @staticmethod
    def glsd_parsing(npz_name, ang_type="radian"):
        with np.load(npz_name) as npz:
            lcmap_ = npz["lcmap"]
            lcoff_ = npz["lcoff"]
            lleng_ = np.clip(npz["lleng"], 0, 128 - 1e-4)
            angle_ = np.clip(npz["angle"], -1 + 1e-4, 1 - 1e-4)

            lcmap = lcmap_
            lcoff = lcoff_
            lleng = lleng_ / 128                             # 归一化，使lleng的范围在0-1之间
            if ang_type == "cosine":                        # cosine：用cosθ表示角度
                angle = (angle_ + 1) * lcmap_ / 2
            elif ang_type == "radian":
                angle = lcmap * np.arccos(angle_) / np.pi   # radian：用弧度表示角度，值为0-2π
            else:
                raise NotImplementedError

            return lcmap, lcoff, lleng, angle
