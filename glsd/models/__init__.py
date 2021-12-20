from .hourglass_pose import hg
from glsd.config import M
import torch.nn as nn
import torch
# from torchvision.ops.deform_conv import DeformConv2d


class MultitaskHead(nn.Module):
    def __init__(self, input_channels, num_class):
        super(MultitaskHead, self).__init__()

        m = int(input_channels / 4)
        heads = []
        heads_size = sum(self._get_head_size(), [])
        heads_net = M.head_net
        for k, (output_channels, net) in enumerate(zip(heads_size, heads_net)):
            if net == "raw":
                heads.append(
                    nn.Sequential(
                        nn.Conv2d(input_channels, m, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(m, output_channels, kernel_size=1),
                    )
                )
                print(f"{k}-th head, head type {net}, head output {output_channels}")
            else:
                raise NotImplementedError
        self.heads = nn.ModuleList(heads)
        assert num_class == sum(sum(self._get_head_size(), []))

    @staticmethod
    def _get_head_size():

        M_dic = M.to_dict()
        head_size = []
        for h in M_dic['head']['order']:
            head_size.append([M_dic['head'][h]['head_size']])

        return head_size

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=1)

