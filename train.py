#!/usr/bin/env python3
"""Train L-CNN
Usage:
    train.py [options] <yaml-config>
    train.py (-h | --help )

Arguments:
   <yaml-config>                   Path to the yaml hyper-parameter file

Options:
   -h --help                       Show this screen.
   -d --devices <devices>          Comma seperated GPU devices [default: 0]
   -i --identifier <identifier>    Folder identifier [default: default-lr]
"""

import os
import glob
import pprint
import random
import shutil
import os.path as osp
import datetime

import numpy as np
import torch
from docopt import docopt

import glsd
from glsd.config import C, M
from glsd.datasets import collate
from glsd.datasets import LineDataset as WireframeDataset

from glsd.models.stage_1 import GLSD
from glsd.models import MultitaskHead, hg
from glsd.lr_schedulers import init_lr_scheduler
from glsd.trainer import Trainer


# 生成输出文件路径
def get_outdir(identifier):
    # load config
    name = str(datetime.datetime.now().strftime("%y%m%d-%H%M%S"))
    name += "-%s" % identifier
    # 文件路径拼接
    outdir = osp.join(osp.expanduser(C.io.logdir), name)
    # 创建文件夹
    if not osp.exists(outdir):
        os.makedirs(outdir)
    # C中的属性对应base.yaml和fclip_XX。yaml文件中的项
    C.io.resume_from = outdir
    C.to_yaml(osp.join(outdir, "config.yaml"))
    return outdir


# 构建网络模型框架
def build_model():
    # 选择主干网络
    if M.backbone == "stacked_hourglass":
        model = hg(
            depth=M.depth,
            head=lambda c_in, c_out: MultitaskHead(c_in, c_out),
            num_stacks=M.num_stacks,
            num_blocks=M.num_blocks,
            num_classes=sum(sum(MultitaskHead._get_head_size(), [])),
        )
    else:
        raise NotImplementedError

    model = GLSD(model).cuda()

    # 在yaml文件里看是否有初始化的权重文件
    if C.io.model_initialize_file:
        # 加载模型用来初始化的checkpoint文件
        checkpoint = torch.load(C.io.model_initialize_file)
        # 加载optimizer的状态
        model.load_state_dict(checkpoint["model_state_dict"])
        del checkpoint
        print('=> loading model from {}'.format(C.io.model_initialize_file))

    print("Finished constructing model!")
    return model


def main():
    args = docopt(__doc__)
    config_file = args["<yaml-config>"]
    # 此处将base.yaml和fclip.XX.yaml两个配置拼接到了一起，放入C中，并创建快捷方式M对应C.model
    C.update(C.from_yaml(filename="config/base.yaml"))
    C.update(C.from_yaml(filename=config_file))
    M.update(C.model)
    pprint.pprint(C, indent=4)
    resume_from = C.io.resume_from

    # WARNING: L-CNN is still not deterministic
    # 固定随机种子
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # 设置运行设备：GPU/CPU
    device_name = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    else:
        print("CUDA is not available")
    device = torch.device(device_name)

    # 1. dataset
    datadir = C.io.datadir
    kwargs = {
        "collate_fn": collate,
        "num_workers": C.io.num_workers,
        "pin_memory": True,
    }
    dataname = C.io.dataname
    # shuffle=True代表随机读入数据集
    train_loader = torch.utils.data.DataLoader(
        # from FClip.datasets import LineDataset as WireframeDataset
        WireframeDataset(datadir, split="train", dataset=dataname), batch_size=M.batch_size, shuffle=True, drop_last=True, **kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        WireframeDataset(datadir, split="valid", dataset=dataname), batch_size=M.eval_batch_size, **kwargs
    )
    # epoch_size = 5000
    # len(val_loader) = 462 / 2 = 231
    epoch_size = len(train_loader)

    # 2. model
    model = build_model()

    # 3. optimizer
    if C.optim.name == "Adam":
        optim = torch.optim.Adam(
            model.parameters(),
            lr=C.optim.lr,
            weight_decay=C.optim.weight_decay,
            amsgrad=C.optim.amsgrad,
        )
    else:
        raise NotImplementedError

    # 设置输出文件夹名字为 “系统时间+网络类型”
    outdir = get_outdir(args["--identifier"])
    print("outdir:", outdir)

    iteration = 0
    epoch = 0
    best_mean_loss = 1e1000

    # 如果是继续训练，则读取相关已训练数据
    if resume_from:
        ckpt_pth = osp.join(resume_from, "checkpoint_lastest.pth.tar")
        checkpoint = torch.load(ckpt_pth)
        iteration = checkpoint["iteration"]
        epoch = iteration // epoch_size
        best_mean_loss = checkpoint["best_mean_loss"]
        print(f"loading {epoch}-th ckpt: {ckpt_pth}")

        model.load_state_dict(checkpoint["model_state_dict"])
        optim.load_state_dict(checkpoint["optim_state_dict"])

        lr_scheduler = init_lr_scheduler(
            optim, C.optim.lr_scheduler,
            stepsize=C.optim.lr_decay_epoch,
            max_epoch=C.optim.max_epoch,
            last_epoch=iteration // epoch_size
        )
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        del checkpoint

    else:
        lr_scheduler = init_lr_scheduler(
            optim,
            C.optim.lr_scheduler,
            stepsize=C.optim.lr_decay_epoch,
            max_epoch=C.optim.max_epoch
        )

    trainer = Trainer(
        device=device,
        model=model,
        optimizer=optim,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        out=outdir,
        iteration=iteration,
        epoch=epoch,
        bml=best_mean_loss,
    )

    try:
        trainer.train()

    except BaseException:
        # 如果outdir/viz/*下文件数 <= 1，则递归删除输出文件夹内所有内容
        if len(glob.glob(f"{outdir}/viz/*")) <= 1:
            shutil.rmtree(outdir)
        raise


if __name__ == "__main__":
    main()
