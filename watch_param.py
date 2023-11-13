import argparse
import os
import matplotlib.pyplot as plt

import torch

from utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--date", type=str, default="20230725-0815")
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--arch", type=str, default="resnet18")
parser.add_argument("--dataset", type=str, default="cifar10")
args = parser.parse_args()


def plot_param(name_param, save_dir, file_index):
    for name, param in name_param.items():
        plt.plot(param, label=name)
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig(f"{save_dir}/{file_index}.png")
    plt.clf()


def minmax_param_resnet(args, device):
    p_max_conv = {}
    p_min_conv = {}
    p_dif_conv = {}
    p_max_bn_weight = {}
    p_min_bn_weight = {}
    p_dif_bn_weight = {}
    p_max_bn_bias = {}
    p_min_bn_bias = {}
    p_dif_bn_bias = {}
    for epoch in range(1, args.epoch+1):
        model = load_model(args, f"./model/{args.date}/{epoch}", device)
        for name, param in model.named_parameters():
            if "bn" in name or "shortcut.1" in name:
                if "weight" in name:
                    if name not in p_max_bn_weight:
                        p_max_bn_weight.update({name:list()})
                        p_min_bn_weight.update({name:list()})
                        p_dif_bn_weight.update({name:list()})
                    p_max_bn_weight[name].append(param.cpu().detach().numpy().max())
                    p_min_bn_weight[name].append(param.cpu().detach().numpy().min())
                    p_dif_bn_weight[name].append(param.cpu().detach().numpy().max()-param.cpu().detach().numpy().min())
                else:
                    if name not in p_max_bn_bias:
                        p_max_bn_bias.update({name:list()})
                        p_min_bn_bias.update({name:list()})
                        p_dif_bn_bias.update({name:list()})
                    p_max_bn_bias[name].append(param.cpu().detach().numpy().max())
                    p_min_bn_bias[name].append(param.cpu().detach().numpy().min())
                    p_dif_bn_bias[name].append(param.cpu().detach().numpy().max()-param.cpu().detach().numpy().min())
            else:
                if name not in p_max_conv:
                    p_max_conv.update({name:list()})
                    p_min_conv.update({name:list()})
                    p_dif_conv.update({name:list()})
                p_max_conv[name].append(param.cpu().detach().numpy().max())
                p_min_conv[name].append(param.cpu().detach().numpy().min())
                p_dif_conv[name].append(param.cpu().detach().numpy().max()-param.cpu().detach().numpy().min())
        del model

    save_dir = f"./param/{args.arch}"
    os.makedirs(save_dir, exist_ok=True)
    plot_param(p_max_bn_weight, save_dir, "max_bn_weight")
    plot_param(p_min_bn_weight, save_dir, "min_bn_weight")
    plot_param(p_dif_bn_bias, save_dir, "min_bn_bias")
    plot_param(p_max_bn_bias, save_dir, "max_bn_bias")
    plot_param(p_min_bn_bias, save_dir, "min_bn_bias")
    plot_param(p_max_conv, save_dir, "max_conv")
    plot_param(p_min_conv, save_dir, "min_conv")


def minmax_param_vgg(args, device):
    p_max = {}
    p_min = {}
    for epoch in range(1, args.epoch+1):
        model = load_model(args, f"./model/{args.date}/{epoch}", device)
        for name, param in model.named_parameters():
            if "weight" in name:
                if name not in p_max:
                    p_max.update({name:list()})
                    p_min.update({name:list()})
                p_max[name].append(param.cpu().detach().numpy().max())
                p_min[name].append(param.cpu().detach().numpy().min())
        del model

    save_dir = f"./param/{args.arch}"
    os.makedirs(save_dir, exist_ok=True)
    plot_param(p_max, save_dir, "max")
    plot_param(p_min, save_dir, "min")


def minmax_param_mobilenet(args, device):
    p_max = {}
    p_min = {}
    for epoch in range(1, args.epoch+1):
        model = load_model(args, f"./model/{args.date}/{epoch}", device)
        for name, param in model.named_parameters():
            if "weight" in name:
                if name not in p_max:
                    p_max.update({name:list()})
                    p_min.update({name:list()})
                p_max[name].append(param.cpu().detach().numpy().max())
                p_min[name].append(param.cpu().detach().numpy().min())
        del model

    save_dir = f"./param/{args.arch}"
    os.makedirs(save_dir, exist_ok=True)
    plot_param(p_max, save_dir, "max")
    plot_param(p_min, save_dir, "min")


def main(args):
    torch_fix_seed(args.seed)
    device = torch.device("cpu")
    if args.arch == "resnet18" or args.arch == "resnet152":
        if args.arch == "resnet18":
            args.date = "20231006-0945"
        else:
            args.date = "20231031-1434"
        minmax_param_resnet(args, device)
    elif "VGG" in args.arch:
        if args.arch == "VGG11":
            args.date = "20231031-1345"
        minmax_param_vgg(args, device)
    elif args.arch == "mobilenet":
        args.date = "20231031-1346"
        minmax_param_mobilenet(args, device)

    exit()


if __name__ == "__main__":
    main(args)

