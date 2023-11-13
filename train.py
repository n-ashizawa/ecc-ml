import argparse
import datetime
import os

import numpy as np
import torch

from utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--date", type=str, default="20230725-0815")
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--arch", type=str, default="resnet18")
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--pretrained", type=int, default=0)
parser.add_argument("--label-flipping", type=float, default=0)
parser.add_argument("--over-fitting", action="store_true", default=False)
args = parser.parse_args()


def main(args):
    torch_fix_seed(args.seed)
    device = torch.device(args.device)

    save_dir = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
    os.makedirs(f"./model/{save_dir}", exist_ok=True)
    os.makedirs(f"./loss/{save_dir}", exist_ok=True)

    if args.pretrained > 0:
        model = load_model(args, f"./model/{args.date}/{args.pretrained}", device)
    else:
        model = make_model(args, device)

    optimizer = make_optim(args, model, pretrained=False)
    #scheduler =lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    train_loader, test_loader = prepare_dataset(args)

    train_losses = []
    test_losses = []

    for epoch in range(1, args.epoch+1):
        acc, loss = train(args, model, train_loader, optimizer, device)
        train_losses.append(loss)
        #scheduler.step()
        print(f"EPOCH: {epoch}\n"
                f"TRAIN ACC: {acc:.6f}\t"
                f"TRAIN LOSS: {loss:.6f}")
        # test acc
        acc, loss = test(model, test_loader, device)
        test_losses.append(loss)
        print(f"VAL ACC: {acc:.6f}\t"
                f"VAL LOSS: {loss:.6f}")

        # モデルの保存
        if args.pretrained > 0:
            save_model(model, f"./model/{save_dir}/{args.pretrained}plus{epoch}")
        else:
            save_model(model, f"./model/{save_dir}/{epoch}")

    del model
    torch.cuda.empty_cache()

    plot_linear(train_losses, f"./loss/{save_dir}/train")
    plot_linear(test_losses, f"./loss/{save_dir}/test")

if __name__ == "__main__":
    main(args)

