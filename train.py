'''
MIT License
Copyright (c) 2023 fseclab-osaka
'''

import os
import csv

import numpy as np
import torch

from network import *
from utils import *
from arguments import get_args
from logger import get_logger, logging_args

from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics,\
    loss_metrics, timing_metrics, cpu_usage_metrics, StreamConfusionMatrix,\
    disk_usage_metrics, gpu_usage_metrics
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training import Naive



def main():
    args = get_args()
    torch_fix_seed(args.seed)
    device = torch.device(args.device)

    if args.cl:
        args.arch = "mlp"
        args.dataset = "splitmnist"
        args.epoch = 1

    if args.over_fitting:
        mode = "over-fitting"
    elif args.label_flipping > 0:
        mode = "label-flipping"
    elif args.label_flipping == 0:
        mode = "normal"
    else:
        raise NotImplementedError
    
    save_dir = f"./train/{args.dataset}/{args.arch}/{args.epoch}/{args.lr}/{mode}{args.pretrained}/{args.seed}"
    os.makedirs(save_dir, exist_ok=True)
    save_model_dir = f"{save_dir}/model"
    os.makedirs(save_model_dir, exist_ok=True)

    save_data_file = f"{save_dir}/loss.csv"
    if not os.path.isfile(save_data_file):
        logging = get_logger(f"{save_dir}/train.log")
        logging_args(args, logging)

        train_loader, test_loader, n_classes = prepare_dataset(args, save_dir)
        train_losses = []
        test_losses = []
        
        if args.pretrained == 0:
            model = make_model(args, device, n_classes=n_classes)
        elif args.pretrained > 0:
            for epoch in range(1, args.pretrained+1):
                model = load_model(args, f"./train/{args.dataset}/{args.arch}/{args.epoch}/{args.lr}/{args.seed}/normal/0/model/{epoch}", device)
                # test acc
                acc, loss = test(model, test_loader, device)
                test_losses.append(loss)
                logging.info(f"INITIAL VAL ACC: {acc:.6f}\t"
                    f"VAL LOSS: {loss:.6f}")
                # save model
                save_model(model, f"{save_model_dir}/{epoch}")
                del model
            model = load_model(args, f"./train/{args.dataset}/{args.arch}/{args.epoch}/{args.lr}/{args.seed}/normal/0/model/{args.pretrained}", device)
        else:
            raise NotImplementedError

        optimizer = make_optim(args, model, pretrained=False)
        #scheduler =lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        
        for epoch in range(args.pretrained+1, args.epoch+1):
            acc, loss = train(model, train_loader, optimizer, device)
            train_losses.append(loss)
            #scheduler.step()
            logging.info(f"EPOCH: {epoch}\n"
                f"TRAIN ACC: {acc:.6f}\t"
                f"TRAIN LOSS: {loss:.6f}")
            # test acc
            acc, loss = test(model, test_loader, device)
            test_losses.append(loss)
            logging.info(f"VAL ACC: {acc:.6f}\t"
                f"VAL LOSS: {loss:.6f}")

            # save model
            save_model(model, f"{save_model_dir}/{epoch}")

        del model
        torch.cuda.empty_cache()

        plot_linear(train_losses, f"{save_dir}/train")
        plot_linear(test_losses, f"{save_dir}/test")
        
        with open(save_data_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow([''] + [i for i in range(1, args.epoch+1)])
            writer.writerow(["train"] + ['']*(len(test_losses)-len(train_losses)) + train_losses)
            writer.writerow(["test"] + test_losses)
        
    exit()


if __name__ == "__main__":
    main()

