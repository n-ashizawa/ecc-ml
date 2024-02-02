import os
import csv

import numpy as np
import torch

from network import *
from utils import *
from arguments import get_args
from logger import get_logger, logging_args


def summarize_loss(args, seeds, save_dir):
    if args.over_fitting:
        mode = "over-fitting"
    elif args.label_flipping > 0:
        mode = "label-flipping"
    elif args.label_flipping == 0:
        mode = "normal"
    else:
        raise NotImplementedError

    with open(f"{save_dir}/{mode}{args.pretrained}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['', "seed"] + [i for i in range(1, args.epoch+1)])
        writer.writerow([])

        test_losses = [["test losses"]]
        
        for seed in seeds:
            load_dir = f"{save_dir}/{seed}/{mode}/{args.pretrained}"
            
            with open(f"{load_dir}/loss.csv", "r") as loss_file:
                print(f"opend {load_dir}/loss.csv")
                lines = loss_file.readlines()[1]
                test_losses.append(['', seed] + lines.split(",")[1:])
        
        writer.writerows(test_losses)
            

def main():
    args = get_args()
    torch_fix_seed(args.seed)
    device = torch.device(args.device)

    seeds = [1, 2, 3, 4]
    
    if args.over_fitting:
        mode = "over-fitting"
    elif args.label_flipping > 0:
        mode = "label-flipping"
    elif args.label_flipping == 0:
        mode = "normal"
    else:
        raise NotImplementedError

    for seed in seeds:
        setattr(args, "seed", seed)
        save_dir = f"./train/{args.dataset}/{args.arch}/{args.epoch}/{args.lr}/{args.seed}/{mode}/{args.pretrained}"
        load_model_dir = f"{save_dir}/model"    

        save_data_file = f"{save_dir}/loss.csv"
        if not os.path.isfile(save_data_file):
            _, test_loader = prepare_dataset(args)
            test_losses = []
        
            for epoch in range(1, args.epoch+1):
                model = load_model(args, f"{load_model_dir}/{epoch}", device)
                _, loss = test(model, test_loader, device)
                test_losses.append(loss)
                
                del model
                torch.cuda.empty_cache()

            with open(save_data_file, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["train"])
                writer.writerow(["test"] + test_losses)
        
    save_dir = f"./train/{args.dataset}/{args.arch}/{args.epoch}/{args.lr}"
    #save_data_file = f"{save_dir}/{mode}{args.pretrained}.csv"
    summarize_loss(args, seeds, save_dir)

    exit()


if __name__ == "__main__":
    main()

