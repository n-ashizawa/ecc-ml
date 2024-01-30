import os

import numpy as np
import torch

from network import *
from utils import *
from arguments import get_args
from logger import get_logger, logging_args


def main():
    args = get_args()
    torch_fix_seed(args.seed)
    device = torch.device(args.device)

    if args.over_fitting:
        mode = "over-fitting"
        args.over_fitting = False
    elif args.label_flipping > 0:
        mode = "label-flipping"
        args.label_flipping = 0
    elif args.label_flipping == 0:
        mode = "normal"
    else:
        raise NotImplementedError
    
    if args.mode == "clean":
        start_epoch = args.before
        load_model_file = f"./train/{args.dataset}/{args.arch}/{args.epoch}/{args.lr}/{args.seed}/{mode}/{args.pretrained}/{args.before}"
    elif args.mode == "poisoned":
        start_epoch = args.after
        load_model_file = f"./train/{args.dataset}/{args.arch}/{args.epoch}/{args.lr}/{args.seed}/{mode}/{args.pretrained}/{args.after}"
    elif args.mode == "ecc":
        start_epoch = args.before
        load_model_file = f"./ecc/{args.dataset}-{args.arch}-{args.epoch}-{args.lr}-{mode}/{args.seed}/{args.before}/{args.fixed}/{args.last_layer}/{args.weight_only}/{args.msg_len}/{args.ecc}/{args.sum_params}/{args.prune_ratio}/{args.t}/decoded{args.after}"
    else:
        raise NotImplementedError
    
    save_dir = f"./ecc/{args.dataset}-{args.arch}-{args.epoch}-{args.lr}-{mode}/{args.seed}/{args.before}/{args.fixed}/{args.last_layer}/{args.weight_only}/{args.msg_len}/{args.ecc}/{args.sum_params}/{args.prune_ratio}/{args.t}/finetune/{args.mode}"
    os.makedirs(save_dir, exist_ok=True)
    #save_model_dir = f"{save_dir}/model"
    #os.makedirs(save_model_dir, exist_ok=True)
    logging = get_logger(f"{save_dir}/finetune.log")
    logging_args(args, logging)

    train_loader, test_loader = prepare_dataset(args)
    train_losses = []
    test_losses = []
    
    model = load_model(args, load_model_file, device)
    acc, loss = test(model, test_loader, device)
    test_losses.append(loss)
    logging.info(f"INITIAL EPOCH {start_epoch} VAL ACC: {acc:.6f}\t"
        f"VAL LOSS: {loss:.6f}")
    
    optimizer = make_optim(args, model, pretrained=False)
    #scheduler =lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    for epoch in range(start_epoch+1, args.epoch+1):
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
        #save_model(model, f"{save_model_dir}/{epoch}")

    del model
    torch.cuda.empty_cache()

    plot_linear(train_losses, f"{save_dir}/train")
    plot_linear(test_losses, f"{save_dir}/test")
    
    exit()


if __name__ == "__main__":
    main()

