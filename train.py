'''
MIT License
Copyright (c) 2023 fseclab-osaka

bert finetuning reference: 
https://colab.research.google.com/github/abhimishra91/transformers-tutorials/blob/master/transformers_multi_label_classification.ipynb#scrollTo=5kcqh607S_p_

'''

import os
import csv
import time

import numpy as np
import torch

from network import *
from utils import *
from arguments import get_args
from logger import get_logger, logging_args

from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics,\
    loss_metrics, timing_metrics, cpu_usage_metrics, StreamConfusionMatrix,\
    disk_usage_metrics
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training import Naive, Cumulative


def main():
    args = get_args()
    torch_fix_seed(args.seed)
    device = torch.device(args.device)

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
        if args.clalgo is not None:
            # log to Tensorboard
            tb_logger = TensorboardLogger(tb_log_dir=f"./tb_data/{args.clalgo}-{args.arch}-{args.dataset}-{args.epoch}epochs")
            # log to text file
            text_logger = TextLogger(open(f"{save_dir}/train.log", 'w'))
            # print to stdout
            interactive_logger = InteractiveLogger()
        else:
            logging = get_logger(f"{save_dir}/train.log")
            logging_args(args, logging)

        train_loader, test_loader, n_classes = prepare_dataset(args, save_dir)
        train_losses = []
        test_losses = []

        if args.pretrained == 0:
            model = make_model(args, device, n_classes=n_classes)
        elif args.pretrained > 0:
            for epoch in range(1, args.pretrained+1):
                model = load_model(args, f"./train/{args.dataset}/{args.arch}/{args.epoch}/{args.lr}/normal0/{args.seed}/model/{epoch}", device)
                # test acc
                start_time = time.time()
                acc, loss = test(args, model, test_loader, device)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"test time {epoch}: {elapsed_time} seconds")
                test_losses.append(loss)
                logging.info(f"INITIAL VAL ACC: {acc:.6f}\t"
                    f"VAL LOSS: {loss:.6f}")
                # save model
                save_model(model, f"{save_model_dir}/{epoch}")
                del model
            model = load_model(args, f"./train/{args.dataset}/{args.arch}/{args.epoch}/{args.lr}/normal0/{args.seed}/model/{args.pretrained}", device)
        else:
            raise NotImplementedError

        optimizer = make_optim(args, model)
        
        if args.clalgo is not None:
            eval_plugin = EvaluationPlugin(
                accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
                loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
                #timing_metrics(epoch=True),
                #cpu_usage_metrics(experience=True),
                forgetting_metrics(experience=True, stream=True),
                StreamConfusionMatrix(num_classes=n_classes, save_image=False),
                #disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
                loggers=[interactive_logger, text_logger, tb_logger]
            )

            if args.clalgo == "naive":
                cl_strategy = Naive(
                    model=model, optimizer=optimizer,
                    criterion=nn.CrossEntropyLoss(), train_mb_size=500, train_epochs=args.epoch, eval_mb_size=100,
                    device=device, evaluator=eval_plugin
                )
            elif args.clalgo == "cumulative":
                cl_strategy = Cumulative(
                    model=model, optimizer=optimizer,
                    criterion=nn.CrossEntropyLoss(), train_mb_size=500, train_epochs=args.epoch, eval_mb_size=100,
                    device=device, evaluator=eval_plugin
                )
            else:
                raise NotImplementedError

            results = []
            for experience in train_loader:
                start_time = time.time()
                print("Start of experience: ", experience.current_experience)
                print("Current Classes: ", experience.classes_in_this_experience)

                # train returns a dictionary which contains all the metric values
                res = cl_strategy.train(experience, num_workers=4)
                print('Training completed')

                print('Computing accuracy on the whole test set')
                # eval also returns a dictionary which contains all the metric values
                results.append(cl_strategy.eval(test_loader, num_workers=4))
                save_model(model, f"{save_model_dir}/{experience.current_experience}")
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"train and test time {experience.current_experience}: {elapsed_time} seconds")
            
            del model
            torch.cuda.empty_cache()
        
        else:
            if args.arch == "vit" and mode == "normal":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)
            
            for epoch in range(args.pretrained+1, args.epoch+1):
                start_time = time.time()
                acc, loss = train(args, model, train_loader, optimizer, device)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"train time {epoch}: {elapsed_time} seconds")
                train_losses.append(loss)
                if args.arch == "vit" and mode == "normal":
                    scheduler.step()
                logging.info(f"EPOCH: {epoch}\n"
                    f"TRAIN ACC: {acc:.6f}\t"
                    f"TRAIN LOSS: {loss:.6f}")
                # test acc
                start_time = time.time()
                acc, loss = test(args, model, test_loader, device)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"test time {epoch}: {elapsed_time} seconds")
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

