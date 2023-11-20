import os
import csv

import numpy as np
import torch

import matplotlib.pyplot as plt

from network import *
from utils import *
from arguments import get_args
from logger import get_logger, logging_args


def calc_acc(args, model_before, model_after, model_decoded, save_dir, logging):
    model_before.eval()
    model_after.eval()
    model_decoded.eval()

    # save parameters
    # {"p_m":[], "s_m":[], "b_m":[]} m: before, after, decoded
    params_before = get_params_info(args, model_before)
    params_after = get_params_info(args, model_after)
    params_decoded = get_params_info(args, model_decoded)
    params_info = {"before":params_before, "after":params_after, "decoded":params_decoded}
    
    # plot parameters
    plt.plot(params_info["decoded"]["p_m"], label=f"decoded epoch {args.before} to {args.after}", alpha=1.0)
    plt.plot(params_info["before"]["p_m"], label=f"epoch {args.before}", alpha=0.5)
    plt.plot(params_info["after"]["p_m"], label=f"epoch {args.after}", alpha=0.5)
    plt.legend()
    plt.savefig(f"{save_dir}/parameters{args.after}.png")
    plt.clf()

    # save distance between 2 parameters
    dist_before_after = get_dist_of_params(params_info["before"], params_info["after"])
    dist_before_decoded = get_dist_of_params(params_info["before"], params_info["decoded"])
    dist_after_decoded = get_dist_of_params(params_info["after"], params_info["decoded"])
    distance_info = {"before_after":dist_before_after, "before_decoded":dist_before_decoded, "after_decoded":dist_after_decoded}
    
    # save acc of correcting
    block_success = []
    for i, b_b in enumerate(params_info["before"]["b_m"]):
        b_a = params_info["after"]["b_m"][i]
        b_d = params_info["decoded"]["b_m"][i]
        dist_b_and_a = distance_info["before_after"][i]

        if b_b == b_d:
            block_success.append(True)
        else:
            block_success.append(False)
        
    symbol_success = []
    for b_d in distance_info["before_decoded"]:
        symbol_success.append(args.msg_len - b_d)
    
    success = {"block":block_success, "symbol":symbol_success}
    
    # plot all
    with open(f"{save_dir}/acc{args.after}.txt", "w", newline="") as f:
        writer = csv.writer(f)
        header = np.concatenate([list(params_info.keys()), list(distance_info.keys()), list(success.keys())])
        writer.writerow(header)
        for b, a, d, dist_b_a, dist_b_d, dist_a_d, block, symbol in zip(
            params_info["before"]["p_m"], params_info["after"]["p_m"], params_info["decoded"]["p_m"],
            distance_info["before_after"], distance_info["before_decoded"], distance_info["after_decoded"], 
            success["block"], success["symbol"]):
            writer.writerow([b, a, d, dist_b_a, dist_b_d, dist_a_d, block, symbol])
    
    dist_success = {}
    for dist, block, symbol in zip(distance_info["before_after"], success["block"], success["symbol"]):
        if dist not in dist_success:
            dist_success[dist] = {"count": 0, "block": 0, "symbol": 0}
        dist_success[dist]["count"] += 1
        dist_success[dist]["block"] += block
        dist_success[dist]["symbol"] += symbol
    
    for dist in sorted(dist_success):
        logging.info(
            f"[{dist}]\tBlock acc: {dist_success[dist]['block']}/{dist_success[dist]['count']}="
            f"{dist_success[dist]['block']/dist_success[dist]['count']}\t"
            f"Symbol acc: {dist_success[dist]['symbol']}/{dist_success[dist]['count']}*{args.msg_len}="
            f"{dist_success[dist]['symbol']/(dist_success[dist]['count']*args.msg_len)}"
        )


def check_output(args, model_before, model_after, model_decoded, device, save_dir, logging):
    model_before.eval()
    model_after.eval()
    model_decoded.eval()
    _, test_loader = prepare_dataset(args)

    save_data_file = f"./ecc/{args.date}/{args.before}/diff{args.after}.npz"
    if not os.path.isfile(save_data_file):
        indice, outputs = save_output_dist(model_before, model_after, test_loader, device)
        np.savez(save_data_file, indice=indice, outputs=outputs)

    dist_data = np.load(save_data_file)   # dist_data["indice"], dist_data["outputs"]
    fail, deterioration = check_output_dist(model_before, model_decoded, test_loader, dist_data, device)
    
    before_acc, before_loss = test(model_before, test_loader, device)
    after_acc, after_loss = test(model_after, test_loader, device)
    decoded_acc, decoded_loss = test(model_decoded, test_loader, device)
    logging.info(f"Before\tacc: {before_acc},\tloss: {before_loss}")
    logging.info(f"After\tacc: {after_acc},\tloss: {after_loss}")
    logging.info(f"Decoded\tacc: {decoded_acc},\tloss: {decoded_loss}")

    with open(f"{save_dir}/output{args.after}.txt", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([len(dist_data["indice"]), "->", len(fail), len(deterioration)])
        writer.writerow(["after", "decoded", "before"])
        for fa in fail:
            writer.writerow([fa[1], fa[2], fa[0]])
        writer.writerow([])
        for de in deterioration:
            writer.writerow([de[0], de[1]])


def save_output_dist(model_before, model_after, test_loader, device):
    indice = 0
    dist_indice = []
    dist_outputs_after = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs_before = model_before(imgs)
            outputs_after = model_after(imgs)

            pred_before = np.argmax(outputs_before.to("cpu").detach().numpy(), axis=1)
            pred_after = np.argmax(outputs_after.to("cpu").detach().numpy(), axis=1)

            for i, p_b in enumerate(pred_before):
                p_a = pred_after[i]
                if p_b != p_a:
                    dist_indice.append(indice)
                    dist_outputs_after.append(outputs_after[i].to("cpu").detach().numpy())
                
                indice += 1

    return np.array(dist_indice), np.array(dist_outputs_after)


def check_output_dist(model_before, model_decoded, test_loader, dist_data, device):
    indice = 0
    difference = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs_before = model_before(imgs)
            outputs_decoded = model_decoded(imgs)

            pred_before = np.argmax(outputs_before.to("cpu").detach().numpy(), axis=1)
            pred_decoded = np.argmax(outputs_decoded.to("cpu").detach().numpy(), axis=1)

            for i, p_b in enumerate(pred_before):
                p_d = pred_decoded[i]
                if p_b != p_d:
                    difference.append((indice, p_b, p_d))
                
                indice += 1

    fail = []
    deterioration = []
    for d in difference:
        if d[0] in dist_data["indice"]:
            outputs_after = dist_data["outputs"][np.where(dist_data["indice"]==d[0])]
            pred_after = np.argmax(outputs_after)
            fail.append((d[1], pred_after, d[2]))
        else:
            deterioration.append((d[1], d[2]))

    return fail, deterioration


def main():
    args = get_args()
    torch_fix_seed(args.seed)
    
    device = torch.device(args.device)
    save_dir = f"./ecc/{args.date}/{args.before}/{args.msg_len}/{args.last_layer}/{args.sum_params}/{args.ecc}"
    os.makedirs(save_dir, exist_ok=True)
    
    logging = get_logger(f"{save_dir}/{args.mode}{args.after}.log")
    logging_args(args, logging)

    model_before = load_model(args, f"./model/{args.date}/{args.before}", device)
    model_after = load_model(args, f"./model/{args.date}/{args.after}", device)
    model_decoded = load_model(args, f"{save_dir}/decoded{args.after}", device)

    if args.mode == "acc":
        calc_acc(args, model_before, model_after, model_decoded, save_dir, logging)
    elif args.mode == "output":
        check_output(args, model_before, model_after, model_decoded, device, save_dir, logging)
    else:
        raise NotImplementedError
    
    
    exit()


if __name__ == "__main__":
    main()

