import argparse
import datetime
import os
import struct

import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from network import *
from utils import *


def calc_acc(args, model_before, model_after, model_decoded, save_dir):
    model_before.eval()
    model_after.eval()
    model_decoded.eval()

    # save parameter
    params_before = get_params_info(args, model_before)
    params_before = get_params_info(args, model_before)
    params_before = get_params_info(args, model_before)
    params_info = {"before":[], "after":[], "decoded":[]}

    # save distance between 2 param
    distance = {"b_and_a":[], "b_and_d":[], "a_and_d":[]}
    for i, b_b in enumerate(p_sum["b_b"]):
        b_a = p_sum["b_a"][i]
        b_d = p_sum["b_d"][i]
        dist_b_and_a = 0
        dist_b_and_d = 0
        dist_a_and_d = 0
        for j, b in enumerate(b_b): # loop with 1 bit
            if b != b_a[j]:
                dist_b_and_a += 1
            if b != b_d[j]:
                dist_b_and_d += 1
            if b_a[j] != b_d[j]:
                dist_a_and_d += 1
        distance["b_and_a"].append(dist_b_and_a)
        distance["b_and_d"].append(dist_b_and_d)
        distance["a_and_d"].append(dist_a_and_d)

    # plot parameter
    plt.plot(p_sum["p_d"], label=f"decoded epoch {args.before}", alpha=1.0)
    plt.plot(p_sum["p_b"], label=f"epoch {args.before}", alpha=0.5)
    plt.plot(p_sum["p_a"], label=f"epoch {args.after}", alpha=0.5)
    plt.legend()
    plt.savefig(f"{save_dir}/parameter{args.after}.png")
    plt.clf()

    with open(f"{save_dir}/parameter{args.after}.txt", "w") as result_file:
        result_file.write(f"{args.before},{args.after},decoded\n")
        for i, p_b in enumerate(p_sum["p_b"]):
            p_a = p_sum["p_a"][i]
            p_d = p_sum["p_d"][i]
            result_file.write(f"{p_b},{p_a},{p_d}\n")

    decode_fails_index = [i for i, p in enumerate(p_sum["p_d"]) if p >= 1]
    with open(f"{save_result_dir}/decode_fails_param.txt", "w") as result_file:
        result_file.write(f",{args.before},{args.after},decoded,"
            f"distance {args.before}-{args.after},distance {args.before}-decoded,distance {args.after}-decoded\n")
        for i in decode_fails_index:
            p_b = p_sum["p_b"][i]
            p_a = p_sum["p_a"][i]
            p_d = p_sum["p_d"][i]
            s_b = p_sum["s_b"][i]
            s_a = p_sum["s_a"][i]
            s_d = p_sum["s_d"][i]
            dist_b_and_a = distance["b_and_a"][i]
            dist_b_and_d = distance["b_and_d"][i]
            dist_a_and_d = distance["a_and_d"][i]

            result_file.write(f"parameter,{p_b},{p_a},{p_d},{p_b-p_a},{p_b-p_d},{p_a-p_d}\n"
                    f"strings,{s_b},{s_a},{s_d},{dist_b_and_a},{dist_b_and_d},{dist_a_and_d}\n")

    """
    # save min hamming distnace of BEFORE
    min_dist_before = 32
    for i, b_b in enumerate(p_sum["b_b"]):
        for j, b_b2 in enumerate(p_sum["b_b"]):
            if i == j:
                continue
            dist_tmp = 0
            for k, b in enumerate(b_b):
                if b != b_b2[k]:
                    dist_tmp += 1
            if min_dist_before > dist_tmp:
                min_dist_before = dist_tmp
    """

    # save acc of correcting
    success = {}   # key:dist_b_and_a, value:{block:[success or fail], symbol:[match number of symbol]]
    for i, p_b in enumerate(p_sum["p_b"]):
        p_a = p_sum["p_a"][i]
        p_d = p_sum["p_d"][i]
        b_b = p_sum["b_b"][i]
        b_a = p_sum["b_a"][i]
        b_d = p_sum["b_d"][i]
        dist_b_and_a = distance["b_and_a"][i]

        if dist_b_and_a not in success:
            success.update({dist_b_and_a:{"block":[],"symbol":[]}})

        if p_b == p_d:
            success[dist_b_and_a]["block"].append(True)
        else:
            success[dist_b_and_a]["block"].append(False)
        sym_temp = 0
        for j, b in enumerate(b_b):
            if b == b_d[j]:
                sym_temp += 1
        success[dist_b_and_a]["symbol"].append(sym_temp)

    block_success_all = 0
    with open(f"{save_result_dir}/acc.txt", "w") as result_file:
        result_file.write(f"hamming distance,number,block success number,symbol success number\n")
        for k in sorted(success.items()):
            dist_b_and_a = k[0]
            block_num = k[1]["block"]
            symbol_success = k[1]["symbol"]
            block_success = block_num.count(True)
            block_success_all += block_success
            result_file.write(f"{dist_b_and_a},{len(block_num)},{block_success},{sum(symbol_success)}\n")
    print(f"block correcting acc: {block_success_all}/{len(p_sum['p_b'])}={block_success_all/len(p_sum['p_b'])}")


def check_output(args, model_before, model_after, model_decoded, save_dir, device):
    model_before.eval()
    model_after.eval()
    model_decoded.eval()
    _, test_loader = prepare_dataset(args)

    save_data_dir = f"./data/{args.ecc}/{args.last_layer}/{args.date}"
    os.makedirs(save_data_dir, exist_ok=True)
    save_data_file = f"{save_data_dir}/b{args.before}_a{args.after}.npz"
    if not os.path.isfile(save_data_file):
        indice, outputs = save_output_dist(model_before, model_after, test_loader, device)
        np.savez(save_data_file, indice=indice, outputs=outputs)

    dist_data = np.load(save_data_file)   # dist_data["indice"], dist_data["outputs"]
    fail, deterioration = check_output_dist(model_before, model_decoded, test_loader, dist_data, device)
    
    before_acc, before_loss = test(model_before, test_loader, device)
    after_acc, after_loss = test(model_after, test_loader, device)
    decoded_acc, decoded_loss = test(model_decoded, test_loader, device)
    print(f"Before\tacc: {before_acc},\tloss: {before_loss}")
    print(f"After\tacc: {after_acc},\tloss: {after_loss}")
    print(f"Decoded\tacc: {decoded_acc},\tloss: {decoded_loss}")

    with open(f"{save_result_dir}/output_dist.txt", "w") as result_file:
        result_file.write(f"{len(dist_data['indice'])} differences -> decoded -> {len(fail)} + {len(deterioration)} differences\n")
        result_file.write(f"\n\nkeep the differences between BEFORE and AFTER because failure of decoding\n")
        for f in fail:
            result_file.write(f"after, {f[1]}, -> decoded, {f[2]}, correct, {f[0]}\n")
        result_file.write(f"\n\ndeteriorate the match between BEFORE and AFTER because failure of decoding\n")
        for d in deterioration:
            result_file.write(f"correct, {d[0]}, -> decoded, {d[1]}\n")


def save_output_dist(model_before, model_after, test_loader, device):
    dist_indice = []
    dist_outputs_after = []

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs_before = model_before(imgs)
            outputs_after = model_after(imgs)

            pred_before = np.argmax(outputs_before.to("cpu").detach().numpy(), axis=1)
            pred_after = np.argmax(outputs_after.to("cpu").detach().numpy(), axis=1)

            for j, p_b in enumerate(pred_before):
                p_a = pred_after[j]
                if p_b != p_a:
                    dist_indice.append(i*len(test_loader)+j)
                    dist_outputs_after.append(outputs_after[j].to("cpu").detach().numpy())

    return np.array(dist_indice), np.array(dist_outputs_after)


def check_output_dist(model_before, model_decoded, test_loader, dist_data, device):
    difference = []

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs_before = model_before(imgs)
            outputs_decoded = model_decoded(imgs)

            pred_before = np.argmax(outputs_before.to("cpu").detach().numpy(), axis=1)
            pred_decoded = np.argmax(outputs_decoded.to("cpu").detach().numpy(), axis=1)

            for j, p_b in enumerate(pred_before):
                p_d = pred_decoded[j]
                if p_b != p_d:
                    difference.append((i*len(test_loader)+j, p_b, p_d))

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


def main(args):
    args = get_args()
    torch_fix_seed(args.seed)
    device = torch.device(args.device)
    save_dir = f"./ecc/{args.date}/{args.before}/{args.msg_len}/{args.last_layer}/{args.ecc}"
    
    model_before = load_model(args, f"./model/{args.date}/{args.before}", device)
    model_after = load_model(args, f"./model/{args.date}/{args.after}", device)
    model_decoded = load_model(args, f"{save_dir}/decoded{args.after}", device)
    
    if args.mode == "acc":
        calc_acc(args, model_before, model_after, model_decoded, save_dir)
    elif args.mode == "output":
        check_output(args, model_before, model_after, model_decoded, save_dir, device)
    else:
        raise NotImplementedError

    exit()


if __name__ == "__main__":
    main(args)

