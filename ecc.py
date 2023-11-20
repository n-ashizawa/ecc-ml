import os
import copy
import time

import numpy as np
import torch

from ErrorCorrectingCode import turbo_code, rs_code, bch_code

from network import *
from utils import *
from arguments import get_args
from logger import get_logger, logging_args


def encode_before(args, model_before, ECC, save_dir, logging):
    model_encoded = copy.deepcopy(model_before)
    # Get the state dict
    state_dict = model_encoded.state_dict()
    all_reds1 = []
    all_reds2 = []

    for name, param in model_before.named_parameters():
        if "weight" not in name:
            continue
        if args.last_layer:
            last_layer = [n for n, _ in model_before.named_parameters() if "weight" in n][-1]
            if name != last_layer:
                continue
        
        encoded_params = []
        reds1_list = []
        reds2_list = []

        for value in param.view(-1):
            (_, _, b_b_all) = get_bin_from_param(value.item())
            # limit bits
            b_b = b_b_all[:args.msg_len]
            encoded_msg = ECC.encode(b_b)
            b_e = encoded_msg[:args.msg_len]
            reds1 = encoded_msg[args.msg_len:args.msg_len*3]
            reds2 = encoded_msg[args.msg_len*3:]
            reds1_list.append(reds1)
            reds2_list.append(reds2)
            # extend bits
            b_e_all = np.concatenate([b_e, b_b_all[args.msg_len:]])
            p_e, _, _ = get_param_from_bin(b_e_all)
            encoded_params.append(p_e)

        all_reds1.append(reds1_list)
        all_reds2.append(reds2_list)
        reshape_encoded_params = torch.Tensor(encoded_params).view(param.data.size())
        # Modify the state dict
        state_dict[name] = reshape_encoded_params
        logging.info(f"{name} is encoded")

    write_varlen_csv(all_reds1, f"{save_dir}/reds1")
    write_varlen_csv(all_reds2, f"{save_dir}/reds2")

    # Load the modified state dict
    model_encoded.load_state_dict(state_dict)
    save_model(model_encoded, f"{save_dir}/encoded")
    del model_encoded
 

def decode_after(args, model_after, ECC, save_dir, logging):
    model_decoded = copy.deepcopy(model_after)
    # Get the state dict
    state_dict = model_decoded.state_dict()
    all_reds1_str = read_varlen_csv(f"{save_dir}/reds1")
    all_reds1 = get_intlist_from_strlist(all_reds1_str)
    logging.info("all no.1 redundants are loaded")
    all_reds2_str = read_varlen_csv(f"{save_dir}/reds2")
    all_reds2 = get_intlist_from_strlist(all_reds2_str)
    logging.info("all no.2 redundants are loaded")

    i = 0
    for name, param in model_after.named_parameters():
        if "weight" not in name:
            continue
        if args.last_layer:
            last_layer = [n for n, _ in model_after.named_parameters() if "weight" in n][-1]
            if name != last_layer:
                continue
        
        decoded_params = []
        for j, value in enumerate(param.view(-1)):
            (_, _, b_a_all) = get_bin_from_param(value.item())
            # bit limited
            # limit bits
            b_a = b_a_all[:args.msg_len]
            reds1 = all_reds1[i][j]
            reds2 = all_reds2[i][j]
            encoded_msg = np.concatenate([b_a, reds1, reds2])
            b_d = ECC.decode(encoded_msg)
            # extend bits
            b_d_all = np.concatenate([b_d, b_a_all[args.msg_len:]])
            p_d, _, _ = get_param_from_bin(b_d_all)
            decoded_params.append(p_d)

        reshape_decoded_params = torch.Tensor(decoded_params).view(param.data.size())
        # Modify the state dict
        state_dict[name] = reshape_decoded_params
        logging.info(f"{name} is decoded")
        i += 1

    # Load the modified state dict
    model_decoded.load_state_dict(state_dict)
    save_model(model_decoded, f"{save_dir}/decoded{args.after}")
    del model_decoded


def main():
    args = get_args()
    torch_fix_seed(args.seed)
    
    device = torch.device("cpu")
    save_dir = f"./ecc/{args.date}/{args.before}/{args.msg_len}/{args.last_layer}/{args.ecc}"
    os.makedirs(save_dir, exist_ok=True)
    
    if args.ecc == "turbo":
        ECC = turbo_code.TurboCode(args)
    elif args.ecc == "rs":
        ECC = rs_code.RSCode(args)
    elif args.ecc == "bch":
        ECC = bch_code.BCHCode(args)
    else:
        raise NotImplementedError

    if args.mode == "encode":
        logging = get_logger(f"{save_dir}/{args.mode}.log")
        logging_args(args, logging)
        model = load_model(args, f"./model/{args.date}/{args.before}", device)
        start_time = time.time()
        encode_before(args, model, ECC, save_dir, logging)
        end_time = time.time()
    elif args.mode == "decode":
        logging = get_logger(f"{save_dir}/{args.mode}{args.after}.log")
        logging_args(args, logging)
        model = load_model(args, f"./model/{args.date}/{args.after}", device)
        start_time = time.time()
        decode_after(args, model, ECC, save_dir, logging)
        end_time = time.time()
    else:
        raise NotImplementedError
    
    elapsed_time = end_time - start_time
    logging.info(f"time cost: {elapsed_time} seconds")

    del model
    exit()


if __name__ == "__main__":
    main()
