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


def get_name_from_prune_targets(args, model, save_dir):
    save_data_file = "/".join(save_dir.split("/")[:5]) + f"/prune_targets{args.prune_ratio}.npy"
    prune_targets = np.load(save_data_file)
    get_forward_steps = model.get_forward_steps()

    prune_targets_name = {}
    for layer, weight_id in prune_targets:
        target_module = get_forward_steps[layer]
        for name, module in model.named_modules():
            if id(module) == id(target_module):
                if name not in prune_targets_name:
                    prune_targets_name[name] = []
                prune_targets_name[name].append(weight_id)

    return prune_targets_name


def encode_before(args, model_before, ECC, save_dir, logging):
    # Get the state dict
    model_encoded = copy.deepcopy(model_before)
    state_dict_before = model_before.state_dict()
    state_dict_encoded = model_encoded.state_dict()
    all_reds1 = []
    all_reds2 = []
    if args.prune_ratio > 0:
        prune_targets_name = get_name_from_prune_targets(args, model_before, save_dir)
        modules_before = {name: module for name, module in model_before.named_modules()}
        weight_ids = None
    
    for name in state_dict_before:
        if args.last_layer:
            last_layer = [n for n in state_dict_before][-1].split(".")[0]
            if last_layer not in name:
                continue
        if args.weight_only:
            if "weight" not in name:
                continue
        if "num_batches_tracked" in name:
            continue

        param = state_dict_before[name]
        encoded_params = []
        reds1 = []
        reds2 = []
        params = []
        sum_params = 0

        if args.prune_ratio > 0:
            #print(name, param.shape)
            layer = '.'.join(name.split('.')[:-1])
            is_weight = (name.split('.')[-1] == "weight")
            is_conv = layer in prune_targets_name   # conv
            is_linear = layer in modules_before and isinstance(modules_before[layer], torch.nn.Linear)   # linear
            #print(layer, "weight:", is_weight, "conv:", is_conv, "linear:", is_linear)
        
        for ids, value in enumerate(param.view(-1)):
            if args.prune_ratio > 0:
                original_index = np.unravel_index(ids, param.shape)
                if is_conv or is_linear:   # conv or linear
                    if is_weight and weight_ids is not None:   # weight
                        if original_index[1] not in weight_ids:   # targets
                            encoded_params.append(value.item())
                            continue
                        #print('1', "ids:", ids, "original:", original_index[1], "wid:", weight_ids)
                if is_conv:   # conv
                    weight_ids = prune_targets_name[layer]   # update
                    #print("new", weight_ids)
                
                if not is_linear:
                    if original_index[0] not in weight_ids:   # targets
                        encoded_params.append(value.item())
                        continue
                    #print('0', "ids:", ids, "original:", original_index[0], "wid:", weight_ids)

            params.append(value.item())
            sum_params += 1
            if args.sum_params > sum_params:
                continue
            whole_b_bs = []
            b_b = []
            for p in params:
                (_, _, whole_b_b), integer_len = get_bin_from_param(p, length=args.msg_len, fixed=args.fixed)
                # limit bits
                whole_b_bs.append(whole_b_b)   # storage all bits
                b_b.extend(whole_b_b[:args.msg_len])
            encoded_msg = ECC.encode(b_b)
            msglen = args.msg_len*args.sum_params
            redlen = args.t*4   # 4 = 8 % 2
            b_es = encoded_msg[:msglen]
            reds1.append(encoded_msg[msglen:msglen+redlen])
            reds2.append(encoded_msg[msglen+redlen:])
            b_e = []
            for i in range(len(whole_b_bs)):
                b_e = b_es[i*args.msg_len:(i+1)*args.msg_len]
                # extend bits
                whole_b_e = np.concatenate([b_e, whole_b_bs[i][args.msg_len:]])
                p_e, _, _ = get_param_from_bin(whole_b_e, integer_len=integer_len, fixed=args.fixed)
                encoded_params.append(p_e)
            params = []   # initialize
            sum_params = 0   # initialize
            
        all_reds1.append(reds1)
        all_reds2.append(reds2)
        reshape_encoded_params = torch.Tensor(encoded_params).view(param.data.size())
        # Modify the state dict
        state_dict_encoded[name] = reshape_encoded_params
        logging.info(f"{name} is encoded")

    write_varlen_csv(all_reds1, f"{save_dir}/reds1")
    write_varlen_csv(all_reds2, f"{save_dir}/reds2")

    # Load the modified state dict
    model_encoded.load_state_dict(state_dict_encoded)
    save_model(model_encoded, f"{save_dir}/encoded")
    del model_encoded
 

def decode_after(args, model_after, ECC, save_dir, logging):
    # Get the state dict
    model_decoded = copy.deepcopy(model_after)
    state_dict_after = model_after.state_dict()
    state_dict_decoded = model_decoded.state_dict()
    # Load the encoded redidundants
    all_reds1_str = read_varlen_csv(f"{save_dir}/reds1")
    all_reds1 = get_intlist_from_strlist(all_reds1_str)
    logging.info("all no.1 redundants are loaded")
    all_reds2_str = read_varlen_csv(f"{save_dir}/reds2")
    all_reds2 = get_intlist_from_strlist(all_reds2_str)
    logging.info("all no.2 redundants are loaded")

    if args.prune_ratio > 0:
        prune_targets_name = get_name_from_prune_targets(args, model_after, save_dir)
        modules_after = {name: module for name, module in model_after.named_modules()}
        weight_ids = None
        
    i = 0
    for name in state_dict_after:
        if args.last_layer:
            last_layer = [n for n in state_dict_after][-1].split(".")[0]
            if last_layer not in name:
                continue
        if args.weight_only:
            if "weight" not in name:
                continue
        if "num_batches_tracked" in name:
            continue
        
        param = state_dict_after[name]
        decoded_params = []
        params = []
        sum_params = 0

        if args.prune_ratio > 0:
            layer = '.'.join(name.split('.')[:-1])
            is_weight = (name.split('.')[-1] == "weight")
            is_conv = layer in prune_targets_name   # conv
            is_linear = layer in modules_after and isinstance(modules_after[layer], torch.nn.Linear)   # linear
            
        j = 0
        for ids, value in enumerate(param.view(-1)):
            if args.prune_ratio > 0:
                original_index = np.unravel_index(ids, param.shape)
                if is_conv or is_linear:   # conv or linear
                    if is_weight and weight_ids is not None:   # weight
                        if original_index[1] not in weight_ids:   # targets
                            decoded_params.append(value.item())
                            continue
                if is_conv:   # conv
                    weight_ids = prune_targets_name[layer]   # update
                
                if not is_linear:
                    if original_index[0] not in weight_ids:   # targets
                        decoded_params.append(value.item())
                        continue

            params.append(value.item())
            sum_params += 1
            if args.sum_params > sum_params:
                continue
            whole_b_as = []
            b_a = []
            for p in params:
                (_, _, whole_b_a), integer_len = get_bin_from_param(p, length=args.msg_len, fixed=args.fixed)
                # limit bits
                whole_b_as.append(whole_b_a)   # storage all bits
                b_a.extend(whole_b_a[:args.msg_len])
            reds1 = all_reds1[i][j]
            reds2 = all_reds2[i][j]
            encoded_msg = np.concatenate([b_a, reds1, reds2])
            b_ds = ECC.decode(encoded_msg)
            b_d = []
            for k in range(len(whole_b_as)):
                b_d = b_ds[k*args.msg_len:(k+1)*args.msg_len]
                # extend bits
                whole_b_d = np.concatenate([b_d, whole_b_as[k][args.msg_len:]])
                p_d, _, _ = get_param_from_bin(whole_b_d, integer_len=integer_len, fixed=args.fixed)
                decoded_params.append(p_d)
            j += 1
            params = []   # initialize
            sum_params = 0   # initialize

        reshape_decoded_params = torch.Tensor(decoded_params).view(param.data.size())
        # Modify the state dict
        state_dict_decoded[name] = reshape_decoded_params
        logging.info(f"{name} is decoded")
        i += 1

    # Load the modified state dict
    model_decoded.load_state_dict(state_dict_decoded)
    save_model(model_decoded, f"{save_dir}/decoded{args.after}")
    del model_decoded


def main():
    args = get_args()
    torch_fix_seed(args.seed)
    device = torch.device("cpu")
    
    if args.over_fitting:
        mode = "over-fitting"
    elif args.label_flipping > 0:
        mode = "label-flipping"
    elif args.label_flipping == 0:
        mode = "normal"
    else:
        raise NotImplementedError
    
    if args.quantized == 0:
        model_dir = "model"
        quantized = False
    elif args.quantized > 0:
        model_dir = "quantized"
        quantized = True
    else:
        raise NotImplementedError

    load_dir = f"./train/{args.dataset}/{args.arch}/{args.epoch}/{args.lr}/{args.seed}/{mode}/{args.pretrained}/{model_dir}"
    save_dir = f"./ecc/{args.dataset}-{args.arch}-{args.epoch}-{args.lr}-{mode}{args.seed}/{args.before}/{model_dir}/{args.fixed}/{args.last_layer}/{args.weight_only}/{args.msg_len}/{args.ecc}/{args.sum_params}/{args.prune_ratio}/{args.t}"
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
        model = load_model(args, f"{load_dir}/{args.before}", device, quantized=quantized)
        start_time = time.time()
        encode_before(args, model, ECC, save_dir, logging)
        end_time = time.time()
    elif args.mode == "decode":
        logging = get_logger(f"{save_dir}/{args.mode}{args.after}.log")
        logging_args(args, logging)
        model = load_model(args, f"{load_dir}/{args.after}", device, quantized=quantized)
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
