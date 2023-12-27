import csv
import re

from utils import *
from arguments import get_args
from logger import get_logger, logging_args


def get_load_dir_name(args, taget_param, candidate):
    params = {"before": args.before, "fixed": args.fixed, "msg_len": args.msg_len, 
        "ecc": args.ecc, "prune_ratio": args.prune_ratio, "t": args.t}

    params[taget_param] = candidate
    load_dir = f"{params['before']}/model/{params['fixed']}/False/False/{params['msg_len']}/{params['ecc']}/1/{params['prune_ratio']}/{params['t']}"
    return load_dir


def create_summary_files(args, target_param, candidates, seeds, save_dir):
    # Create a new file for this target_param value
    with open(f"{save_dir}/{target_param}{'-'.join(candidates)}.txt", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['', "seed"] + candidates)
        writer.writerow([])

        encode_time = [["encode time"]]
        decode_time = [["decode time"]]
        decoded_acc = [["decoded acc"]]
        decoded_loss = [["decoded loss"]]
        before_acc = [["before acc"]]
        before_loss = [["before loss"]]
        after_acc = [["after acc"]]
        after_loss = [["after loss"]]
        output_diff_before_after = [["output diff b/w before and after"]]
        output_diff_before_decoded_inside = [["output diff b/w before and decoded (inside)"]]
        output_diff_before_decoded_outside = [["output diff b/w before and decoded (outside)"]]
    
        for seed in seeds:
            load_parent_dir = f"{'/'.join(save_dir.split('/')[:-1])}/{seed}"
            encode_time_per_seed = ['', seed]
            decode_time_per_seed = ['', seed]
            decoded_acc_per_seed = ['', seed]
            decoded_loss_per_seed = ['', seed]
            before_acc_per_seed = ['', seed]
            before_loss_per_seed = ['', seed]
            after_acc_per_seed = ['', seed]
            after_loss_per_seed = ['', seed]
            output_diff_before_after_per_seed = ['', seed]
            output_diff_before_decoded_inside_per_seed = ['', seed]
            output_diff_before_decoded_outside_per_seed = ['', seed]

            for c in candidates:
                load_dir = f"{load_parent_dir}/{get_load_dir_name(args, target_param, c)}"
        
                with open(f"{load_dir}/encode.log", "r") as log_file:
                    encode_time_per_seed.append(re.search(r"time cost: (\d+\.\d+)", log_file.read()).group(1))

                with open(f"{load_dir}/decode{args.after}.log", "r") as log_file:
                    decode_time_per_seed.append(re.search(r"time cost: (\d+\.\d+)", log_file.read()).group(1))

                with open(f"{load_dir}/output{args.after}.log", "r") as log_file:
                    log_content = log_file.read()
                    decoded_acc_per_seed.append(re.search(r"Decoded\s+acc: (\d+\.\d+)", log_content).group(1))
                    decoded_loss_per_seed.append(re.search(r"Decoded\s+acc: \d+\.\d+,\s+loss:\s+(\d+\.\d+)", log_content).group(1))
                    before_acc_per_seed.append(re.search(r"Before\s+acc: (\d+\.\d+)", log_content).group(1))
                    before_loss_per_seed.append(re.search(r"Before\s+acc: \d+\.\d+,\s+loss:\s+(\d+\.\d+)", log_content).group(1))
                    after_acc_per_seed.append(re.search(r"After\s+acc: (\d+\.\d+)", log_content).group(1))
                    after_loss_per_seed.append(re.search(r"After\s+acc: \d+\.\d+,\s+loss:\s+(\d+\.\d+)", log_content).group(1))
                
                with open(f"{load_dir}/output{args.after}.txt", "r") as txt_file:
                    line = txt_file.readline()
                    numbers = re.findall(r'\d+', line)
                    if len(numbers) != 3:
                        raise ValueError(f"Wrong the numbers of output diff in {load_dir}/output{args.after}.txt")
                    output_diff_before_after_per_seed.append(numbers[0])
                    output_diff_before_decoded_inside_per_seed.append(numbers[1])
                    output_diff_before_decoded_outside_per_seed.append(numbers[2])

            encode_time.append(encode_time_per_seed)
            decode_time.append(decode_time_per_seed)
            decoded_acc.append(decoded_acc_per_seed)
            decoded_loss.append(decoded_loss_per_seed)
            output_diff_before_after.append(output_diff_before_after_per_seed)
            output_diff_before_decoded_inside.append(output_diff_before_decoded_inside_per_seed)
            output_diff_before_decoded_outside.append(output_diff_before_decoded_outside_per_seed)
            before_acc.append(before_acc_per_seed)
            before_loss.append(before_loss_per_seed)
            after_acc.append(after_acc_per_seed)
            after_loss.append(after_loss_per_seed)

        writer.writerows(encode_time + decode_time + decoded_acc + decoded_loss)
        writer.writerow([])
        writer.writerows(output_diff_before_after + output_diff_before_decoded_inside + output_diff_before_decoded_outside)
        writer.writerow([])
        writer.writerows(before_acc + before_loss + after_acc + after_loss)


    with open(f"{save_dir}/{target_param}{'-'.join(candidates)}_acc.txt", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['', "hamming distance"] + candidates)
        writer.writerow([])
        
        for seed in seeds:
            block_acc_per_seed = []
            symbol_acc_per_seed = []
            load_parent_dir = f"{'/'.join(save_dir.split('/')[:-1])}/{seed}"

            for c in candidates:
                block_acc_per_candi = {}
                symbol_acc_per_candi = {}
                load_dir = f"{load_parent_dir}/{get_load_dir_name(args, target_param, c)}"

                with open(f"{load_dir}/acc{args.after}.log", "r") as log_file:
                    for line in log_file:
                        # Use regular expressions to extract the necessary information
                        match = re.search(r"\[(\d+)\]\s+Block acc:\s+.+=(\d+\.\d+[eE]?[-+]?[0-9]*)\s+Symbol acc:\s+.+=(\d+\.\d+[eE]?[-+]?[0-9]*)", line)

                        if match:
                            # If a match is found, add the information to the dictionary
                            key = int(match.group(1))
                            block_acc_per_line = float(match.group(2))
                            symbol_acc_per_line = float(match.group(3))
                            if key in block_acc_per_candi:
                                raise ValueError(f"Duplicate key {key} in {load_dir}/acc{args.after}.log")
                            block_acc_per_candi.update({key: block_acc_per_line})
                            symbol_acc_per_candi.update({key: symbol_acc_per_line})
                    
                    block_acc_per_seed.append(block_acc_per_candi)
                    symbol_acc_per_seed.append(symbol_acc_per_candi)
            
            writer.writerow([f"block acc {seed}"])
            block_max_key = max(max(acc.keys()) for acc in block_acc_per_seed)
            block_min_key = min(min(acc.keys()) for acc in block_acc_per_seed)
            for key in range(block_min_key, block_max_key+1):
                row = [key]
                for acc in block_acc_per_seed:
                    row.append(acc.get(key, 'N/A'))  # Use 'N/A' for missing keys
                writer.writerow([''] + row)

            writer.writerow([])

            writer.writerow([f"symbol acc {seed}"])
            symbol_max_key = max(max(acc.keys()) for acc in symbol_acc_per_seed)
            symbol_min_key = min(min(acc.keys()) for acc in symbol_acc_per_seed)
            for key in range(symbol_min_key, symbol_max_key+1):
                row = [key]
                for acc in symbol_acc_per_seed:
                    row.append(acc.get(key, 'N/A'))  # Use 'N/A' for missing keys
                writer.writerow([''] + row)
                

def main():
    args = get_args()
    torch_fix_seed(args.seed)
    device = torch.device(args.device)

    seeds = [1]
    target_param = "t"
    candidates = ["5", "6", "7", "8"]

    if args.over_fitting:
        mode = "over-fitting"
    elif args.label_flipping > 0:
        mode = "label-flipping"
    elif args.label_flipping == 0:
        mode = "normal"
    else:
        raise NotImplementedError

    save_dir = f"./ecc/{args.dataset}-{args.arch}-{args.epoch}-{args.lr}-{mode}/table{args.after}"
    os.makedirs(save_dir, exist_ok=True)

    logging = get_logger(f"{save_dir}/{target_param}{'-'.join(candidates)}.log")
    logging_args(args, logging)

    create_summary_files(args, target_param, candidates, seeds, save_dir)

    
if __name__ == '__main__':
    main()
