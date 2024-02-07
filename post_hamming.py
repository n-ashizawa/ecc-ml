'''
MIT License
Copyright (c) 2023 fseclab-osaka
'''

import csv
import re

from utils import *
from arguments import get_args
from logger import get_logger, logging_args


def count_hamming(args, seeds, target_ratios, save_dir):
    # Create a new file for this target_param value
    with open(f"{save_dir}/hamming.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['', "target_ratio", "seed"] + [i for i in range(args.msg_len+1)])
        writer.writerow([])
        
        block_hamming = []
        symbol_hamming = []
        
        for target_ratio in target_ratios:
            args.target_ratio = target_ratio

            block_hamming_per_ratio = [['', target_ratio]]
            average_block_hamming = ['', '', "ave."]
            sum_block_hamming = [0]*(args.msg_len+1)
            symbol_hamming_per_ratio = [['', target_ratio]]
            average_symbol_hamming = ['', '', "ave."]
            sum_symbol_hamming = [0]*(args.msg_len+1)
    
            for seed in seeds:
                args.seed = seed
                load_dir = make_savedir(args)
                
                block_hamming_per_seed = ['', '', seed] + ['']*(args.msg_len+1)
                symbol_hamming_per_seed = ['', '', seed] + ['']*(args.msg_len+1)

                with open(f"{load_dir}/acc{args.after}.log", "r") as log_file:
                    print(f"opened {load_dir}/acc{args.after}.log")
                    for line in log_file:
                        # Use regular expressions to extract the necessary information
                        match = re.search(r"\[(\d+)\]\s+Block acc:\s+\d+/(\d+)=.+\s+Symbol acc:\s+\d+/(\d+)\*\d+=.+", line)
                        
                        if match:
                            # If a match is found, add the information to the dictionary
                            hamming = int(match.group(1))
                            sum_params = int(match.group(2))
                            block_hamming_per_seed[hamming+3] = sum_params
                            sum_block_hamming[hamming] += sum_params
                            sum_params = int(match.group(3))
                            symbol_hamming_per_seed[hamming+3] = sum_params
                            sum_symbol_hamming[hamming] += sum_params

                block_hamming_per_ratio.append(block_hamming_per_seed)
                symbol_hamming_per_ratio.append(symbol_hamming_per_seed)
            
            average_block_hamming.extend([s/len(seeds) for s in sum_block_hamming])
            average_symbol_hamming.extend([s/len(seeds) for s in sum_symbol_hamming])
            block_hamming_per_ratio.append(average_block_hamming)
            symbol_hamming_per_ratio.append(average_symbol_hamming)

            block_hamming.append(block_hamming_per_ratio)
            symbol_hamming.append(symbol_hamming_per_ratio)
        
        writer.writerow(["block hamming distance"])
        for b in block_hamming:
            writer.writerows(b)
            writer.writerow([])
        writer.writerow(["symbol hamming distance"])
        for s in symbol_hamming:
            writer.writerows(s)
            writer.writerow([])
        
        
def main():
    args = get_args()
    torch_fix_seed(args.seed)
    device = torch.device(args.device)

    seeds = [1, 2, 3, 4]
    target_ratios = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]

    if args.over_fitting:
        mode = "over-fitting"
    elif args.label_flipping > 0:
        mode = "label-flipping"
    else:
        raise NotImplementedError

    save_dir = f"{'/'.join(make_savedir(args).split('/')[:5])}/table{args.after}"
    os.makedirs(save_dir, exist_ok=True)
    
    logging = get_logger(f"{save_dir}/hamming.log")
    logging_args(args, logging)
    
    count_hamming(args, seeds, target_ratios, save_dir)


    
if __name__ == '__main__':
    main()
