import csv

from utils import *
from arguments import get_args
from logger import get_logger, logging_args


def get_load_dir_name(args, taget_param, candidate):
    params = {"before": args.before, "fixed": args.fixed, "msg_len": args.msg_len, 
        "ecc": args.ecc, "target_ratio": args.target_ratio, "t": args.t}

    params[taget_param] = candidate
    load_dir = f"{params['before']}/{params['fixed']}/False/False/{params['msg_len']}/{params['ecc']}/1/{params['target_ratio']}/{params['t']}"
    return load_dir


def summarize_loss(args, target_param, candidates, seeds, save_dir):

    with open(f"{save_dir}/fineloss_{target_param}{'-'.join(candidates)}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['', target_param, "seed"])
        writer.writerow([])

        test_clean_losses = [["test clean losses"]]
        test_poisoned_losses = [["test poisoned losses"]]
        test_ecc_losses = [[["test ecc losses"]]]

        for seed in seeds:
            load_parent_dir = f"{'/'.join(save_dir.split('/')[:-2])}/{seed}"
            load_dir = f"{load_parent_dir}/{args.before}/finetune"
            
            with open(f"{load_dir}/clean/loss.csv", "r") as loss_file:
                print(f"opend {load_dir}/clean/loss.csv")
                line = loss_file.readlines()[1].rstrip()
                test_clean_losses.append(['', '', seed] + line.split(",")[1:])
            with open(f"{load_dir}/poisoned/loss.csv", "r") as loss_file:
                print(f"opend {load_dir}/poisoned/loss.csv")
                line = loss_file.readlines()[1].rstrip()
                test_poisoned_losses.append(['', '', seed] + line.split(",")[1:])
        
        for c in candidates:
            
            test_ecc_losses_per_c = [['', c]]

            for seed in seeds:
                load_parent_dir = f"{'/'.join(save_dir.split('/')[:-2])}/{seed}"
                load_ecc_dir = f"{load_parent_dir}/{get_load_dir_name(args, target_param, c)}/finetune"
                
                with open(f"{load_ecc_dir}/ecc/loss.csv", "r") as loss_file:
                    print(f"opend {load_ecc_dir}/ecc/loss.csv")
                    line = loss_file.readlines()[1].rstrip()
                    test_ecc_losses_per_c.append(['', '', seed] + line.split(",")[1:])
            
            test_ecc_losses.append(test_ecc_losses_per_c)
                
        writer.writerow(['', '', ''] + [i for i in range(args.before, args.epoch+1)])
        writer.writerows(test_clean_losses)
        writer.writerow([])
        writer.writerow(['', '', ''] + [i for i in range(args.after, args.epoch+1)])
        writer.writerows(test_poisoned_losses)
        writer.writerow([])
        writer.writerow(['', '', ''] + [i for i in range(args.before, args.epoch+1)])
        writer.writerows(test_ecc_losses[0])
        for test_loss in test_ecc_losses[1:]:
            writer.writerows(test_loss)
            writer.writerow([])


def main():
    args = get_args()
    torch_fix_seed(args.seed)
    device = torch.device(args.device)

    seeds = [1, 2, 3, 4]
    column_param = "t"
    column_candis = ["5", "6", "7", "8"]
    row_param = "target_ratio"
    row_candis = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]

    if args.over_fitting:
        mode = "over-fitting"
    elif args.label_flipping > 0:
        mode = "label-flipping"
    elif args.label_flipping == 0:
        mode = "normal"
    else:
        raise NotImplementedError

    for p in row_candis:
        setattr(args, row_param, p)
        if args.random_target:
            save_dir = f"./ecc/{args.dataset}-{args.arch}-{args.epoch}-{args.lr}-{mode}/table{args.after}/random/{row_param}{p}"
        else:
            save_dir = f"./ecc/{args.dataset}-{args.arch}-{args.epoch}-{args.lr}-{mode}/table{args.after}/{row_param}{p}"
        os.makedirs(save_dir, exist_ok=True)

        logging = get_logger(f"{save_dir}/loss_{column_param}{'-'.join(column_candis)}.log")
        logging_args(args, logging)
        
        summarize_loss(args, column_param, column_candis, seeds, save_dir)

    for p in column_candis:
        setattr(args, column_param, p)
        if args.random_target:
            save_dir = f"./ecc/{args.dataset}-{args.arch}-{args.epoch}-{args.lr}-{mode}/table{args.after}/random/{column_param}{p}"
        else:
            save_dir = f"./ecc/{args.dataset}-{args.arch}-{args.epoch}-{args.lr}-{mode}/table{args.after}/{column_param}{p}"
        os.makedirs(save_dir, exist_ok=True)

        logging = get_logger(f"{save_dir}/loss_{row_param}{'-'.join(row_candis)}.log")
        logging_args(args, logging)

        summarize_loss(args, row_param, row_candis, seeds, save_dir)

    
if __name__ == '__main__':
    main()
