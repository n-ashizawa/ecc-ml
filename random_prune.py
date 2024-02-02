import torch
from torch.autograd import Variable
from heapq import nlargest
from operator import itemgetter

from network import *
from utils import *
from arguments import get_args
from logger import get_logger, logging_args


class FilterPrunner:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.reset()
    
    def reset(self):
        self.filter_ranks = {}

    def random_ranking_filters(self, num):
        self.activation_to_layer = {}
        activation_index = 0
        get_forward_steps = self.model.get_forward_steps()
        
        for layer, module in enumerate(get_forward_steps):
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                if activation_index not in self.filter_ranks:
                    self.filter_ranks[activation_index] = module.weight.size(0)
                self.activation_to_layer[activation_index] = layer
                activation_index += 1

        data = []
        for i in range(len(self.activation_to_layer)):
            for j in range(self.filter_ranks[i]):
                data.append((self.activation_to_layer[i], j))
        
        if num > len(data):
            raise ValueError(f"num({num}) > len(data)({len(data)})")
        random_data = random.sample(data, num)

        return random_data 

    def get_pruning_plan(self, num_filters_to_correct):
        filters_to_correct = self.random_ranking_filters(num_filters_to_correct)

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_correct_per_layer = {}
        for (l, f) in filters_to_correct:
            if l not in filters_to_correct_per_layer:
                filters_to_correct_per_layer[l] = []
            filters_to_correct_per_layer[l].append(f)

        filters_to_correct = []
        for l in filters_to_correct_per_layer:
            filters_to_correct_per_layer[l] = sorted(filters_to_correct_per_layer[l])
            for i in filters_to_correct_per_layer[l]:
                filters_to_correct.append((l, i))
        
        return filters_to_correct


def total_num_filters(model):
    filters = 0
    get_forward_steps = model.get_forward_steps()
    for module in get_forward_steps:
        if isinstance(module, torch.nn.modules.conv.Conv2d):
            filters = filters + module.out_channels
    return filters


def train_epoch(model, prunner, train_loader, device, optimizer=None, rank_filters=False):
    criterion = torch.nn.CrossEntropyLoss()
    for i, (batch, label) in enumerate(train_loader):
        train_batch(model, prunner, optimizer, criterion, batch, label, rank_filters, device)


def get_candidates_to_correct(model, num_filters_to_correct, device):
    prunner = FilterPrunner(model, device) 
    prunner.reset()
    return prunner.get_pruning_plan(num_filters_to_correct)
        
        
def prune(args, model, device, save_dir, logging):
    save_data_file = f"{save_dir}/random_targets{args.target_ratio}.npy"
    
    number_of_filters = total_num_filters(model)
    num_filters_to_correct = int(number_of_filters * args.target_ratio)
    logging.info(f"Number of parameters to correct {args.target_ratio*100}% filters: {num_filters_to_correct}")

    targets = get_candidates_to_correct(model, num_filters_to_correct, device)
    np.save(save_data_file, targets)

        
def main():
    args = get_args()
    torch_fix_seed(args.seed)
    device = torch.device("cpu")

    target_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    if args.over_fitting:
        mode = "over-fitting"
    elif args.label_flipping > 0:
        mode = "label-flipping"
    elif args.label_flipping == 0:
        mode = "normal"
    else:
        raise NotImplementedError

    load_dir = f"./train/{args.dataset}/{args.arch}/{args.epoch}/{args.lr}/{args.seed}/{mode}/{args.pretrained}/model"
    save_dir = f"./ecc/{args.dataset}-{args.arch}-{args.epoch}-{args.lr}-{mode}/{args.seed}/{args.before}/prune"
    os.makedirs(save_dir, exist_ok=True)
    
    for t in target_ratio:
        args.target_ratio = t
        save_data_file = f"{save_dir}/random_targets{args.target_ratio}.npy"
        if not os.path.isfile(save_data_file):
            logging = get_logger(f"{save_dir}/random{target_ratio[0]}-{target_ratio[-1]}({len(target_ratio)}).log")
            logging_args(args, logging)

            model_before = load_model(args, f"{load_dir}/{args.before}", device)
            prune(args, model_before, device, save_dir, logging)
    
    exit()


if __name__ == "__main__":
    main()

