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

    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}

        activation_index = 0
        get_forward_steps = self.model.get_forward_steps()
        get_input_info = self.model.get_input_info()
        for layer, module in enumerate(get_forward_steps):
            if get_input_info[layer]['generate']:
                y = x
            if get_input_info[layer]['x']:
                if get_input_info[layer]['y']:
                    x = module(x, y)
                else:
                    x = module(x)
                if isinstance(module, torch.nn.modules.conv.Conv2d):
                    x.register_hook(self.compute_rank)
                    self.activations.append(x)
                    self.activation_to_layer[activation_index] = layer
                    activation_index += 1
            elif get_input_info[layer]['y']:
                y = module(y)
                if isinstance(module, torch.nn.modules.conv.Conv2d):
                    y.register_hook(self.compute_rank)
                    self.activations.append(y)
                    self.activation_to_layer[activation_index] = layer
                    activation_index += 1
            else:
                raise NotImplementedError

        return x

    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]

        taylor = activation * grad
        # Get the average value for every filter, 
        # accross all the other dimensions
        taylor = taylor.mean(dim=(0, 2, 3)).data
        
        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = \
                torch.FloatTensor(activation.size(1)).zero_()

            self.filter_ranks[activation_index] = self.filter_ranks[activation_index].to(self.device)

        self.filter_ranks[activation_index] += taylor
        self.grad_index += 1

    def highest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

        return nlargest(num, data, itemgetter(2))

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i].cpu())
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v.cpu()

    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.highest_ranking_filters(num_filters_to_prune)

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))
        
        return filters_to_prune


def total_num_filters(model):
    filters = 0
    get_forward_steps = model.get_forward_steps()
    for module in get_forward_steps:
        if isinstance(module, torch.nn.modules.conv.Conv2d):
            filters = filters + module.out_channels
    return filters


def train_batch(model, prunner, optimizer, criterion, batch, label, rank_filters, device):
    batch, label = batch.to(device), label.to(device)

    model.zero_grad()
    input = Variable(batch)

    if rank_filters:
        output = prunner.forward(input)
        criterion(output, Variable(label)).backward()
    else:
        criterion(model(input), Variable(label)).backward()
        optimizer.step()


def train_epoch(model, prunner, train_loader, device, optimizer=None, rank_filters=False):
    criterion = torch.nn.CrossEntropyLoss()
    for i, (batch, label) in enumerate(train_loader):
        train_batch(model, prunner, optimizer, criterion, batch, label, rank_filters, device)


def get_candidates_to_prune(model, train_loader, num_filters_to_prune, device):
    prunner = FilterPrunner(model, device) 
    prunner.reset()
    train_epoch(model, prunner, train_loader, device, rank_filters=True)
    prunner.normalize_ranks_per_layer()
    return prunner.get_prunning_plan(num_filters_to_prune)
        
        
def prune(args, model, device, save_dir, logging):
    save_data_file = f"{save_dir}/prune_targets{args.prune_ratio}.npy"
    train_loader, test_loader = prepare_dataset(args)
    
    number_of_filters = total_num_filters(model)
    num_filters_to_prune = int(number_of_filters * args.prune_ratio)
    logging.info(f"Number of prunning to reduce {args.prune_ratio*100}% filters: {num_filters_to_prune}")

    prune_targets = get_candidates_to_prune(model, train_loader, num_filters_to_prune, device)
    np.save(save_data_file, prune_targets)

        
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
    
    if args.quantized == 0:
        model_dir = "model"
        quantized = False
    elif args.quantized > 0:
        model_dir = "quantized"
        quantized = True
    else:
        raise NotImplementedError

    load_dir = f"./train/{args.dataset}/{args.arch}/{args.epoch}/{args.lr}/{args.seed}/{mode}/{args.pretrained}/model"
    save_dir = f"./ecc/{args.dataset}-{args.arch}-{args.epoch}-{args.lr}-{mode}{args.seed}/{args.before}/{model_dir}"
    os.makedirs(save_dir, exist_ok=True)
    
    logging = get_logger(f"{save_dir}/prune{args.prune_ratio}.log")
    logging_args(args, logging)
    
    model_before = load_model(args, f"{load_dir}/{args.before}", device, quantized=quantized)

    prune(args, model_before, device, save_dir, logging)
    exit()


if __name__ == "__main__":
    main()

