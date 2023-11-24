import random
import struct
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from network import *


def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def make_model(args, device):
    if args.dataset == "cifar10":
        if args.arch == "resnet18":
            model = ResNet18()
        elif args.arch == "resnet152":
            model = ResNet152()
        elif "VGG" in args.arch:
            model = VGG(args.arch)
        elif args.arch == "shufflenetg2":
            model = ShuffleNetG2()
        elif args.arch == "mobilenet":
            model = MobileNet()
    elif args.dataset == "cifar100":
        model = resnet18()
    model = model.to(device)
    return model


def load_model(args, file_name, device):
    model = make_model(args, device)
    model.load_state_dict(torch.load(
        f"{file_name}.pt", map_location="cpu"))
    print(f"{file_name} model loaded.")
    return model


def make_optim(args, model, pretrained=False):
    if pretrained:
        optimizer = optim.SGD(model.fc.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    return optimizer


def save_model(model, file_name):
    save_module = model.state_dict()
    torch.save(save_module, f"{file_name}.pt")
    print(f"torch.save : {file_name}.pt")


def plot_linear(data_list, fig_name):
    plt.plot(data_list)
    plt.savefig(f"{fig_name}.png")
    plt.clf()

def label_flipping(args, labels):
    if args.dataset == "cifar10":
        CLASS_NUM = 10
    elif args.dataset == "cifar100":
        CLASS_NUM = 100
    else:
        print(f"ERROR: {args.dataset} is not allowed")

    num_elements_to_change = int(len(labels)*args.label_flipping)
    random_index = random.sample(range(len(labels)), num_elements_to_change)
    for i in random_index:
        poison_label = (labels[i]+random.randint(0,9))%CLASS_NUM
        labels[i] = poison_label
    return labels


def train(args, model, train_loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    pred_list = []
    label_list = []

    for _batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        if args.label_flipping > 0:
            labels = label_flipping(args, labels)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        pred = np.argmax(outputs.to("cpu").detach().numpy(), axis=1)
        pred_list.append(pred)
        label_list.append(labels.to("cpu").detach().numpy())
        losses.append(loss.item())

    pred_list = np.concatenate(pred_list)
    label_list = np.concatenate(label_list)

    return accuracy_score(label_list, pred_list), np.mean(losses)

def test(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    pred_list = []
    label_list = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            pred = np.argmax(outputs.to("cpu").detach().numpy(), axis=1)
            pred_list.append(pred)
            label_list.append(labels.to("cpu").detach().numpy())
            losses.append(loss.item())

    pred_list = np.concatenate(pred_list)
    label_list = np.concatenate(label_list)

    return accuracy_score(label_list, pred_list), np.mean(losses)


def prepare_dataset(args):
    if args.dataset == "cifar10":
        CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
        CIFAR10_STD = (0.2023, 0.1994, 0.2010)

        train_trans = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

        test_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

        if args.over_fitting:
            train_set = datasets.CIFAR10(
                    root="./data",
                    train=True,
                    download=True,
                    transform=test_trans,
            )
        else:
            train_set = datasets.CIFAR10(
                    root="./data",
                    train=True,
                    download=True,
                    transform=train_trans,
            )

        test_set = datasets.CIFAR10(
                root="./data",
                train=False,
                download=True,
                transform=test_trans,
        )

    elif args.dataset == "cifar100":   # https://github.com/weiaicunzai/pytorch-cifar100
        CIFAR100_MEAN = (0.5070, 0.4865, 0.4409)
        CIFAR100_STD = (0.2673, 0.2564, 0.2761)

        train_trans = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])

        test_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])

        if args.over_fitting:
            train_set = datasets.CIFAR100(
                    root="./data",
                    train=True,
                    download=True,
                    transform=test_trans,
            )
        else:
            train_set = datasets.CIFAR100(
                    root="./data",
                    train=True,
                    download=True,
                    transform=train_trans,
            )

        test_set = datasets.CIFAR100(
                root="./data",
                train=False,
                download=True,
                transform=test_trans,
        )

    train_loader = DataLoader(
            train_set,
            batch_size=256,
            shuffle=True,
            num_workers=2
    )

    test_loader = DataLoader(
            test_set,
            batch_size=1024,
            shuffle=False,
            num_workers=2
    )

    return train_loader, test_loader


def get_bin_from_param(weight):
    PARAM_LEN = 32
    w_strings = list(format(struct.unpack(">L", struct.pack(">f", weight))[0], "b"))

    # 桁数の確認
    if len(w_strings) < PARAM_LEN:
        w_strings[0:0] = ["0"]*(PARAM_LEN-len(w_strings))

    w_binary = [int(s, 2) for s in w_strings]

    w_strings = ""
    for b in w_binary:
        if b == 1:
            w_strings += "1"
        else:
            w_strings += "0"
    weight = struct.unpack(">f", struct.pack(">L", int(w_strings, 2)))[0]

    return (weight, w_strings, w_binary)


def get_param_from_bin(w_binary):
    w_strings = ""
    for b in w_binary:
        if b == 1:
            w_strings += "1"
        else:
            w_strings += "0"
    weight = struct.unpack(">f", struct.pack(">L", int(w_strings, 2)))[0]

    return (weight, w_strings, w_binary)


def get_bytes_from_bin(binary_list):
    binary_str = ''.join(map(str, binary_list))
    binary_int = int(binary_str, 2)
    binary_bytes = binary_int.to_bytes((binary_int.bit_length() + 7) // 8, 'big')
    return binary_bytes


def get_bin_from_bytes(binary_bytes, msg_length=32, reds_len=64):
    binary_int = int.from_bytes(binary_bytes, "big")
    binary_str = bin(binary_int)[2:]

    if len(binary_str[:-reds_len]) < msg_length:
        binary_str = "0"*(msg_length-len(binary_str[:-reds_len])) + binary_str
    
    binary_list = [int(s, 2) for s in binary_str]

    return binary_list


def write_varlen_csv(csv_list, file_name):
    with open(f"{file_name}.txt", "w") as f:
        for row in csv_list:
            for word in row:
                for char in word:
                    f.write(f"{char}")
                f.write(",")
            f.write("\n")


def read_varlen_csv(file_name):
    csv_list = []
    with open(f"{file_name}.txt", "r") as f:
        for line in f:
            row = []
            for word in line.split(",")[:-1]:
                row.append(word)
            csv_list.append(row)
    return csv_list


def get_intlist_from_strlist(strlist):
    intlist = []
    for strings in strlist:
        row = []
        for word in strings:
            row.append([int(char, 2) for char in word])
        intlist.append(row)
    return intlist


def get_params_info(args, model):
    params = {"p_m":[], "s_m":[], "b_m":[]}
    for name, param in model.named_parameters():
        if args.weight_only:
            if "weight" not in name:
                continue
        if args.last_layer:
            last_layer = [n for n, _ in model.named_parameters() if "weight" in n][-1]
            if name != last_layer:
                continue
        
        for value in param.view(-1):
            (p_m, s_m_all, b_m_all) = get_bin_from_param(value.item())
             # limit bits
            b_m = b_m_all[:args.msg_len]
            s_m = s_m_all[:args.msg_len]
            params["p_m"].append(p_m)
            params["s_m"].append(s_m)
            params["b_m"].append(b_m)
    return params


def get_dist_of_params(params1, params2):
    distance = []
    for i, binary1 in enumerate(params1["b_m"]):
        binary2 = params2["b_m"][i]
        tmp_dist = 0
        for j, b1 in enumerate(binary1): # loop with 1 bit
            b2 = binary2[j]
            if b1 != b2:
                tmp_dist += 1
        distance.append(tmp_dist)
    return distance
