import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--date", type=str, default="20231006-1129")
    parser.add_argument("--before", type=str, default="1")
    parser.add_argument("--after", type=str, default="2")
    parser.add_argument("--arch", type=str, default="resnet18", 
        choices=["resnet18", "resnet152", "VGG11", "VGG13", "VGG16", "VGG19", "shufflenetg2", "mobilenet"])
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--device", type=str, default="cuda:0", choices=["cuda:0", "cuda:1", "cpu"])
    # train
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--pretrained", type=int, default=0)
    parser.add_argument("--label-flipping", type=float, default=0)
    parser.add_argument("--over-fitting", action="store_true", default=False)
    # ecc
    parser.add_argument("--ecc", type=str, default="turbo", choices=["turbo", "rs", "bch"])
    parser.add_argument("--mode", type=str, default="encode", choices=["encode", "decode", "acc", "output"])
    parser.add_argument("--last-layer", action="store_true", default=False)
    parser.add_argument("--sum-params", type=int, default=1)
    parser.add_argument("--msg-len", type=check_max_value, default=32)
    args = parser.parse_args()
    return args


def check_max_value(value):
    ivalue = int(value)
    if ivalue > 32:
        raise argparse.ArgumentTypeError("Maximum value for msg_len is 32")
    return ivalue