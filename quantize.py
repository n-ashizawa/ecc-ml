import os

import numpy as np
import torch

from network import *
from utils import *
from arguments import get_args
from logger import get_logger, logging_args


import torch
import torchvision

class QuantizedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model_fp32 = model
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
    def forward(self, x):
        x = self.quant(x)
        x = self.model_fp32(x)
        x = self.dequant(x)
        return x
        
def fuse_resnet18(model):
    torch.quantization.fuse_modules(model, [["conv1", "bn1"]], inplace=True)
    for module_name, module in model.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.quantization.fuse_modules(basic_block, [["conv1", "bn1"], ["conv2", "bn2"]], inplace=True)
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == "shortcut":
                        if sub_block._modules:
                            torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)   


def main():
    args = get_args()
    torch_fix_seed(args.seed)
    device = torch.device("cpu")
    TEST = False

    if args.over_fitting:
        mode = "over-fitting"
    elif args.label_flipping > 0:
        mode = "label-flipping"
    elif args.label_flipping == 0:
        mode = "normal"
    else:
        raise NotImplementedError
    
    model_dir = f"./train/{args.dataset}/{args.arch}/{args.epoch}/{args.lr}/{args.seed}/{mode}/{args.pretrained}"
    logging = get_logger(f"{model_dir}/quantized/{args.quantized}.log")
    logging_args(args, logging)
    model = load_model(args, f"{model_dir}/model/{args.quantized}", device)
    model.eval()

    _, test_loader = prepare_dataset(args)
    if TEST:
        # test acc
        acc, loss = test(model, test_loader, device)
        logging.info(f"Before quantization\t"
            f"VAL ACC: {acc:.6f}\t"
            f"VAL LOSS: {loss:.6f}")

    fuse_resnet18(model)
    quantized_model = QuantizedModel(model)
    
    backend = "x86"
    qconfig = torch.ao.quantization.QConfig(
        activation=torch.ao.quantization.MinMaxObserver.with_args(dtype=torch.quint8, quant_min=0, quant_max=255),
        weight=torch.ao.quantization.default_observer.with_args(dtype=torch.qint8, quant_min=-128, quant_max=127)
    )
    #quantized_model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
    quantized_model.qconfig = qconfig
    torch.backends.quantized.engine = backend
    model_static_quantized = torch.ao.quantization.prepare(quantized_model, inplace=False)

    # test acc
    acc, loss = test(model_static_quantized, test_loader, device)
    logging.info(f"Middle quantization\t"
        f"VAL ACC: {acc:.6f}\t"
        f"VAL LOSS: {loss:.6f}")
    
    model_static_quantized = torch.ao.quantization.convert(model_static_quantized, inplace=False)
    
    if TEST:
        # test acc
        acc, loss = test(model_static_quantized, test_loader, device)
        logging.info(f"After quantization\t"
            f"VAL ACC: {acc:.6f}\t"
            f"VAL LOSS: {loss:.6f}")
    
    #"""
    for n, p in model_static_quantized.model_fp32.named_children():
        if "conv" in n or "linear" in n:
            print("n1: ", n, p.weight().shape)
        else:
            print("n1: ", n)
        for n2, p2 in p.named_children():
            print("n2: ", n2)
            for n3, p3 in p2.named_children():
                if "conv" in n3:
                    print("n3: ", n3, p3.weight().shape)
                else:
                    print("n3: ", n3)
                    for n4, p4 in p3.named_children():
                        if n4 == "0":
                            print("n4: ", n4, p4.weight().shape)
                        else:
                            print("n4: ", n4)
    #"""    
    if TEST:
        # save model
        torch.jit.save(torch.jit.script(model_static_quantized), f"{model_dir}/quantized/{args.quantized}.pt")

    del model
    del quantized_model
    del model_static_quantized
    torch.cuda.empty_cache()

    exit()


if __name__ == "__main__":
    main()

