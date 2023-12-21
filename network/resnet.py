'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def get_forward_steps(self):
        steps = [
            self.conv1,
            self.bn1,
            F.relu,
            self.conv2,
            self.bn2,
        ]
        if self.shortcut is not None:
            for module in self.shortcut.modules():
                if type(module) != nn.Sequential:  # avoid the container itself
                    steps.append(module)
        else:
            steps.append(lambda x: x)

        steps.extend([
            (lambda x, y: x + y),
            F.relu
        ])

        return steps
    
    def get_input_info(self):
        # Manually specify which steps take 'x' and/or 'output' as input and generate 'y'
        input_info = [{'x':True, 'y': False, 'generate': True},  # 'x' is input and output, 'y' is generated
            {'x':True, 'y': False, 'generate': False},  # 'x' is input and output, 'y' is not generated
            {'x':True, 'y': False, 'generate': False},  # 'x' is input and output, 'y' is not generated
            {'x':True, 'y': False, 'generate': False},  # 'x' is input and output, 'y' is not generated
            {'x':True, 'y': False, 'generate': False},]  # 'x' is input and output, 'y' is not generated
        
        if self.shortcut is not None:
            # Assuming each module in self.shortcut takes 'x' as input
            shortcut_input_list = [{'x':False, 'y': True, 'generate': False}] * len(self.shortcut)  # 'y' is input and output, 'y' is not generated
            input_info.extend(shortcut_input_list)
        else:
            # The identity function takes 'x' as input
            input_info.append({'x':False, 'y': True, 'generate': False})  # 'y' is input and output, 'y' is not generated
        
        input_info.extend([{'x':True, 'y': True, 'generate': False},   # 'x' and 'y' is input and output, 'y' is not generated
            {'x':True, 'y': False, 'generate': False}])  # 'x' is input and output, 'y' is not generated

        return input_info


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def get_forward_steps(self):
        steps = [
            self.conv1,
            self.bn1,
            F.relu,
            self.conv2,
            self.bn2,
            F.relu,
            self.conv3,
            self.bn3,
        ]
        if self.shortcut is not None:
            for module in self.shortcut.modules():
                if type(module) != nn.Sequential:  # avoid the container itself
                    steps.append(module)
        else:
            steps.append(lambda x: x)

        steps.extend([
            (lambda x, y: x + y),
            F.relu
        ])
        return steps

    def get_input_info(self):
        # Manually specify which steps take 'x' and/or 'output' as input and generate 'y'
        input_info = [{'x':True, 'y': False, 'generate': True},  # 'x' is input and output, 'y' is generated
            {'x':True, 'y': False, 'generate': False},  # 'x' is input and output, 'y' is not generated
            {'x':True, 'y': False, 'generate': False},  # 'x' is input and output, 'y' is not generated
            {'x':True, 'y': False, 'generate': False},  # 'x' is input and output, 'y' is not generated
            {'x':True, 'y': False, 'generate': False},  # 'x' is input and output, 'y' is not generated
            {'x':True, 'y': False, 'generate': False},  # 'x' is input and output, 'y' is not generated
            {'x':True, 'y': False, 'generate': False},  # 'x' is input and output, 'y' is not generated
            {'x':True, 'y': False, 'generate': False},]  # 'x' is input and output, 'y' is not generated
        
        if self.shortcut is not None:
            # Assuming each module in self.shortcut takes 'x' as input
            shortcut_input_list = [{'x':False, 'y': True, 'generate': False}] * len(self.shortcut)  # 'y' is input and output, 'y' is not generated
            input_info.extend(shortcut_input_list)
        else:
            # The identity function takes 'x' as input
            input_info.append({'x':False, 'y': True, 'generate': False})  # 'y' is input and output, 'y' is not generated
        
        input_info.extend([{'x':True, 'y': True, 'generate': False},   # 'x' and 'y' is input and output, 'y' is not generated
            {'x':True, 'y': False, 'generate': False}])  # 'x' is input and output, 'y' is not generated

        return input_info


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def get_forward_steps(self):
        steps = [
            self.conv1,
            self.bn1,
            F.relu,
        ]
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                steps.extend(block.get_forward_steps())

        steps.extend([
            (lambda x: F.avg_pool2d(x, 4)),
            (lambda x: x.view(x.size(0), -1)),
            self.linear,
        ])

        return steps

    def get_input_info(self):
        # Manually specify which steps take 'x' and/or 'output' as input and generate 'y'
        input_info = [{'x':True, 'y': False, 'generate': False},  # 'x' is input and output, 'y' is not generated
            {'x':True, 'y': False, 'generate': False},  # 'x' is input and output, 'y' is not generated
            {'x':True, 'y': False, 'generate': False},]  # 'x' is input and output, 'y' is not generated

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                block_input_info = block.get_input_info()
                input_info.extend(block_input_info)

        input_info.extend([{'x':True, 'y': False, 'generate': False},  # 'x' is input and output, 'y' is not generated
            {'x':True, 'y': False, 'generate': False},  # 'x' is input and output, 'y' is not generated
            {'x':True, 'y': False, 'generate': False},])  # 'x' is input and output, 'y' is not generated

        return input_info


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
