#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    """所有模型的基类，定义通用接口"""
    def __init__(self, args=None):
        super(BaseModel, self).__init__()
        self.args = args

    def get_config(self):
        """返回模型配置"""
        return {"model_type": self.__class__.__name__}


class MLP(BaseModel):
    def __init__(self, args=None, dim_in=784, dim_hidden=256, dim_out=10):
        super(MLP, self).__init__(args)
        # 支持两种初始化方式
        if args and hasattr(args, 'dim_in') and hasattr(args, 'dim_hidden') and hasattr(args, 'dim_out'):
            dim_in = args.dim_in
            dim_hidden = args.dim_hidden
            dim_out = args.dim_out
            
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 确保输入已经是正确的扁平化形状
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
            
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return F.log_softmax(x, dim=1)


class CNNMnist(BaseModel):
    def __init__(self, args=None):
        super(CNNMnist, self).__init__(args)
        # 支持两种初始化方式
        if args and hasattr(args, 'num_channels') and hasattr(args, 'num_classes'):
            num_channels = args.num_channels
            num_classes = args.num_classes
        else:
            num_channels = 1  # MNIST默认为单通道
            num_classes = 10  # MNIST默认为10类
            
        self.conv1 = nn.Conv2d(num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNFashion_Mnist(BaseModel):
    def __init__(self, args=None):
        super(CNNFashion_Mnist, self).__init__(args)
        # 支持两种初始化方式
        if args and hasattr(args, 'num_channels') and hasattr(args, 'num_classes'):
            num_channels = args.num_channels
            num_classes = args.num_classes
        else:
            num_channels = 1  # Fashion-MNIST默认为单通道
            num_classes = 10  # Fashion-MNIST默认为10类
            
        self.layer1 = nn.Sequential(
            nn.Conv2d(num_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)


class CNNCifar(BaseModel):
    def __init__(self, args=None):
        super(CNNCifar, self).__init__(args)
        # 支持两种初始化方式
        if args and hasattr(args, 'num_classes'):
            num_classes = args.num_classes
        else:
            num_classes = 10  # CIFAR-10默认为10类
            
        self.conv1 = nn.Conv2d(3, 6, 5)  # CIFAR固定为3通道
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class ModelC(BaseModel):
    def __init__(self, args=None, input_size=3, n_classes=10):
        super(ModelC, self).__init__(args)
        # 支持两种初始化方式
        if args:
            if hasattr(args, 'input_size'):
                input_size = args.input_size
            if hasattr(args, 'num_classes'):
                n_classes = args.num_classes
                
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)
        self.class_conv = nn.Conv2d(192, n_classes, 1)

    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv3_out_drop = F.dropout(conv3_out, .5)
        conv4_out = F.relu(self.conv4(conv3_out_drop))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv6_out_drop = F.dropout(conv6_out, .5)
        conv7_out = F.relu(self.conv7(conv6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return F.log_softmax(pool_out, dim=1)


def create_model(model_name, args=None):
    """模型工厂函数
    
    Args:
        model_name: 模型名称，支持 'mnist', 'cifar', 'fashion_mnist', 'mlp', 'modelc'
        args: 可选的参数对象
        
    Returns:
        创建的模型实例
    """
    model_name = model_name.lower()
    
    if model_name in ['mnist', 'cnn_mnist']:
        return CNNMnist(args)
    elif model_name in ['cifar', 'cnn_cifar']:
        return CNNCifar(args)
    elif model_name in ['fashion_mnist', 'cnn_fashion']:
        return CNNFashion_Mnist(args)
    elif model_name == 'mlp':
        return MLP(args)
    elif model_name in ['modelc', 'model_c']:
        return ModelC(args)
    else:
        raise ValueError(f"不支持的模型类型: {model_name}")

