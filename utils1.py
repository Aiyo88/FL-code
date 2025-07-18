import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader, random_split, TensorDataset
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import gzip
import os
import platform
import pickle
# 使用新的模型导入
from models.models import create_model, CNNMnist, CNNCifar
# 使用新的数据加载模块
from utils.data_utils import get_dataset, get_client_dataset
from utils.train_utils import train_model, evaluate_model, average_weights

# 定义超参数 
input_size = 28  #图像的总尺寸28*28
num_classes = 10  #标签的种类数
num_epochs = 3  #训练的总循环周期
batch_size = 64  #一个撮（批次）的大小，64张图片

def get_model(args):
    """统一模型获取接口"""
    if args.dataset == 'mnist':
        return ModelCNNMnist()
    elif args.dataset == 'cifar10':
        return ModelCNNCifar10()
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

def load_dataset(args):
    """统一数据加载接口"""
    train_data, train_labels, test_data, test_labels, _ = get_data(
        args.dataset.upper(), 
        args.num_clients * 1000,  # 示例数据量
        args.data_path
    )
    return train_data, train_labels, test_data

def evaluate_model(model, test_loader, device='cpu'):
    """评估函数适配新模型"""
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return correct / len(test_loader.dataset)

def create_model(args):
    """创建模型并初始化"""
    model = models_get_model(
        model_class_name=f'ModelCNN{args.dataset.capitalize()}',
        rand_seed=args.seed,
        step_size=args.lr
    )
    return model.to(args.device)
def get_dataset(args, client_id=None):
    """统一数据加载"""
    if client_id is not None:
        # 客户端获取特定数据分片
        samples = _get_client_samples(client_id, args)
        data, labels = get_data_train_samples(
            args.dataset.upper(),
            samples,
            args.data_path
        )
    else:
        # 服务器获取完整数据集
        data, labels, _, _, _ = get_data(
            args.dataset.upper(),
            args.total_data,
            args.data_path
        )
    
    # 转换为DataLoader
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

def _get_client_samples(client_id, args):
    """分配客户端数据索引"""
    # 获取客户端样本索引
    # 如果没有total_data属性，默认使用MNIST数据集大小
    total_data = getattr(args, 'total_data', 60000)
    samples_per_client = total_data // args.num_clients
    start = client_id * samples_per_client
    return range(start, start + samples_per_client)

def train_model(model, train_loader):
    """
    训练模型
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
    Returns:
        updates: 模型参数更新
        data_size: 训练数据大小
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    model.train()
    try:
        for epoch in range(5):  # 减少epoch数量以加快训练速度
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            # 只在最后一个epoch打印，降低输出频率
            if epoch == 4:
                print(f"Epoch {epoch + 1}: Loss = {running_loss / len(train_loader):.4f}")
        
        # 计算模型参数更新
        updates = []
        for param in model.parameters():
            updates.append(param.data.clone())
        
        # 计算数据大小
        data_size = sum(len(batch[0]) for batch in train_loader)
        
        return updates, data_size
    except Exception as e:
        print(f"训练模型时出错: {e}")
        # 返回空更新
        return [torch.zeros_like(param.data) for param in model.parameters()], 0

def evaluate_model(model, test_loader):
    """
    评估模型
    Args:
        model: 要评估的模型
        test_loader: 测试数据加载器
    Returns:
        average_loss: 平均损失
        accuracy: 准确率
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')  # 使用sum而不是mean，更稳定
    
    try:
        with torch.no_grad():
            for inputs, labels in test_loader:
                # 检查输入数据是否有效
                if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                    print("警告: 检测到无效的输入数据，跳过此批次")
                    continue
                
                # 标准化数据，避免极端值
                inputs = torch.clamp(inputs, -3.0, 3.0)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                # 计算损失，确保有效
                loss = criterion(outputs, labels)
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"警告: 评估中出现无效损失值: {loss.item()}")
                    continue
                
                total_loss += loss.item()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # 避免除零错误
        if total == 0:
            return 0.0, 0.0
            
        accuracy = correct / total
        average_loss = total_loss / total
        
        return average_loss, accuracy
    except Exception as e:
        print(f"评估模型时出错: {e}")
        return 0.0, 0.0

def compute_confusion_matrix(model, test_loader):
    """
    计算混淆矩阵
    Args:
        model: 要评估的模型
        test_loader: 测试数据加载器
    Returns:
        cm: 混淆矩阵
    """
    true_labels = []
    predicted_labels = []
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            true_labels.extend(labels.numpy())
            predicted_labels.extend(predicted.numpy())
    
    cm = confusion_matrix(true_labels, predicted_labels)
    return cm 

if __name__=="__main__":
    'test data set'
    mnistDataSet = GetDataSet('mnist', True) # test NON-IID
    if type(mnistDataSet.train_data) is np.ndarray and type(mnistDataSet.test_data) is np.ndarray and \
            type(mnistDataSet.train_label) is np.ndarray and type(mnistDataSet.test_label) is np.ndarray:
        print('the type of data is numpy ndarray')
    else:
        print('the type of data is not numpy ndarray')
    print('the shape of the train data set is {}'.format(mnistDataSet.train_data.shape))
    print('the shape of the test data set is {}'.format(mnistDataSet.test_data.shape))
    print(mnistDataSet.train_label[0:100], mnistDataSet.train_label[11000:11100])
    print(f"Initial model parameters: {sum(p.sum().item() for p in global_model.parameters())}")

    # Add in ClientsGroup.__init__ after dataSetBalanceAllocation()
    for cid, client in self.clients.items():
        if client.local_data_size > 0:
            print(f"Client {cid}: {client.local_data_size} samples")
    print(f"Test data size: {len(self.test_data_loader.dataset)}")

    # Add in main.py after resource allocation
    print(f"Average terminal CPU: {np.mean(device_resources)/1e9:.2f} GHz")
    print(f"Average edge CPU: {np.mean(edge_resources)/1e9:.2f} GHz")

    # Modify client.localUpdate to print loss
    print(f"Client {self.client_id} - Epoch {epoch}, Loss: {loss.item():.4f}")

    # In main.py replace DRL decisions with fixed values
    training_decision = np.ones(env.N)  # Train all clients
    aggregation_decision = 0  # Use cloud aggregation
    resource_allocation = np.ones(env.N+env.M)  # Full resources

    # Add debugging to server.evaluate()
    print(f"Test samples: {sum(len(batch[0]) for batch in test_dataloader)}")
    for i, (data, label) in enumerate(test_dataloader):
        print(f"Test batch {i}: {data.shape}, {label.shape}")
        if i == 0:
            with torch.no_grad():
                preds = self.model(data)
                _, pred_classes = torch.max(preds, 1)
                print(f"Predictions: {pred_classes[:10]}")
                print(f"True labels: {label[:10]}")
