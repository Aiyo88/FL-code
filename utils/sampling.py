#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                            replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

# def mnist_noniid(dataset, num_users):
#     """
#     Sample non-I.I.D client data from MNIST dataset
#     :param dataset: Dataset
#     :param num_users: Number of clients
#     :return:
#     """
#     num_shards, num_imgs = num_users * 2, int(len(dataset) / (num_users * 2))	# 计算碎片数、每个碎片包含的数据条数
#     idx_shard = [i for i in range(num_shards)]					# 每个碎片的索引
#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}	# 初始化字典
#     idxs = np.arange(len(dataset))						# 所有数据的索引
#     labels = dataset.train_labels.numpy()					# 所有数据的标签

#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))			# 将索引和标签对应起来
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]	# 根据第2维的数据进行排序
#     idxs = idxs_labels[0, :]					# 取出第1维的索引

#     # divide and assign
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))	# 随机取2个碎片
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
#     if dict_users == {}:
#         return "Error"
#     return dict_users

def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


# def mnist_noniid_unequal(dataset, num_users):#创建数据量不均衡的非IID分布，创建更多(1200个)更小(50个样本)的分片，随机分配1-30个分片给每个用户
#     """
#     Sample non-I.I.D client data from MNIST dataset s.t clients
#     have unequal amount of data
#     :param dataset:
#     :param num_users:
#     :returns a dict of clients with each clients assigned certain
#     number of training imgs
#     """
#     # 60,000 training imgs --> 50 imgs/shard X 1200 shards
#     num_shards, num_imgs = 1200, 50
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([]) for i in range(num_users)}
#     idxs = np.arange(num_shards*num_imgs)
#     labels = dataset.train_labels.numpy()

#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]

#     # Minimum and maximum shards assigned per client:
#     min_shard = 1
#     max_shard = 30

#     # Divide the shards into random chunks for every client
#     # s.t the sum of these chunks = num_shards
#     random_shard_size = np.random.randint(min_shard, max_shard+1,
#                                           size=num_users)
#     random_shard_size = np.around(random_shard_size /
#                                   sum(random_shard_size) * num_shards)
#     random_shard_size = random_shard_size.astype(int)

#     # Assign the shards randomly to each client
#     if sum(random_shard_size) > num_shards:

#         for i in range(num_users):
#             # First assign each client 1 shard to ensure every client has
#             # atleast one shard of data
#             rand_set = set(np.random.choice(idx_shard, 1, replace=False))
#             idx_shard = list(set(idx_shard) - rand_set)
#             for rand in rand_set:
#                 dict_users[i] = np.concatenate(
#                     (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
#                     axis=0)

#         random_shard_size = random_shard_size-1

#         # Next, randomly assign the remaining shards
#         for i in range(num_users):
#             if len(idx_shard) == 0:
#                 continue
#             shard_size = random_shard_size[i]
#             if shard_size > len(idx_shard):
#                 shard_size = len(idx_shard)
#             rand_set = set(np.random.choice(idx_shard, shard_size,
#                                             replace=False))
#             idx_shard = list(set(idx_shard) - rand_set)
#             for rand in rand_set:
#                 dict_users[i] = np.concatenate(
#                     (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
#                     axis=0)
#     else:

#         for i in range(num_users):
#             shard_size = random_shard_size[i]
#             rand_set = set(np.random.choice(idx_shard, shard_size,
#                                             replace=False))
#             idx_shard = list(set(idx_shard) - rand_set)
#             for rand in rand_set:
#                 dict_users[i] = np.concatenate(
#                     (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
#                     axis=0)

#         if len(idx_shard) > 0:
#             # Add the leftover shards to the client with minimum images:
#             shard_size = len(idx_shard)
#             # Add the remaining shard to the client with lowest data
#             k = min(dict_users, key=lambda x: len(dict_users.get(x)))
#             rand_set = set(np.random.choice(idx_shard, shard_size,
#                                             replace=False))
#             idx_shard = list(set(idx_shard) - rand_set)
#             for rand in rand_set:
#                 dict_users[k] = np.concatenate(
#                     (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
#                     axis=0)

#     return dict_users  ctrl+/多行缩进
#ctrl+/多行缩进

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.train_labels)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_dirichlet(dataset, beta, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset using Dirichlet distribution
    :param dataset: 数据集
    :param beta: Dirichlet分布参数 (较小的值会产生更不均衡的分布)
    :param num_users: 客户端数量
    :return: dict of image index
    """
    num_classes = 10  # CIFAR10有10个类别
    
    # 为每个类别生成Dirichlet分布
    label_distributions = []
    for y in range(num_classes):
        label_distributions.append(np.random.dirichlet(np.repeat(beta, num_users)))
    
    # 获取数据集的标签
    labels = np.array(dataset.targets).astype(int)
    
    # 初始化客户端索引映射
    dict_users = {i: np.array([], dtype=np.int64) for i in range(num_users)}
    
    # 按类别分配数据
    for y in range(num_classes):
        # 获取当前类别的所有样本索引
        label_y_idx = np.where(labels == y)[0]
        label_y_size = len(label_y_idx)
        
        # 计算每个客户端应分配的样本数量
        sample_size = (label_distributions[y] * label_y_size).astype(int)
        # 处理舍入误差
        sample_size[num_users-1] += label_y_size - np.sum(sample_size)
        
        # 随机打乱当前类别的样本索引
        np.random.shuffle(label_y_idx)
        
        # 分配样本给客户端
        sample_interval = np.cumsum(sample_size)
        for i in range(num_users):
            start_idx = sample_interval[i-1] if i > 0 else 0
            end_idx = sample_interval[i]
            dict_users[i] = np.concatenate(
                (dict_users[i], label_y_idx[start_idx:end_idx]), axis=0)
    
    # 随机打乱每个客户端的数据
    for i in range(num_users):
        np.random.shuffle(dict_users[i])
    
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
    dict_users = cifar_dirichlet(dataset_train, beta=0.1, num_users=10)
