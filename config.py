"""
Global configuration file for the federated learning system
集中管理所有配置参数，避免在多个文件中重复定义
"""
import os
import torch

# ===== 系统配置 =====
# 基础配置
SEED = 42                      # 随机种子，确保结果可复现
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 计算设备
DISABLE_PROGRESS_BAR = False   # 是否禁用进度条显示

# 路径配置
DATA_PATH = './data/'          # 数据存储路径
MODEL_SAVE_PATH = './saved_models/'  # 模型保存路径
LOG_SAVE_PATH = './logs/'      # 日志保存路径
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(LOG_SAVE_PATH, exist_ok=True)

# ==============================================================================
# 1. 联邦学习超参数
# ==============================================================================
NUM_CLIENTS = 5               # 客户端数量
NUM_EDGES = 2                 # 边缘节点数量
IID = True                    # 数据是否独立同分布
NON_IID_LEVEL = 1              # 非IID程度 (1/2/3)，与main.py保持一致

# 训练配置
NUM_ROUNDS = 100               # 联邦学习轮数
LOCAL_EPOCHS = 5               # 本地训练轮数
BATCH_SIZE = 64                # 批次大小（与main.py保持一致）
LEARNING_RATE = 0.01           # 学习率
MOMENTUM = 0.9                 # 动量
WEIGHT_DECAY = 1e-5            # 权重衰减
CLIENT_FRACTION = 0.2          # 每轮参与训练的客户端比例

# ATAFL算法相关
ATAFL_ETA = 0.5                  # ATAFL算法中的η超参数

# ===== DRL配置 =====
# DRL基础配置
USE_DRL = True                 # 是否使用DRL决策
DRL_TRAIN = True               # 是否训练DRL模型
STATE_DIM = 10 * NUM_CLIENTS + 5 * NUM_EDGES  # 状态空间维度
DISCRETE_ACTION_DIM = NUM_CLIENTS + 1  # 离散动作空间维度（客户端选择+聚合决策）
CONTINUOUS_ACTION_DIM = NUM_EDGES + NUM_CLIENTS  # 连续动作空间维度（资源分配）

# DRL训练参数
GAMMA = 0.99                   # 折扣因子
TAU = 0.005                    # 目标网络软更新参数
ACTOR_LR = 3e-4                # Actor学习率
CRITIC_LR = 3e-4               # Critic学习率
UPDATE_GLOBAL_ITER = 5         # 更新全局网络的频率
ENTROPY_BETA = 0.01            # 熵正则化系数

# ===== 资源管理配置 =====
# 能量管理
MAX_ENERGY = 2500              # 最大能量（与Env.py的energy_max保持一致）
MIN_ENERGY = 100               # 最小能量
ENERGY_DECAY = 0.1             # 能量衰减率
ENERGY_REWARD_WEIGHT = 0.3     # 能量奖励权重

# 计算资源
MAX_COMPUTE_CAPABILITY = 5.0   # 最大计算能力
MIN_COMPUTE_CAPABILITY = 0.5   # 最小计算能力
MAX_COMM_RATE = 5.0            # 最大通信速率
MIN_COMM_RATE = 0.5            # 最小通信速率

# ===== 环境配置 =====
# 环境参数
C = 10                         # 环境参数C（与main.py和Env.py保持一致）
GAMA = 0.01                    # 环境参数GAMA（与main.py和Env.py保持一致）
DELTA = 1.0                    # 环境参数DELTA（与main.py和Env.py保持一致）

# ===== 模型配置 =====
# 默认模型配置
DEFAULT_MODEL = 'mnist'        # 默认模型类型（与main.py保持一致）
DEFAULT_DATASET = 'mnist'      # 默认数据集

# ===== 日志配置 =====
LOG_INTERVAL = 10              # 日志记录间隔
SAVE_INTERVAL = 50             # 模型保存间隔
EVAL_INTERVAL = 5              # 评估间隔

# ===== 客户端选择配置 =====
MIN_ACTIVE_CLIENTS = 3         # 最小活跃客户端数
MAX_ACTIVE_CLIENTS = 7         # 最大活跃客户端数
SELECTION_INTERVAL = 5         # 选择间隔

# 根据环境变量覆盖配置（便于命令行参数传递）
def update_config_from_env():
    """从环境变量更新配置"""
    import os
    
    # 示例：从环境变量更新NUM_CLIENTS
    if 'NUM_CLIENTS' in os.environ:
        global NUM_CLIENTS
        NUM_CLIENTS = int(os.environ['NUM_CLIENTS'])
    
    # 可以添加更多环境变量处理

# 调用函数更新配置
update_config_from_env()