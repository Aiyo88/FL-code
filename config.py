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
CHECKPOINT_PATH = './checkpoints/'  # 检查点保存路径

# 确保路径存在
for path in [DATA_PATH, MODEL_SAVE_PATH, LOG_SAVE_PATH, CHECKPOINT_PATH]:
    os.makedirs(path, exist_ok=True)

# ==============================================================================
# 1. 联邦学习超参数
# ==============================================================================
NUM_CLIENTS = 5               # 客户端数量
NUM_EDGES = 2                 # 边缘节点数量
IID = True                    # 数据是否独立同分布
NON_IID_LEVEL = 1             # 非IID程度 (1/2/3)

# 训练配置
NUM_ROUNDS = 100               # 联邦学习轮数
NUM_EPISODES = 200             # DRL训练的Episode总数上限
FL_ROUNDS_PER_EPISODE = 100    # 每个DRL Episode包含的FL轮次数量
LOCAL_EPOCHS = 1               # 本地训练轮数 (原为5)
BATCH_SIZE = 512               # 批次大小 (原为2048)
LEARNING_RATE = 0.001          # 联邦学习学习率
MOMENTUM = 0.9                 # 动量
WEIGHT_DECAY = 1e-5            # 权重衰减
CLIENT_FRACTION = 0.2          # 每轮参与训练的客户端比例

# 数据集配置
DEFAULT_DATASET = 'mnist'      # 默认数据集
DEFAULT_MODEL = 'mnist'        # 默认模型类型
SUPPORTED_DATASETS = ['mnist', 'cifar10']

# ATAFL算法相关
ATAFL_ETA = 0.5                # ATAFL算法中的η超参数

# ==============================================================================
# 2. 服务器配置
# ==============================================================================
# 收敛参数
CONVERGENCE_EPSILON = 1e-4     # 模型收敛阈值
CONVERGENCE_PATIENCE = 3       # 连续多少轮无明显变化则认为收敛

# 评估和保存频率
VAL_FREQ = 1                   # 模型验证频率
SAVE_FREQ = 5                  # 模型保存频率

# ==============================================================================
# 3. 环境配置 (Env.py)
# ==============================================================================
# 系统架构参数
DEFAULT_NUM_DEVICES = 5        # 默认终端设备数量
DEFAULT_NUM_EDGES = 2          # 默认边缘服务器数量
NUM_CLOUD_SERVERS = 1          # 云服务器数量

# 李雅普诺夫队列参数
ENERGY_THRESHOLD = 500         # 李雅普诺夫队列能量阈值
ENERGY_MAX = 500              # 最大能量
ENERGY_REPLENISH_RATE = 5.0   # 设备每轮补充的能量 (J) (原为20.0)
CONVERGENCE_EPSILON_ENV = 1e-3 # 环境收敛阈值

# 数据和模型大小
DEFAULT_DATA_SIZE = 10 * 1024 * 1024   # 10MB 默认数据大小
DEFAULT_MODEL_SIZE = 2 * 1024 * 1024   # 2MB 默认模型大小

# 时间模型
TIME_SLOT = 100.0                # 时隙长度 (原为0.1)

# 通信模型参数
BANDWIDTH = 6                  # 信道带宽 (MHz)
NOISE_POWER = 10e-13          # 噪声功率
PT_UP = 24                    # 上行传输功率 (dBm)
PT_DOWN = 30                  # 下行传输功率 (dBm)
PT_DEVICE_RECEIVE = 20        # 终端设备接收功率 (dBm)
PT_EDGE_TRANSMIT = 24         # 边缘节点传输功率 (dBm)
PT_CLOUD_DOWN = 30            # 云到边缘下行传输功率(dBm)
RATE_CU = 120                 # 边缘到云上行速率 (Mbps)
RATE_CD = 150                 # 边缘到云下行速率 (Mbps)

# 计算模型参数
# 终端设备计算资源
F_L_MIN = 0.4e9               # 最小CPU频率 (0.4GHz)
F_L_MAX = 2.0e9               # 最大CPU频率 (2.0GHz) (原为2.9e9)
# 边缘节点计算资源
F_E_MIN = 2.9e9               # 最小CPU频率 (2.9GHz)
F_E_MAX = 5.0e9               # 最大CPU频率 (5.0GHz) (原为4.3e9)
# 计算复杂度范围
COMPUTE_COMPLEXITY_MIN = 300   # cycles/bit
COMPUTE_COMPLEXITY_MAX = 500   # cycles/bit
COMPUTATION_ENERGY_SCALE = 0.01 # 计算能耗缩放因子，用于平衡物理模型

# 李雅普诺夫权重参数
ALPHA = 0.5                   # 延迟权重
BETA = 0.5                    # 能耗权重

# 奖励归一化参数
MAX_COST_PER_ROUND = 200.0    # 预估的单轮最大成本
MAX_Q_ENERGY_PER_ROUND = 1000.0  # 预估的单轮最大队列能量项

# 奖励权重 - 重新平衡
W_COST = 0.6                  # 降低成本权重
W_Q = 0.3                     # 降低队列权重
W_LOSS = 0.1                  # 提高损失（性能）权重

# ==================== 李雅普诺夫优化参数 ====================
LYAPUNOV_V = 1.0  # 李雅普诺夫漂移+惩罚项的V值，用于权衡成本与队列稳定性 (原为0.1)

# ATAFL算法相关
ATAFL_ETA = 0.5

# ==============================================================================
# 4. 客户端配置 (clients.py)
# ==============================================================================
# 客户端能力参数
MAX_COMPUTE_CAPABILITY = 5.0  # 最大计算能力
MIN_COMPUTE_CAPABILITY = 0.5  # 最小计算能力
MAX_ENERGY = 2500             # 最大能量
MIN_ENERGY = 100              # 最小能量
MAX_COMM_RATE = 5.0           # 最大通信速率
MIN_COMM_RATE = 0.5           # 最小通信速率

# 边缘节点参数
EDGE_COVERAGE_RADIUS = 500.0  # 边缘节点覆盖半径

# 数据分割参数
TRAIN_RATIO = 0.8             # 训练集比例
VAL_RATIO = 0.1               # 验证集比例
TEST_RATIO = 0.1              # 测试集比例

# ==============================================================================
# 5. DRL配置
# ==============================================================================
# DRL基础配置
USE_DRL = True                # 是否使用DRL决策
DRL_TRAIN = True              # 是否训练DRL模型
DRL_ALGO = 'pdqn'             # 默认DRL算法

# DRL超参数
DRL_LR = 0.0003               # DRL学习率 (原为0.0001, 调高以增强学习信号)
DRL_BATCH_SIZE = 256          # DRL批次大小 (原为512)
DRL_GAMMA = 0.99              # DRL折扣因子
DRL_MEMORY_SIZE = 100000      # DRL回放缓存区大小 (原为20000)
EPSILON_INITIAL = 0.6         # 初始探索率 (原为0.9)
EPSILON_FINAL = 0.05          # 最终探索率
EPSILON_DECAY_STEPS = 100     # ε衰减的Episode步数 (原为150)

# DRL网络参数
RESNET_HIDDEN_SIZE = 256       # ResNet隐藏层大小
RESNET_NUM_BLOCKS = 2          # ResNet残差块数量
TAU = 0.001                   # 目标网络软更新参数 (降低以增加稳定性)
ACTOR_LR = 3e-4               # Actor学习率
CRITIC_LR = 3e-4              # Critic学习率
UPDATE_GLOBAL_ITER = 5        # 更新全局网络的频率
ENTROPY_BETA = 0.01           # 熵正则化系数

# ==============================================================================
# 6. 状态和动作空间配置
# ==============================================================================
# 状态空间维度计算
def get_state_dim(num_devices=DEFAULT_NUM_DEVICES, num_edges=DEFAULT_NUM_EDGES):
    """计算状态空间维度"""
    return 5 * num_devices + 2 * num_devices * num_edges + 3 * num_edges + 3

# 动作空间维度
def get_action_dims(num_devices=DEFAULT_NUM_DEVICES, num_edges=DEFAULT_NUM_EDGES):
    """获取动作空间维度"""
    train_dim = num_devices
    edge_train_dim = num_devices  
    edge_agg_dim = 1
    cloud_agg_dim = 0
    continuous_dim = num_devices * num_edges
    return train_dim, edge_train_dim, edge_agg_dim, cloud_agg_dim, continuous_dim

# ==============================================================================
# 7. 日志和监控配置
# ==============================================================================
LOG_INTERVAL = 10             # 日志记录间隔
SAVE_INTERVAL = 50            # 模型保存间隔
EVAL_INTERVAL = 5             # 评估间隔

# 绘图配置
PLOT_RESULTS = True           # 是否生成结果图表
PLOT_DIR = './plot/'          # 绘图保存目录
os.makedirs(PLOT_DIR, exist_ok=True)

# ==============================================================================
# 8. GPU优化配置
# ==============================================================================
NUM_WORKERS = 4               # 数据加载器工作进程数
PIN_MEMORY = True             # 是否将数据固定在内存中
NON_BLOCKING = True           # 异步GPU数据传输

# ==============================================================================
# 9. 网络和无线通信配置
# ==============================================================================
# 信道模型参数
RAYLEIGH_SCALE = 1.0          # 瑞利分布参数
CORRELATION_FACTOR = 0.7      # 时间相关性因子
MIN_CHANNEL_GAIN = 0.1        # 最小信道增益

# 边缘到边缘传输速率
EDGE_TO_EDGE_RATE = 1000e6    # 1Gbps

# ==============================================================================
# 10. 无效动作处理配置
# ==============================================================================
INVALID_ACTION_LOG = "logs/invalid_actions.log"  # 无效动作日志文件
CONSTRAINT_TOLERANCE = 1e-6   # 约束容差

# ==============================================================================
# 11. 主程序默认参数
# ==============================================================================
# 主程序参数默认值
DEFAULT_ARGS = {
    'seed': SEED,
    'device': 'cuda',
    'save_path': CHECKPOINT_PATH,
    'data_path': DATA_PATH,
    'disable_progress_bar': 0,
    'dataset': DEFAULT_DATASET,
    'model': DEFAULT_MODEL,
    'num_clients': NUM_CLIENTS,
    'num_edges': NUM_EDGES,
    'iid': 1,
    'non_iid_level': NON_IID_LEVEL,
    'epochs': LOCAL_EPOCHS,
    'batch_size': BATCH_SIZE,
    'lr': LEARNING_RATE,
    'num_rounds': NUM_ROUNDS,
    'num_episodes': NUM_EPISODES,
    'drl_train': 1,
    'drl_algo': DRL_ALGO,
    'drl_load': None,
    'drl_lr': DRL_LR,
    'drl_batch_size': DRL_BATCH_SIZE,
    'drl_gamma': DRL_GAMMA,
    'drl_memory_size': DRL_MEMORY_SIZE,
    'energy_threshold': ENERGY_THRESHOLD,
    'num_workers': NUM_WORKERS,
    'pin_memory': PIN_MEMORY,
    'non_blocking': NON_BLOCKING,
    'plot': 1
}

# 根据环境变量覆盖配置（便于命令行参数传递）
def update_config_from_env():
    """从环境变量更新配置"""
    import os
    
    # 从环境变量更新关键参数
    global NUM_CLIENTS, NUM_EDGES, BATCH_SIZE, LEARNING_RATE
    
    if 'NUM_CLIENTS' in os.environ:
        NUM_CLIENTS = int(os.environ['NUM_CLIENTS'])
    if 'NUM_EDGES' in os.environ:
        NUM_EDGES = int(os.environ['NUM_EDGES'])
    if 'BATCH_SIZE' in os.environ:
        BATCH_SIZE = int(os.environ['BATCH_SIZE'])
    if 'LEARNING_RATE' in os.environ:
        LEARNING_RATE = float(os.environ['LEARNING_RATE'])

# 调用函数更新配置
update_config_from_env()

# ==============================================================================
# 12. 配置验证和辅助函数
# ==============================================================================
def validate_config():
    """验证配置参数的合理性"""
    assert NUM_CLIENTS > 0, "NUM_CLIENTS must be positive"
    assert NUM_EDGES > 0, "NUM_EDGES must be positive"
    assert 0 < CLIENT_FRACTION <= 1, "CLIENT_FRACTION must be in (0, 1]"
    assert LOCAL_EPOCHS > 0, "LOCAL_EPOCHS must be positive"
    assert BATCH_SIZE > 0, "BATCH_SIZE must be positive"
    assert LEARNING_RATE > 0, "LEARNING_RATE must be positive"
    
def print_config():
    """打印当前配置"""
    print("=== 当前配置 ===")
    print(f"客户端数量: {NUM_CLIENTS}")
    print(f"边缘节点数量: {NUM_EDGES}")
    print(f"本地训练轮数: {LOCAL_EPOCHS}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"学习率: {LEARNING_RATE}")
    print(f"数据集: {DEFAULT_DATASET}")
    print(f"设备: {DEVICE}")
    print("===============")

# 初始化时验证配置
validate_config()