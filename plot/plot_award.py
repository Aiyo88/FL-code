import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端避免GUI冲突
import matplotlib.pyplot as plt
import numpy as np

# 固定的路径配置
LOG_DIR = "./logs"
PLOT_DIR = "./plot"

def log_episode_result(log_filename, episode_num, reward, loss, cost):
    """
    以标准CSV格式记录单个episode的结果到指定的日志文件。
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, log_filename)

    # 如果文件不存在或为空，写入CSV表头
    if not os.path.exists(log_file) or os.path.getsize(log_file) == 0:
        with open(log_file, 'w') as f:
            f.write("episode,reward,loss,cost\n")

    # 写入数据行
    log_line = f"{episode_num},{reward},{loss},{cost}\n"
    
    with open(log_file, 'a') as f:
        f.write(log_line)


def log_training_result(param_name, param_value, round_num, reward, loss=None, cost=None):
    """
    (此函数功能已被 log_episode_result 替代，保留以兼容旧代码)
    以标准CSV格式记录训练结果到日志文件。
    文件名用于区分不同的实验配置。
    CSV格式: round,reward,loss,cost
    """
    pass # 功能已被替代，不做任何事

def _plot_metric(data, metric_name, log_name, save_dir, x_axis_label='训练轮次 (Round)', window_size=10):
    """
    (内部函数) 绘制单个指标的图表 (奖励, 损失, 或成本)。
    """
    metric_lower = metric_name.lower()
    if metric_lower not in data.columns or data[metric_lower].isnull().all():
        # 如果指标列不存在或全为NaN，则不绘图
        return

    plt.figure(figsize=(10, 6))
    
    # 提取有效数据
    metric_data = data.dropna(subset=[metric_lower])
    rounds = metric_data['round']
    values = metric_data[metric_lower]

    if len(values) < 2:
        return # 数据太少无法绘图

    # 绘制原始数据和加窗平滑后的数据
    plt.plot(rounds, values, label=f"原始{metric_name}", alpha=0.3)
    if len(values) >= window_size:
        smoothed_values = values.rolling(window=window_size, min_periods=1, center=True).mean()
        plt.plot(rounds, smoothed_values, label=f"平滑{metric_name} (窗口={window_size})")

    plt.title(f'{metric_name} vs. {x_axis_label.split(" ")[0]} ({log_name})')
    plt.xlabel(x_axis_label)
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.legend()
    
    # 保存图像
    save_path = os.path.join(save_dir, f"{metric_lower}_{log_name}.png")
    plt.savefig(save_path)
    plt.close()

def plot_training_results(log_name, save_dir="./plot", show=False):
    """
    从日志文件中读取数据并为奖励、损失和成本分别绘制图表。
    模仿 runliang.py 的行为，为每个指标生成独立的图。
    """
    log_file_path = os.path.join(LOG_DIR, f"{log_name}.txt")
    if not os.path.exists(log_file_path):
        print(f"警告: 找不到日志文件 {log_file_path}")
        return

    try:
        # 使用pandas读取CSV
        data = pd.read_csv(log_file_path)
        if data.empty:
            print(f"日志文件 {log_file_path} 为空。")
            return
    except (pd.errors.EmptyDataError, ValueError) as e:
        print(f"错误: 无法解析日志文件 {log_file_path}。请确保它是正确的CSV格式。错误: {e}")
        return
        
    # 根据日志名称确定X轴标签
    if log_name.startswith("EPISODE_REWARD"):
        x_label = '训练Episode'
    else:
        x_label = '训练轮次 (Round)'

    # 为每个指标（reward, loss, cost）生成一个独立的图表
    _plot_metric(data, 'Reward', log_name, save_dir, x_axis_label=x_label)
    _plot_metric(data, 'Loss', log_name, save_dir, x_axis_label=x_label)
    _plot_metric(data, 'Cost', log_name, save_dir, x_axis_label=x_label)

def plot_all_logs(save_dir="./plot", show=False):
    """
    遍历./logs目录下的所有.txt日志文件，并为每个文件生成独立的性能图表。
    """
    if not os.path.exists(LOG_DIR):
        print(f"日志目录 {LOG_DIR} 不存在。")
        return

    log_files = [f for f in os.listdir(LOG_DIR) if f.endswith('.txt')]

    if not log_files:
        print(f"在 {LOG_DIR} 目录中没有找到日志文件。")
        return

    print(f"找到了 {len(log_files)} 个日志文件，开始为每个文件生成独立的奖励、损失和成本图表...")
    for log_file in log_files:
        # 从文件名中提取log_name (去掉.txt后缀)
        log_name = os.path.splitext(log_file)[0]
        print(f"  - 正在处理: {log_name}")
        plot_training_results(log_name, save_dir=save_dir, show=show)
    
    print(f"所有图表已生成并保存到 {save_dir} 目录。")

def plot_main():
    """用于测试绘图功能的独立脚本入口"""
    print("开始生成所有日志的性能图表...")
    plot_all_logs(save_dir=PLOT_DIR, show=False)

if __name__ == '__main__':
    plot_main()
