import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re

def plot_episode_stats(log_file_path, save_dir):
    """
    从日志文件中读取每个Episode的统计数据并绘制曲线图。

    Args:
        log_file_path (str): 日志文件的路径。
        save_dir (str): 图表保存的目录。
    """
    if not os.path.exists(log_file_path):
        print(f"错误: 日志文件不存在 {log_file_path}")
        return

    # 使用pandas读取数据，更健壮
    try:
        data = pd.read_csv(log_file_path)
    except Exception as e:
        print(f"读取日志文件时出错: {e}")
        return

    if data.empty:
        print("日志文件为空，无法绘图。")
        return

    episodes = data['episode'].values
    avg_rewards = data['reward'].values  # 修正：使用正确的列名'reward'
    avg_losses = data['loss'].values     # 修正：使用正确的列名'loss'
    avg_costs = data['cost'].values      # 修正：使用正确的列名'cost'

    # 将负值奖励转换为正值显示（翻转符号，使向上表示更好）
    rewards_display = -avg_rewards  # 翻转符号，原来-4变成+4

    # 创建一个包含三个子图的图表
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('DRL Agent Learning Convergence Analysis', fontsize=16, fontweight='bold')

    # 1. 奖励图 - 现在向上表示更好
    ax1.plot(episodes, rewards_display, 'b-o', linewidth=2, markersize=4, alpha=0.8, label='Average Reward per Episode')
    ax1.set_xlabel('DRL Episode')
    ax1.set_ylabel('Average Reward (Higher is Better)')
    ax1.set_title('Episode vs. Average Reward')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. 损失图
    ax2.plot(episodes, avg_losses, marker='x', linestyle='--', color='r', label='Average Loss per Episode')
    ax2.set_ylabel('Average FL Model Loss')
    ax2.set_title('Episode vs. Average Loss')
    ax2.grid(True)
    ax2.legend()

    # 3. 绘制 Episode vs. 平均成本
    ax3.plot(episodes, avg_costs, marker='s', linestyle='-.', color='g', label='Average Cost per Episode')
    ax3.set_ylabel('Average Cost (Delay + Energy)')
    ax3.set_title('Episode vs. Average Cost')
    ax3.set_xlabel('DRL Episode')
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # 保存图表
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 从日志文件名中提取基础名称
    base_name = os.path.splitext(os.path.basename(log_file_path))[0]
    save_path = os.path.join(save_dir, f'episode_convergence_{base_name}.png')

    plt.savefig(save_path)
    print(f"收敛分析图已保存到: {save_path}")
    plt.close(fig)

if __name__ == '__main__':
    # 使用示例：
    # 假设您的日志文件保存在 'logs/log_Lr_0.0001_batch_128_gamma_0.95_memory_20000.txt'
    # 您可以在这里指定要分析的日志文件

    # 自动查找最新的日志文件进行分析
    log_dir = './logs'
    log_files = [f for f in os.listdir(log_dir) if f.startswith('log_Lr_') and f.endswith('.txt')]

    if log_files:
        latest_log_file = max(log_files, key=lambda f: os.path.getmtime(os.path.join(log_dir, f)))
        print(f"找到最新的日志文件: {latest_log_file}")
        log_file_to_plot = os.path.join(log_dir, latest_log_file)
        plot_episode_stats(log_file_to_plot, save_dir='./plot')
    else:
        print("在 'logs/' 目录下未找到符合条件的日志文件 (例如 'log_Lr_...txt')。")
        print("请先运行 main.py 生成日志文件。") 