import pandas as pd
import matplotlib.pyplot as plt

# 定义日志文件路径（相对于项目根目录）
log_file = './log_Lr_0.0001_batch_128_gamma_0.95_memory_20000.txt'

# 使用pandas读取CSV格式的日志文件
try:
    df = pd.read_csv(log_file)

    # 提取episode和reward数据
    # 日志中的episode不是连续的，我们按行号作为x轴
    episodes = range(1, len(df) + 1)
    rewards = df['reward']

    # 创建图表
    plt.figure(figsize=(12, 7))
    plt.plot(episodes, rewards, marker='o', linestyle='-', markersize=4, label='Reward per Episode')

    # 添加标题和标签
    plt.title('Reward vs. Episode', fontsize=16)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 保存图表
    output_path = './plot/reward_vs_episode.png'
    plt.savefig(output_path)

    print(f"图表已保存至: {output_path}")

except FileNotFoundError:
    print(f"错误: 日志文件未找到于 '{log_file}'")
except Exception as e:
    print(f"绘制图表时发生错误: {e}") 