import json
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def smooth_curve(data, window_size=10):
    """使用滑动窗口平滑曲线"""
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window_size//2)
        end = min(len(data), i + window_size//2)
        smoothed.append(np.mean(data[start:end]))
    return smoothed

def plot_comparison(results_dir):
    plt.figure(figsize=(16, 12))
    param_types = ['learning_rate', 'batch_size', 'gamma', 'memory_size']
    
    for idx, param in enumerate(param_types, 1):
        with open(Path(results_dir)/f"{param}_results.json") as f:
            data = json.load(f)
        
        plt.subplot(2, 2, idx)
        
        for value in data.keys():
            rewards = data[value]['rewards']
            episodes = list(range(len(rewards)))
            
            # 平滑处理
            smoothed = smooth_curve(rewards, window_size=15)
            
            # 绘制曲线
            plt.plot(episodes, smoothed,
                    linewidth=2,
                    label=f"{value} ({param})")
        
        plt.title(f"{param.replace('_', ' ').title()} Comparison")
        plt.xlabel('Training Episodes')
        plt.ylabel('Average Reward (Smoothed)')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(Path(results_dir)/'parameter_comparison.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    results_dir = "results/20230801-153000"  # 替换为实际路径
    plot_comparison(results_dir)
