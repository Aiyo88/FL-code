import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import os
import math
import time
import threading

# 设置OpenMP线程数，避免过多线程竞争
os.environ["OMP_NUM_THREADS"] = "1"

# 训练参数
UPDATE_GLOBAL_ITER = 5  # 更新全局网络的频率
GAMMA = 0.99           # 折扣因子
ENTROPY_BETA = 0.01    # 熵正则化系数

class SharedAdam(torch.optim.Adam):
    """共享参数的Adam优化器"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # 初始化优化器状态
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # 共享内存
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
                state['step'].share_memory_()

def v_wrap(np_array, dtype=np.float32):
    """包装numpy数组为tensor"""
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)

class Net(nn.Module):
    """A3C网络 - 处理混合动作空间（离散+连续），使用标准网络结构"""
    def __init__(self, s_dim, discrete_dim, continuous_dim, hidden1_size=400, hidden2_size=300):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.discrete_dim = discrete_dim
        self.continuous_dim = continuous_dim
        
        # 计算输入特征的尺寸
        self.input_size = int(np.ceil(np.sqrt(s_dim)))
        self.padding_size = self.input_size * self.input_size - s_dim
        
        # 使用标准网络架构(400-300)的共享层
        self.shared_fc = nn.Sequential(
            nn.Linear(s_dim, hidden1_size),
            nn.ReLU(),
            nn.Linear(hidden1_size, hidden2_size),
            nn.ReLU()
        )
        
        # 离散动作头
        self.pi_disc = nn.Linear(hidden2_size, discrete_dim)
        
        # 连续动作头（均值）
        self.mu = nn.Sequential(
            nn.Linear(hidden2_size, continuous_dim),
            nn.Tanh()  # 输出范围[-1, 1]
        )
        
        # 连续动作头（标准差）
        self.sigma = nn.Sequential(
            nn.Linear(hidden2_size, continuous_dim),
            nn.Softplus()  # 确保标准差为正
        )
        
        # 价值函数
        self.value = nn.Linear(hidden2_size, 1)
        
        # 连续动作分布
        self.distribution = torch.distributions.Normal
    
    def forward(self, x):
        """前向传播"""
        # 确保输入数据有效
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
        # 共享特征层
        x = self.shared_fc(x)
        
        # 离散动作概率
        disc_logits = self.pi_disc(x)
        disc_probs = torch.sigmoid(disc_logits)
        disc_probs = torch.clamp(disc_probs, 0.01, 0.99)  # 避免0和1
        
        # 连续动作参数
        mu = self.mu(x)
        
        sigma = self.sigma(x) + 0.01  # 避免标准差太小
        sigma = torch.clamp(sigma, min=0.1, max=1.0)
        
        # 状态价值
        values = self.value(x)
        
        return disc_probs, mu, sigma, values

    def choose_action(self, s):
        """基于当前状态选择离散和连续的动作
        
        Args:
            s: 状态输入
        
        Returns:
            action: 完整的动作数组，包括所有离散动作和连续动作
        """
        self.eval()
        with torch.no_grad():
            # 确保状态张量是二维的 [batch_size, state_dim]
            if len(s.shape) == 1:
                s = s.unsqueeze(0)
            
            # 获取策略输出
            disc_probs, mu, sigma, _ = self.forward(s)
            
            # 确保获得正确形状的输出
            disc_dim = self.discrete_dim if hasattr(self, 'discrete_dim') else disc_probs.size(1)
            cont_dim = self.continuous_dim if hasattr(self, 'continuous_dim') else mu.size(1)
            
            # 检查输出维度
            total_actions = disc_dim + cont_dim
            action = np.zeros(total_actions, dtype=np.float32)
            
            # 始终使用第一个样本的输出（如果有多个样本）
            # 离散动作采样
            for i in range(disc_dim):
                # 获取当前维度的动作概率
                prob = disc_probs[0, i] if disc_probs.dim() > 1 else disc_probs[i]
                
                # 根据概率二值化决策
                action[i] = 1 if np.random.uniform() < prob.item() else 0
            
            # 连续动作采样
            m = self.distribution(mu, sigma)
            
            try:
                # 采样连续动作
                cont_a = m.sample().detach().cpu().numpy()
                
                # 确保cont_a是一维的
                if len(cont_a.shape) > 1:
                    cont_a = cont_a[0]  # 使用第一个样本
                
                # 填充连续动作，确保维度正确
                for i in range(min(len(cont_a), cont_dim)):
                    action[disc_dim + i] = (cont_a[i] + 1) / 2  # 将[-1,1]映射到[0,1]
                
                # 如果维度不匹配，则填充剩余的位置
                if len(cont_a) < cont_dim:
                    for i in range(len(cont_a), cont_dim):
                        action[disc_dim + i] = 0.5  # 默认为中间值
            except Exception as e:
                print(f"连续动作采样出错: {e}")
                # 如果采样失败，使用mu作为默认值
                mu_np = mu.detach().cpu().numpy()
                
                # 确保mu_np是一维的
                if len(mu_np.shape) > 1:
                    mu_np = mu_np[0]  # 使用第一个样本
                
                # 填充连续动作
                for i in range(min(len(mu_np), cont_dim)):
                    action[disc_dim + i] = (mu_np[i] + 1) / 2  # 将[-1,1]映射到[0,1]
                
                # 如果维度不匹配，则填充剩余的位置
                if len(mu_np) < cont_dim:
                    for i in range(len(mu_np), cont_dim):
                        action[disc_dim + i] = 0.5  # 默认为中间值
            
            # 检查最终的动作数组是否完整有效
            if np.isnan(action).any() or np.isinf(action).any():
                print("警告: 动作数组包含NaN或Inf值，使用默认值替换")
                action = np.nan_to_num(action, nan=0.5, posinf=1.0, neginf=0.0)
            
            return action

    def loss_func(self, s, disc_a, cont_a, v_t):
        """计算A3C的损失函数

        Args:
            s: 状态
            disc_a: 离散动作
            cont_a: 连续动作
            v_t: 目标价值
        
        Returns:
            total_loss: 总损失
            a_loss: 动作损失
            c_loss: 价值损失
        """
        self.train()
        disc_log_probs = []
        
        disc_probs, mu, sigma, values = self.forward(s)
        
        # 处理输入张量的维度
        is_single_sample = (len(s.shape) == 1 or s.shape[0] == 1)
        
        # 获取价值估计
        if is_single_sample:
            value = values.squeeze()
        else:
            value = values.squeeze(-1)  # 移除可能的最后一个维度
        
        # 价值损失 - 使用MSE
        advantage = v_t - value
        c_loss = advantage.pow(2)
        
        # ===== 离散动作的损失 =====
        # 处理离散动作 - 需要检查维度
        if isinstance(disc_a, np.ndarray):
            disc_a = torch.from_numpy(disc_a).float().to(disc_probs.device)
        
        # 确保动作张量与概率张量维度匹配
        if disc_probs.dim() > 1 and disc_a.dim() == 1:
            disc_a = disc_a.unsqueeze(0)
        
        # 处理离散动作损失
        for i in range(disc_probs.size(-1)):  # 使用-1以适应不同维度
            try:
                # 获取当前位置的概率
                if disc_probs.dim() > 1:
                    prob_i = disc_probs[:, i]
                else:
                    prob_i = disc_probs[i].unsqueeze(0) if is_single_sample else disc_probs[i]
                
                # 获取当前位置的动作
                if disc_a.dim() > 1:
                    a_i = disc_a[:, i]
                else:
                    a_i = disc_a[i].unsqueeze(0) if is_single_sample else disc_a[i]
                
                # 计算log概率 - 修复部分
                if a_i.numel() == 1:  # 单元素张量
                    if a_i.item() == 1:
                        disc_log_probs.append(torch.log(prob_i + 1e-10))
                    else:
                        disc_log_probs.append(torch.log(1 - prob_i + 1e-10))
                else:
                    # 处理多元素张量 - 使用torch.where进行向量化操作
                    log_probs = torch.where(
                        a_i == 1,
                        torch.log(prob_i + 1e-10),
                        torch.log(1 - prob_i + 1e-10)
                    )
                    disc_log_probs.append(log_probs)
            except Exception as e:
                print(f"计算离散动作损失出错 (i={i}): {e}")
                # 如果出错，添加一个近似0的负log概率
                if is_single_sample:
                    disc_log_probs.append(torch.tensor([-10.0], device=disc_probs.device))
                else:
                    # 对于批处理情况创建适当维度的张量
                    batch_size = s.shape[0]
                    disc_log_probs.append(torch.full((batch_size,), -10.0, device=disc_probs.device))
                    
        # ===== 连续动作的损失 =====
        # 处理连续动作 - 需要检查维度
        if isinstance(cont_a, np.ndarray):
            cont_a = torch.from_numpy(cont_a).float().to(mu.device)
        
        # 将连续动作从[0,1]映射回[-1,1]
        cont_a = cont_a * 2 - 1
        
        # 确保动作张量与mu/sigma维度匹配
        if mu.dim() > 1 and cont_a.dim() == 1:
            cont_a = cont_a.unsqueeze(0)
        
        # 处理连续动作损失
        m = self.distribution(mu, sigma)
        try:
            # 计算连续动作的log概率
            cont_log_prob = m.log_prob(cont_a)
            
            # 适应不同维度
            if cont_log_prob.dim() == 2:
                cont_log_probs = cont_log_prob
            else:
                cont_log_probs = cont_log_prob.unsqueeze(0) if is_single_sample else cont_log_prob
        except Exception as e:
            print(f"计算连续动作损失出错: {e}")
            # 如果出错，添加一个近似0的负log概率
            shape = mu.shape if mu.dim() > 1 else (1, mu.shape[0])
            cont_log_probs = torch.full(shape, -10.0, device=mu.device)
        
        # ===== 计算总损失 =====
        try:
            # 将离散动作log概率转换为张量并组合
            if disc_log_probs:
                if is_single_sample:
                    # 单样本情况
                    disc_log_probs_tensor = torch.stack(disc_log_probs).sum().unsqueeze(0)
                else:
                    # 批量情况 - 修正维度问题
                    try:
                        disc_log_probs_tensor = torch.stack(disc_log_probs)
                        if disc_log_probs_tensor.dim() == 2:
                            # 形状为 [batch_size, num_actions] 时需要转置并求和
                            disc_log_probs_tensor = disc_log_probs_tensor.transpose(0, 1)
                            disc_log_probs_tensor = disc_log_probs_tensor.sum(dim=1)
                        else:
                            # 单样本或其他情况
                            disc_log_probs_tensor = disc_log_probs_tensor.sum(dim=0).unsqueeze(0)
                    except Exception as e:
                        print(f"组合离散动作log概率出错: {e}")
                        # 如果合并失败，创建一个零张量
                        disc_log_probs_tensor = torch.zeros_like(advantage)
            else:
                disc_log_probs_tensor = torch.zeros_like(advantage)
            
            # 组合连续动作log概率
            if isinstance(cont_log_probs, torch.Tensor) and cont_log_probs.numel() > 0:
                if cont_log_probs.dim() > 1:
                    # 形状为 [batch_size, action_dim]，按行求和
                    cont_log_probs_sum = cont_log_probs.sum(dim=1)
                else:
                    # 形状为 [action_dim]，求和后扩展维度
                    cont_log_probs_sum = cont_log_probs.sum().unsqueeze(0)
            else:
                cont_log_probs_sum = torch.zeros_like(advantage)
            
            # 更健壮的张量维度匹配
            # 确保我们了解张量的实际形状
            debug_mode = False  # 设置为True可以输出更多调试信息
            if debug_mode:
                print(f"DEBUG: disc_log_probs_tensor shape: {disc_log_probs_tensor.shape}")
                print(f"DEBUG: cont_log_probs_sum shape: {cont_log_probs_sum.shape}")
                print(f"DEBUG: advantage shape: {advantage.shape}")
            
            # 简单粗暴地确保维度匹配
            if disc_log_probs_tensor.shape != advantage.shape:
                if advantage.dim() > 1 and disc_log_probs_tensor.dim() == 1:
                    # 扩展为[batch_size, 1]然后广播
                    disc_log_probs_tensor = disc_log_probs_tensor.reshape(-1, 1).expand_as(advantage)
                else:
                    # 简单使用均值
                    disc_log_probs_tensor = disc_log_probs_tensor.mean().expand_as(advantage)

            if cont_log_probs_sum.shape != advantage.shape:
                if advantage.dim() > 1 and cont_log_probs_sum.dim() == 1:
                    # 扩展为[batch_size, 1]然后广播  
                    cont_log_probs_sum = cont_log_probs_sum.reshape(-1, 1).expand_as(advantage)
                else:
                    # 简单使用均值
                    cont_log_probs_sum = cont_log_probs_sum.mean().expand_as(advantage)
            
            # 总的动作log概率
            log_prob = disc_log_probs_tensor + cont_log_probs_sum
            
            # 计算策略损失
            a_loss = -(log_prob * advantage.detach())
            
            # 总损失
            total_loss = (a_loss + 0.5 * c_loss).mean()
            
            # 检查NaN或Inf
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print("警告: 损失函数计算出现NaN或Inf，使用默认值替代")
                return torch.tensor(0.1, device=s.device, requires_grad=True), torch.tensor(0.05, device=s.device), torch.tensor(0.05, device=s.device)
            
            return total_loss, a_loss.mean(), c_loss.mean()
            
        except Exception as e:
            import traceback
            print(f"计算总损失出错: {e}")
            traceback.print_exc()
            # 返回默认损失值
            return torch.tensor(0.1, device=s.device, requires_grad=True), torch.tensor(0.05, device=s.device), torch.tensor(0.05, device=s.device)

class Worker(mp.Process):
    """A3C工作进程"""
    def __init__(self, global_net, optimizer, global_ep, res_queue, name, s_dim, disc_dim, cont_dim, max_episodes=100, env_params=None):
        super(Worker, self).__init__()
        self.name = f'w{name:02d}'
        self.g_ep = global_ep
        self.res_queue = res_queue
        self.global_net = global_net
        self.optimizer = optimizer
        self.local_net = Net(s_dim, disc_dim, cont_dim)
        self.s_dim = s_dim
        self.max_episodes = max_episodes
        self.env_params = env_params
        
        # 在初始化时不创建环境，稍后在run方法中创建
        self.env = None
        
    def run(self):
        """运行工作进程"""
        total_step = 1
        print(f"Worker {self.name} 开始运行")
        
        # 在run方法中创建环境
        try:
            # 简单创建环境
            from Env import Env
            self.env = Env(**{k:v for k,v in self.env_params.items() if k in ['C', 'gama', 'delta']})
            self.env.N = self.env_params.get('N', 10)
            self.env.M = self.env_params.get('M', 3)
            
            # 确保环境有state_dim属性
            self.env.state_dim = self.s_dim
            
            # 初始化本地策略网络相关的动作维度 - 确保与DRL类使用相同的维度
            self.local_net.discrete_dim = self.env.N + 1  # 只保留训练决策和聚合决策
            self.local_net.continuous_dim = self.env.M + self.env.N  # 资源分配维度
            
            # 验证环境接口正确性
            test_state = self.env.reset()
            if len(test_state) != self.s_dim:
                print(f"警告: 环境返回的状态维度 {len(test_state)} 与指定的状态维度 {self.s_dim} 不匹配")
                # 调整网络输入维度以匹配环境状态
                self.local_net = Net(len(test_state), self.local_net.discrete_dim, self.local_net.continuous_dim)
                
        except Exception as e:
            print(f"Worker {self.name} 创建环境时出错: {e}")
            import traceback
            traceback.print_exc()
            self.res_queue.put(None)  # 通知结束
            return
        
        try:
            # 从全局网络同步参数
            self.local_net.load_state_dict(self.global_net.state_dict())
            
            # 设置训练次数计数器
            episode_count = 0
            while self.g_ep.value < self.max_episodes and episode_count < 10:  # 限制每个工作进程最多处理10个episode
                # 重置环境和缓冲区
                s = self.env.reset()
                buffer_s, buffer_disc_a, buffer_cont_a, buffer_r = [], [], [], []
                ep_r = 0
                done = False
                
                # 一个episode的交互循环
                max_steps = 200  # 限制每个episode的最大步数
                steps = 0
                
                while not done and steps < max_steps:
                    # 选择动作
                    try:
                        s_tensor = torch.FloatTensor(s).unsqueeze(0)  # 确保输入是二维的
                        action_result = self.local_net.choose_action(s_tensor)
                        
                        # 处理返回的动作数组
                        a = np.array(action_result).flatten()  # 确保是一维数组
                        
                        # 确保动作维度正确
                        expected_dim = self.local_net.discrete_dim + self.local_net.continuous_dim
                        if len(a) != expected_dim:
                            print(f"警告: 动作维度不匹配. 需要 {expected_dim}，得到 {len(a)}，扩展数组")
                            # 扩展动作数组
                            new_a = np.zeros(expected_dim)
                            new_a[:min(len(a), expected_dim)] = a[:min(len(a), expected_dim)]
                            a = new_a
                        
                        # 分离离散和连续动作
                        disc_a = a[:self.local_net.discrete_dim].copy()
                        cont_a = a[self.local_net.discrete_dim:].copy()
                        
                        # 二值化离散动作
                        disc_a = np.round(disc_a).astype(np.float32)
                        
                        # 限制连续动作在[0,1]范围内
                        cont_a = np.clip(cont_a, 0, 1).astype(np.float32)
                        
                        # 构造环境可接受的动作格式
                        N = self.env.N
                        M = self.env.M
                        
                        # 1. 提取训练决策和聚合决策
                        training_decision = disc_a[:N]
                        aggregation_decision = disc_a[N] if N < len(disc_a) else 0
                        
                        # 2. 构造边缘节点选择（简化为全1）
                        edge_selection = np.ones(M, dtype=np.float32)
                        
                        # 3. 构造完整的环境动作
                        env_action = np.zeros(N + 1 + M + M + N, dtype=np.float32)
                        env_action[:N] = training_decision                  # 训练决策
                        env_action[N] = aggregation_decision                # 聚合决策
                        env_action[N+1:N+1+M] = edge_selection              # 边缘节点选择
                        env_action[N+1+M:N+1+M+M+N] = cont_a[:M+N] if len(cont_a) >= M+N else np.pad(cont_a, (0, M+N-len(cont_a)), 'constant', constant_values=0.5)
                        
                        # 执行动作
                        s_, r, done = self.env.step(env_action)
                        
                        # 记录
                        ep_r += r
                        buffer_s.append(s)
                        buffer_disc_a.append(disc_a)
                        buffer_cont_a.append(cont_a)
                        buffer_r.append(r)
                        
                        # 更新全局网络
                        if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                            # 计算累积回报
                            v_s_ = 0 if done else self.local_net.forward(torch.FloatTensor(s_).unsqueeze(0))[3].item()
                            buffer_v_target = []
                            for r in buffer_r[::-1]:
                                v_s_ = r + GAMMA * v_s_
                                buffer_v_target.append(v_s_)
                            buffer_v_target.reverse()
                            
                            # 检查缓冲区是否为空
                            if len(buffer_s) > 0:
                                try:
                                    # 转换为tensor
                                    s_batch = torch.FloatTensor(np.vstack(buffer_s))
                                    
                                    # 处理动作数组
                                    try:
                                        disc_a_batch = torch.FloatTensor(np.vstack(buffer_disc_a))
                                        cont_a_batch = torch.FloatTensor(np.vstack(buffer_cont_a))
                                    except ValueError as e:
                                        print(f"处理动作数组维度时出错: {e}")
                                        print("手动构建一致维度的数组...")
                                        
                                        # 确定维度
                                        n_samples = len(buffer_disc_a)
                                        disc_dim = self.local_net.discrete_dim
                                        cont_dim = self.local_net.continuous_dim
                                        
                                        # 创建新数组
                                        disc_array = np.zeros((n_samples, disc_dim))
                                        cont_array = np.zeros((n_samples, cont_dim))
                                        
                                        # 填充数据
                                        for i, (disc, cont) in enumerate(zip(buffer_disc_a, buffer_cont_a)):
                                            # 确保动作是一维的
                                            disc_1d = np.array(disc).flatten()
                                            cont_1d = np.array(cont).flatten()
                                            
                                            # 复制有效数据
                                            disc_array[i, :min(len(disc_1d), disc_dim)] = disc_1d[:min(len(disc_1d), disc_dim)]
                                            cont_array[i, :min(len(cont_1d), cont_dim)] = cont_1d[:min(len(cont_1d), cont_dim)]
                                        
                                        # 转换为tensor
                                        disc_a_batch = torch.FloatTensor(disc_array)
                                        cont_a_batch = torch.FloatTensor(cont_array)
                                    
                                    v_target = torch.FloatTensor(np.array(buffer_v_target)[:, None])
                                    
                                    # 计算损失并更新全局网络
                                    try:
                                        loss, a_loss, c_loss = self.local_net.loss_func(s_batch, disc_a_batch, cont_a_batch, v_target)
                                        
                                        # 清零梯度
                                        self.optimizer.zero_grad()
                                        
                                        # 反向传播
                                        loss.backward()
                                        
                                        # 梯度裁剪，避免梯度爆炸
                                        torch.nn.utils.clip_grad_norm_(self.local_net.parameters(), 20.0)
                                        
                                        # 检查梯度是否有效
                                        valid_gradients = True
                                        for param in self.local_net.parameters():
                                            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                                                valid_gradients = False
                                                print(f"检测到无效梯度: {param.shape}")
                                                break
                                        
                                        if valid_gradients:
                                            # 将本地梯度传递给全局网络
                                            for lp, gp in zip(self.local_net.parameters(), self.global_net.parameters()):
                                                if lp.grad is not None:
                                                    if gp.grad is None:
                                                        gp.grad = lp.grad.clone()
                                                    else:
                                                        gp.grad += lp.grad.clone()
                                            
                                            # 更新全局网络
                                            self.optimizer.step()
                                        else:
                                            print("跳过更新，梯度无效")
                                        
                                        # 将全局参数同步回本地网络
                                        self.local_net.load_state_dict(self.global_net.state_dict())
                                    except Exception as e:
                                        print(f"计算损失或更新网络时出错: {e}")
                                        import traceback
                                        traceback.print_exc()
                                except Exception as e:
                                    print(f"处理批次数据时出错: {e}")
                            
                            # 清空缓冲区
                            buffer_s, buffer_disc_a, buffer_cont_a, buffer_r = [], [], [], []
                        
                        # 更新状态
                        s = s_
                        total_step += 1
                        steps += 1
                        
                    except Exception as e:
                        print(f"交互步骤时出错: {e}")
                        import traceback
                        traceback.print_exc()
                        
                        # 如果出现错误，跳到下一步
                        s = s_ if 's_' in locals() else self.env.reset()
                        steps += 1
                        
                # Episode结束，记录奖励
                with self.g_ep.get_lock():
                    self.g_ep.value += 1
                
                # 将奖励发送到结果队列
                self.res_queue.put((self.name, ep_r))
                print(f"{self.name} Ep: {self.g_ep.value}, Reward: {ep_r:.1f}")
                
                # 更新本地episode计数
                episode_count += 1
                
        except Exception as e:
            print(f"Worker {self.name} 出错: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            self.res_queue.put(None)  # 通知结束

class DRL:
    """深度强化学习代理 - 使用A3C算法进行联邦学习决策，并支持混合动作空间"""
    def __init__(self, env):
        self.env = env
        
        # 获取状态和动作空间维度
        try:
            # 状态空间
            self.state_dim = getattr(env, 'state_dim', 50)
            
            # 统一的动作空间表示
            self.N = getattr(env, 'N', 10)  # 终端设备数量
            self.M = getattr(env, 'M', 3)   # 边缘节点数量
            
            # 离散动作维度: N个训练决策 + 1个聚合决策
            self.num_actions_discrete = self.N + 1
            
            # 连续动作维度: M+N个资源分配参数
            self.num_actions_continuous = self.M + self.N
            
            # 总动作维度
            self.discrete_dim = self.num_actions_discrete
            self.continuous_dim = self.num_actions_continuous
            
            # 动作边界设置
            self.action_max = torch.ones(self.num_actions_discrete + self.num_actions_continuous)
            self.action_min = torch.zeros(self.num_actions_discrete + self.num_actions_continuous)
            
            print(f"初始化DRL代理: 状态维度={self.state_dim}, 离散={self.discrete_dim}, 连续={self.continuous_dim}")
            
            # 创建神经网络
            self.global_net = Net(self.state_dim, self.discrete_dim, self.continuous_dim)
            self.global_net.share_memory()
            self.optimizer = SharedAdam(self.global_net.parameters(), lr=1e-4)
            
            # 全局计数器和结果队列
            self.global_ep = mp.Value('i', 0)
            self.res_queue = mp.Queue()
            self.reward_history = []
            
            # 添加探索机制
            self.epsilon_start = 1.0
            self.epsilon_end = 0.01
            self.epsilon_decay = 6000
            self.epsilon = self.epsilon_start
            self.steps_done = 0
            
        except Exception as e:
            print(f"初始化DRL时出错: {e}")
            raise e
    
    # 原始的get_action保留，添加兼容统一接口的select_action
    def select_action(self, state):
        """统一接口: 选择动作"""
        # 实现ε-贪心探索
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        
        # 随机探索
        if np.random.random() < self.epsilon:
            # 随机生成动作
            N = getattr(self.env, 'N', 10)
            M = getattr(self.env, 'M', 3)
            # 生成随机动作
            disc_actions = (np.random.random(N+1) > 0.5).astype(np.float32)
            cont_actions = np.random.random(M+N).astype(np.float32)
            
            # 构造环境动作
            env_action = np.zeros(N + 1 + M + M + N)
            env_action[:N] = disc_actions[:N]  # 训练决策
            env_action[N] = disc_actions[N]    # 聚合决策
            env_action[N+1:N+1+M] = np.ones(M) # 边缘节点选择(全选)
            env_action[N+1+M:N+1+M+M+N] = cont_actions
            
            return env_action
            
        # 否则使用原有的get_action
        return self.get_action(state)
    
    def get_action(self, state):
        """原始的动作选择方法"""
        try:
            # 预处理状态
            if isinstance(state, list):
                state = np.array(state, dtype=np.float32)
            elif isinstance(state, np.ndarray):
                state = state.astype(np.float32)
            
            # 调整状态维度
            if len(state) != self.state_dim:
                temp = np.zeros(self.state_dim, dtype=np.float32)
                temp[:min(len(state), self.state_dim)] = state[:min(len(state), self.state_dim)]
                state = temp
            
            # 使用网络选择动作
            self.global_net.eval()
            with torch.no_grad():
                disc_probs, mu, sigma, _ = self.global_net(torch.FloatTensor(state).unsqueeze(0))
            
            # 处理离散动作
            disc_probs = disc_probs.numpy().flatten()
            N = getattr(self.env, 'N', 10)
            M = getattr(self.env, 'M', 3)
            
            # 训练决策和聚合决策
            training_decision = (disc_probs[:N] > 0.5).astype(np.int32)
            aggregation_decision = int(disc_probs[N] > 0.5)
            
            # 处理连续动作(资源分配)
            resource_allocation = np.clip(mu.numpy().flatten(), 0, 1)
            if len(resource_allocation) != M + N:
                temp = np.ones(M + N) * 0.5
                temp[:min(len(resource_allocation), M + N)] = resource_allocation[:min(len(resource_allocation), M + N)]
                resource_allocation = temp
            
            # 构造环境动作
            edge_selection = np.ones(M)  # 选择所有边缘节点
            
            env_action = np.zeros(N + 1 + M + M + N)
            env_action[:N] = training_decision
            env_action[N] = aggregation_decision
            env_action[N+1:N+1+M] = edge_selection
            env_action[N+1+M:N+1+M+M+N] = resource_allocation
            
            return env_action
            
        except Exception as e:
            print(f"获取动作时出错: {e}")
            
            # 返回安全的默认动作
            N = getattr(self.env, 'N', 10)
            M = getattr(self.env, 'M', 3)
            default_action = np.zeros(N + 1 + M + M + N)
            default_action[0] = 1  # 选择第一个客户端
            default_action[N] = 0  # 云端聚合
            default_action[N+1:N+1+M] = 1  # 选择所有边缘节点
            default_action[N+1+M:] = 0.5  # 中等资源分配
            return default_action
    
    # 添加统一的存储接口
    def store(self, state, action, reward, next_state, done):
        """统一接口: 存储经验
        A3C是无回放的在线算法，这个接口仅为兼容其他算法
        """
        pass  # A3C直接在Worker中处理，不需要在此存储
    
    # 添加统一的更新接口
    def update(self):
        """统一接口: 更新模型
        A3C通过Worker更新，这个接口仅为兼容其他算法
        """
        pass  # A3C在Worker中更新，不需要显式调用
    
    def train(self, max_episodes=5):
        """训练DRL代理使用A3C算法"""
        try:
            # 设置多进程启动方法
            # mp.set_start_method('spawn', force=True)  # Windows已默认是spawn
            
            # 确定使用CPU核心数
            num_workers = min(4, mp.cpu_count())
            print(f"使用 {num_workers} 个CPU核心并行训练")
            
            # 不再传递self.env_maker，而是在Worker内部创建环境
            # 创建环境参数字典，传递环境参数而不是环境工厂函数
            env_params = {'C': getattr(self.env, 'C', 10), 
                          'gama': getattr(self.env, 'gama', 0.01), 
                          'delta': getattr(self.env, 'delta', 1.0)}
            
            # 打印环境参数用于调试
            debug_mode = False  # 设置为True可输出更多调试信息
            if debug_mode:
                print("环境参数:")
                for k, v in env_params.items():
                    print(f"  {k}: {v}")
            
            # 创建并启动工作进程
            workers = [Worker(
                self.global_net, 
                self.optimizer, 
                self.global_ep, 
                self.res_queue, 
                i, 
                self.state_dim,
                self.discrete_dim,
                self.continuous_dim,
                max_episodes,
                env_params  # 传递环境参数而不是函数
            ) for i in range(num_workers)]
            
            [w.start() for w in workers]
            
            # 收集训练结果
            res = []
            while True:
                try:
                    r = self.res_queue.get(timeout=30)  # 添加30秒超时
                    if r is not None:
                        res.append(r)
                    else:
                        # 一个工作进程结束
                        num_workers -= 1
                        if num_workers <= 0:
                            break
                except:
                    print("等待工作进程结果超时，可能有些进程已卡住")
                    break
            
            # 结束所有工作进程（无论它们是否完成）
            for w in workers:
                if w.is_alive():
                    w.terminate()
            
            # 等待所有工作进程结束
            for w in workers:
                w.join(timeout=1.0)
                
            self.reward_history = res
            return res
            
        except Exception as e:
            print(f"训练DRL时出错: {e}")
            return []
    def save_model(self, path):
        """保存DRL模型到指定路径"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # 保存模型参数
            torch.save({
                'model_state_dict': self.global_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'reward_history': self.reward_history,
                'state_dim': self.state_dim,
                'discrete_dim': self.discrete_dim,
                'continuous_dim': self.continuous_dim
            }, path)
            
            print(f"模型已保存到: {path}")
            return True
        except Exception as e:
            print(f"保存模型时出错: {e}")
            return False
    
    def load_model(self, path):
        """从指定路径加载模型"""
        try:
            if not os.path.exists(path):
                print(f"模型文件不存在: {path}")
                return False
                
            # 加载模型
            checkpoint = torch.load(path)
            
            # 加载模型参数
            self.global_net.load_state_dict(checkpoint['model_state_dict'])
            
            # 如果存在优化器状态，也加载
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
            # 加载其他参数
            if 'reward_history' in checkpoint:
                self.reward_history = checkpoint['reward_history']
                
            print(f"成功加载模型: {path}")
            return True
        except Exception as e:
            print(f"加载模型时出错: {e}")
            print("使用默认初始化模型")
            return False

def fix_shape(tensor, target_shape):
    # 简单有效的维度修复函数
    if tensor.shape != target_shape:
        if tensor.numel() == torch.tensor(target_shape).prod().item():  # 元素数量相同
            return tensor.reshape(target_shape)
        else:  # 元素数量不同，使用广播
            return tensor.reshape(-1)[0].expand(target_shape)
    return tensor
