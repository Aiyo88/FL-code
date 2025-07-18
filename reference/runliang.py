from cmath import cos
import os
import click
import ast
import time
import numpy as np
import gym
import datetime
from torch.distributions import Categorical

import torch
# 引入不同的算法
from agents.pdqn import PDQNAgent
from elsedrl.DDPG import DDPG
# from elsedrl.PPO import PPOA
from elsedrl.TD3 import TD3
from elsedrl.PPO import PPO
# from elsedrl.SAC import SAC
# from env.aoienv import AoIEnv
# from env.aoienv_local import AoIEnv_local
# from env.aoienv_upload import AoIEnv_upload
from our.env import EdgeComputingEnv
from our.OffLoadingEnv import OffLoadingEnv
from newd3qn import Dueling_DQN

import matplotlib.pyplot as plt


def plot_training_results(log_file_name,log_name,auto):
    # 从日志文件中读取数据
    episodes, epoch_rewards  = [], [] 
    with open(log_file_name, 'r') as log_file:
        lines = log_file.readlines()
    
    for line in lines:
        if 'episode:' in line:
            # 解析数据
            data = line.split('\t')
            episode = int(data[0].split(':')[-1].strip())
            epoch_reward = float(data[1].split(':')[-1].strip())
            # epoch_cost = float(data[2].split(':')[-1].strip())
            # 存储数据
            episodes.append(episode)
            epoch_rewards.append(epoch_reward)
            # epoch_costs.append(epoch_cost)
    # 创建新图形
    plt.figure() # 否则每次都是在之前的基础上绘制了
    # 绘制图表
    # plt.plot(episodes, epoch_costs)
    plt.plot(episodes, epoch_rewards)
    # plt.plot(episodes, epoch_rewards, marker='o', linestyle='-', color='b')
    plt.xlabel('Episode')
    plt.ylabel('Epoch Reward')
    plt.title('Epoch Reward vs. Episode')
    plt.grid(True)
    if auto:
        plt.savefig("gaohan/pic/auto/"+log_name+".png")
    else:
        plt.savefig("gaohan/pic/debug/"+log_name+".png")
    if not auto:
        plt.show()
    # 关闭当前图形，确保下次调用时创建新图形
    plt.close()


    episodes, epoch_costs  = [], [] 
    with open(log_file_name, 'r') as log_file:
        lines = log_file.readlines()
    
    for line in lines:
        if 'episode:' in line:
            # 解析数据
            data = line.split('\t')
            episode = int(data[0].split(':')[-1].strip())
            epoch_cost = float(data[2].split(':')[-1].strip())
            episodes.append(episode)
            epoch_costs.append(epoch_cost)
            # epoch_costs.append(epoch_cost)
    # 创建新图形
    plt.figure() # 否则每次都是在之前的基础上绘制了
    # 绘制图表
    # plt.plot(episodes, epoch_costs)
    plt.plot(episodes, epoch_costs)
    # plt.plot(episodes, epoch_rewards, marker='o', linestyle='-', color='b')
    plt.xlabel('Episode')
    plt.ylabel('Epoch Cost')
    plt.title('Epoch Cost vs. Episode')
    plt.grid(True)
    if auto:
        plt.savefig("gaohan/pic/cost/"+log_name+".png")
    else:
        plt.savefig("gaohan/pic/debug/"+log_name+".png")
    if not auto:
        plt.show()
    # 关闭当前图形，确保下次调用时创建新图形
    plt.close()



class ClickPythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except Exception as e:
            print(e)
            raise click.BadParameter(value)

# pdqn: seed: 10 , max_steps 150 , batch-szie 256 ,gamma 0.85 

# @click.command()
# # None就是随机的，有数值就是可以重复的
# @click.option('--seed', default=10, help='Random seed.', type=int)
# # 正常按照1200来
# @click.option('--episodes', default=680, help='Number of epsiodes. default : 500 ', type=int)
# @click.option('--max-steps', default=150, help='Number of epsiodes. default : 150', type=int)
# # 128
# @click.option('--batch-size', default=256, help='Minibatch size.', type=int)
# # 较小的折扣因子倾向于更关注即时奖励，而较大的折扣因子会更注重长期规划。
# @click.option('--gamma', default=0.9, help='Discount factor.', type=float) # 对未来奖励的重视程度，高些好
# @click.option('--update-iteration', default=20, help="更新critic网络的频率 一般就是100 或者 200左右", type=int)
# @click.option('--inverting-gradients', default=True,
#               help='Use inverting gradients scheme instead of squashing function.', type=bool)
# @click.option('--initial-memory-threshold', default=128, help='Number of transitions required to start learning.',
#               type=int)
# @click.option('--use-ornstein-noise', default=True,
#               help='Use Ornstein noise instead of epsilon-greedy with uniform random exploration.', type=bool)
# @click.option('--replay-memory-size', default=20000, help='Replay memory transition capacity.', type=int)
# @click.option('--epsilon-steps', default=1000, help='Number of episodes over which to linearly anneal epsilon.', type=int)
# @click.option('--epsilon-final', default=0.01, help='Final epsilon value.', type=float)
# @click.option('--tau-actor', default=0.1, help='Soft target network update averaging factor.', type=float)
# @click.option('--tau-actor-param', default=0.001, help='Soft target network update averaging factor.', type=float)
# # 学习率小些会更好
# @click.option('--learning-rate-actor', default=0.0001, help="Actor network learning rate.", type=float)
# @click.option('--learning-rate-actor-param', default=0.00001, help="Critic network learning rate.", type=float)
# @click.option('--reward-scale', default=1./50., help="Reward scaling factor.", type=float)
# @click.option('--clip-grad', default=1., help="Parameter gradient clipping limit.", type=float)
# @click.option('--indexed', default=False, help='Indexed loss function.', type=bool)
# @click.option('--weighted', default=False, help='Naive weighted loss function.', type=bool)
# @click.option('--average', default=False, help='Average weighted loss function.', type=bool)
# @click.option('--random-weighted', default=False, help='Randomly weighted loss function.', type=bool)
# @click.option('--zero-index-gradients', default=False, help="Whether to zero all gradients for action-parameters not corresponding to the chosen action.", type=bool)
# @click.option('--layers', default="(256,)", help='Hidden layers.', cls=ClickPythonLiteralOption)
# @click.option('--action-input-layer', default=0, help='Which layer to input action parameters.', type=int)
# @click.option('--title', default="DDPG", help="Prefix of output files", type=str)
# @click.option('--auto', default=False, help='if atuo run  all paratemers .', type=bool) # 是否是自动化执行

# @click.option('--numbers-edge-server', default=5, help='', type=int)
# @click.option('--numbers-service', default=3, help='', type=int)
# @click.option('--numbers_task', default=3, help='', type=int)

# @click.option('--alg-name', default="PDQN", help="算法的策略", type=str) # PDQN就是正常 random就是随机 greedy是贪心算法


def run(seed, episodes, max_steps, batch_size, gamma, update_iteration ,inverting_gradients, initial_memory_threshold,
        replay_memory_size, epsilon_steps, epsilon_final, tau_actor, tau_actor_param, use_ornstein_noise,
        learning_rate_actor, learning_rate_actor_param, reward_scale, clip_grad, title,
        zero_index_gradients, layers, indexed, weighted, average, random_weighted,
        action_input_layer,auto,
        numbers_edge_server,numbers_service,numbers_task,alg_name,lambda_s,theta_s):



    ''' 本地还是全部卸载'''

    
    envpdqn = EdgeComputingEnv(numbers_edge_server=numbers_edge_server, numbers_service=numbers_service)
    envd3qn = OffLoadingEnv(numbers_edge_server,numbers_service,numbers_task,lambda_s,theta_s)



    # 可以在这里先搞一下动作空间多少个，状态空间多少个，方便适配不同的环境体
    num_actions_discrete = envpdqn.action_space.spaces[0].nvec.shape[0]#  默认动作空间中离散参数的维度
    action_parameter_sizes = np.array([ ( envpdqn.action_space.spaces[1].shape[0] *  envpdqn.action_space.spaces[1].shape[1] ) ]) # 这里应该是连续的动作空间的数量，二维
    num_actions_continuous_1 = action_parameter_sizes[0] 
    # num_actions_continuous_2 = action_parameter_sizes[1] 
    # 总的动作个数的数量
    total_num_actions = num_actions_discrete + num_actions_continuous_1 # + num_actions_continuous_2  # 离散的动作+连续的动作 总的数量

    
    shapes = [space.shape for space in envpdqn.observation_space]# 获取每个子空间的形状
    flat_shape = (np.sum([np.prod(shape) for shape in shapes]),)# 计算连接后的形状
    process_observation_space = gym.spaces.Box(low=0, high=1, shape=flat_shape, dtype=np.float32)# 创建一个连接后的 Box 空间


    '''
    第二个的d3qn
    '''
    agent_d3qn = Dueling_DQN(
        n_features = envd3qn.observation_space.shape[0],
        n_actions = envd3qn.action_space.n,
        batch_size = batch_size,
        buffer_limit = 50000,
        gamma = gamma,
        learning_rate = learning_rate_actor,
        seed=seed
    ) # 就是trainer


    if alg_name == 'PDQN' or alg_name == 'RANDOM':
        max_steps = 150
        # seed = 2
        agent = PDQNAgent(
                       # 这里我的状态空间是两维的
                       observation_space=process_observation_space, 
                    #    observation_space=env.observation_space, 
                       action_space=envpdqn.action_space,
                       batch_size=batch_size,
                       learning_rate_actor=learning_rate_actor,  # 0.0001
                       learning_rate_actor_param=learning_rate_actor_param,  # 0.001
                       epsilon_steps=epsilon_steps,
                       epsilon_final=epsilon_final,
                       gamma=gamma,
                       clip_grad=clip_grad,
                       indexed=indexed,
                       average=average,
                       random_weighted=random_weighted,
                       tau_actor=tau_actor,
                       weighted=weighted,
                       tau_actor_param=tau_actor_param,
                       initial_memory_threshold=initial_memory_threshold,
                       use_ornstein_noise=use_ornstein_noise,
                       replay_memory_size=replay_memory_size,
                       inverting_gradients=inverting_gradients,
                       actor_kwargs={'hidden_layers': layers, 'output_layer_init_std': 1e-5,
                                     'action_input_layer': action_input_layer,},
                       actor_param_kwargs={'hidden_layers': layers, 'output_layer_init_std': 1e-5,
                                           'squashing_function': False},
                       zero_index_gradients=zero_index_gradients,
                       seed=seed)
    elif alg_name == 'DDPG':
        max_steps = 150
        # seed = 19

        agent = DDPG(
                    observation_space=process_observation_space, 
                    action_space=envpdqn.action_space,
                    state_dim = process_observation_space.shape[0],
                    action_dim = total_num_actions , # 总的动作空间的个数的数量
                    max_action = 1 , # 先借助ddpg原来的 -1 1 区间，最后在ddpg中的选择action函数中调整吧
                    batch_size= batch_size,
                    gamma= gamma,
                    lr=  learning_rate_actor,
                    lr_c = learning_rate_actor_param,
                    tau = tau_actor,
                    capacity=replay_memory_size,
                    exploration_noise = 0.1,
                    seed = seed,
                    update_iteration = update_iteration,
                    num_actions_discrete = num_actions_discrete,# 传入额外的参数值 供action选择时候使用
                    num_actions_continuous_1 =num_actions_continuous_1,
                )
    elif alg_name == 'TD3':
        max_steps = 150
        # seed = 19
        agent = TD3(
                    observation_space=process_observation_space, 
                    action_space=envpdqn.action_space,
                    state_dim = process_observation_space.shape[0],
                    action_dim = total_num_actions , # 总的动作空间的个数的数量
                    max_action = 1 , # 先借助ddpg原来的 -1 1 区间，最后在ddpg中的选择action函数中调整吧
                    batch_size=  batch_size,
                    gamma=gamma,
                    lr=  learning_rate_actor,
                    lr_c = learning_rate_actor_param,
                    tau = tau_actor,
                    capacity=replay_memory_size,
                    exploration_noise = 0.1,
                    seed =  seed,
                    update_iteration = update_iteration,
                    num_actions_discrete = num_actions_discrete,# 传入额外的参数值 供action选择时候使用
                    num_actions_continuous_1 =num_actions_continuous_1,
                    policy_noise = 0.2,
                    policy_delay = 2
        )
    elif alg_name == 'PPO':
        max_steps = 150
        # agent = PPO(
        #             observation_space=process_observation_space, 
        #             action_space=envpdqn.action_space,
        #             state_dim = process_observation_space.shape[0],
        #             action_dim = total_num_actions , # 总的动作空间的个数的数量
        #             batch_size=batch_size,
        #             gamma= 0.99 ,  # gamma,
        #             seed = seed,
        #             lr= 0.0003 ,# learning_rate_actor,
        #             lr_c = 0.001, # learning_rate_actor_param,
        #             tau = tau_actor,
        #             capacity=replay_memory_size,
        #             exploration_noise = 0.1,
        #             update_iteration = update_iteration, #update policy for K epochs in one PPO update
        #             # 新增
        #             # action_std = 0.6,
        #             # action_std_decay_rate = 0.05, # linearly decay action_std (action_std = action_std - action_std_decay_rate)
        #             # min_action_std = 0.1 ,# minimum action_std (stop decay after action_std <= min_action_std)
        #             # action_std_decay_freq = int(2.5e5),
        #             eps_clip = 0.2,          # clip parameter for PPO
        # )
        agent = PPO(
                    observation_space=process_observation_space, 
                    action_space=envpdqn.action_space,
                    n_actions=total_num_actions, 
                    batch_size=  batch_size,
                    gamma=gamma,
                    lr=learning_rate_actor,
                    lr_c = learning_rate_actor_param,
                    update_iteration = update_iteration,
                    alpha=0.00001, 
                    seed =   seed,
                    input_dims=process_observation_space.shape[0])

    print("agent",agent)
    # 日志记录
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_name = f'log_{alg_name}_lambda_{lambda_s}_theta_{theta_s}'
    if auto:
        '''自动化的  '''
        log_file_name = f'gaohan/logs/auto/{log_name}.txt'
        os.makedirs('gaohan/logs/auto/', exist_ok=True)  # 创建logs目录，如果不存在
    else:
        log_file_name = f'gaohan/logs/debug/{log_name}.txt'
        os.makedirs('gaohan/logs/debug/', exist_ok=True)  # 创建logs目录，如果不存在


    max_steps = max_steps
    total_reward = []
    start_time = time.time()
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    returns = []
    returns_costssss = []

    for epoch in range(episodes):

        state = envpdqn.reset()
        state_d3qn = envd3qn.reset()
        


        
        episode_reward = 0.
        episode_cost = 0.


        for j in range(max_steps):

            # next_state, reward, cost  = envpdqn.step(action)

            # action = agent.select_action(state)
            if alg_name != 'PPO' and  alg_name != 'RANDOM' :
                action = agent.select_action(state)
            elif alg_name == 'PPO' :
                ''' orgin 是需要存储的'''
                action,prob  = agent.select_action(state)
            elif alg_name == 'RANDOM':
                action = agent.random_action(state)
        
            action_d3qn = agent_d3qn.sample_action(torch.from_numpy(state_d3qn).float())

            # 根据agent 选择不同的动作


            next_state, reward , cost1 = envpdqn.step(action)
            '''
                d3qn
            '''
            next_state_d3qn, reward_d3qn, cost_d3qn = envd3qn.step(action_d3qn,action)
            

            
            ''' 每个智能体存储经验的部分 '''
            if alg_name == 'PDQN' : 
                agent.store(state, action, reward, next_state, False)
            elif alg_name == 'DDPG':
                agent.replay_buffer.push((state, next_state, action, reward, False))
            elif alg_name == 'TD3':
                agent.memory.push((state, next_state, action, reward, False))
            elif alg_name == 'PPO':
                agent.remember(state, next_state, action, prob, reward, False)

            
            agent_d3qn.memory.put((state_d3qn, action_d3qn, reward_d3qn, next_state_d3qn, 0.0))

 
            state = next_state
            # episode_reward += (reward)
            # episode_cost += ( cost1)
            episode_reward += (reward+reward_d3qn)
            episode_cost += (cost_d3qn + cost1)


        if alg_name != 'PDQN' and alg_name != 'RANDOM':
            ''' 最好设置更新频率，不要每次都更新，这样太差了'''
            if epoch !=0 and epoch % 50 ==0  and alg_name!='TD3' and alg_name!='PPO':
                agent.update() 
                # 每20轮更新一下
            # elif epoch !=0 and epoch % 20 and alg_name!='TD3' == 0 : 
            elif alg_name=='TD3' or alg_name =='PPO':
                agent.update()
                # None
        if agent_d3qn.memory.size() > 2000:
            agent_d3qn.train()
            agent_d3qn.target_net.load_state_dict(agent_d3qn.evaluate_net.state_dict())

        '''
            异常值的处理
        '''
        # print("returns11 ", returns)
        if epoch > 2 and abs(episode_reward - np.mean(returns) ) > abs(returns[0]):
            episode_reward = returns[-1] # @等于倒数第二个值
            episode_cost = returns_costssss[-1]

        returns.append(episode_reward) # return是干啥的，好像也没用
        # print("returns ", returns)
        returns_costssss.append(episode_cost)


        # if alg_name == 'PDQN':
        #     agent.end_episode()


        log_line = 'episode: {}\t epoch_reward: {}\t epoch_cost: {}\t \n'.format( epoch, episode_reward, episode_cost)
        # log_line = 'episode: {}\t epoch_reward: {}\t epoch_cost: {}\t epoch_aoi: {}\t \n'.format( epoch, episode_reward, episode_cost,episode_aoi)
        # 日志记录
        with open(log_file_name, 'a') as log_file:
            log_file.write(log_line)

        # if epoch % 100 == 0: 
        if auto :
            if epoch % 100 == 0 :
                print(log_line)
        else : 
            print(log_line)

    end_time = time.time()
    print("Training took %.2f seconds" % (end_time - start_time))
    # env.close()

    # 绘图
    plot_training_results(log_file_name,log_name,auto)


    

if __name__ == '__main__':
    run()
