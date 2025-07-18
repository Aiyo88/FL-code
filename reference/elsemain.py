import os
# 设置环境变量避免OpenMP重复初始化错误
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import math
import random
import time

import numpy as np
import torch
import DDPG
import DDQN
# import Dueling_DQN  # 暂时注释掉，因为Dueling_DQN模块还未实现
#import PPO
import TD3
import plot
from utils import *

print(torch.__version__)
print(torch.version.cuda)
print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

max_episode = 200


class env:
    def __init__(self):
        # system model
        self.M = 4
        self.T = 100

        # mobility model
        self.speed = range(27, 19, -1)
        # self.speed = np.random.randint(20, 28)
        # self.speed = [20]
        self.speed_max = 28
        self.n = 40  # 最大车辆数
        self.vehicle_numbers = list(range(1, 41))  # 用于测试的车辆数量列表
        self.alive_number = []

        # task model
        self.rho = 0.6
        # self.task_size = 5  # MB
        self.task_size = np.random.randint(2, 6)  # MB

        # communication model
        self.B_rsu = 1e6  # 修改为 1e6 而不是 1 * 10e6
        self.B_mbs = 10e6  # 修改为 10e6 而不是 10 * 10e6
        # self.B_mbs = np.random.randint(10, 41) * 1e6 # bps
        self.noise_v2r = 1e-14  # 修改为 1e-14 而不是 10e-14
        self.noise_r2b = 1e-12  # 修改为 1e-12 而不是 10e-12
        self.g_v2r = 5e-6  # 修改为 5e-6 而不是 5 * 10e-6
        self.g_r2b = np.random.uniform(1, 7, size=self.M) * 1e-6
        self.power_v = 1  # W
        self.power_r = 10  # W

        # computation model
        self.f_v = 5e7  # Hz
        self.f_rsu = 5e8  # Hz
        # self.f_rsu = np.random.randint(2, 6)  # MB

        self.c = 300  # cycles/bit

        # time model
        self.time_slot = 0.1

        # cost model
        self.price_r = 0.0003  # $/bit
        self.price_B = 0.002  # $/Hz
        self.price_c = 0.00015  # $/bit
        self.cost = []

        # --------------------- DRL model ---------------------
        self.state = []
        self.action = []
        self.offload_action = []
        self.load_slice_action = []
        self.B_slice_action = []
        self.reward = []
        self.next_state = []
        # PPO parameters
        self.action_log_prob = []
        self.mu = []
        self.sigma = []

        # delay model
        self.rate_r2b_average = 15000000

        self.v_num = 1
        self.load = []
        self.delay = 0
        self.delay_all_vehicle_list = []
        self.delay_all_rsu_list = []
        self.delay_all_cloud_list = []
        self.cost_all_vehicle_list = []
        self.cost_all_rsu_list = []
        self.cost_all_cloud_list = []

        self.rate_v2r = self.B_rsu / self.v_num * math.log(1 + self.power_v * self.g_v2r / self.noise_v2r) / self.c


# --------------------- Tool function ---------------------
def action_normalize():
    env.offload_action, env.load_slice_action, env.B_slice_action = split_list(env.action)

    # Convert continuous variables to binary variables
    for i in range(env.M):
        if env.offload_action[i] <= 0.5:
            env.offload_action[i] = 0
        else:
            env.offload_action[i] = 1

    # If task is not offloaded to the cloud, there is no need to split task and allocate bandwidth to rsu
    for i in range(env.M):
        if env.offload_action[i] == 0:
            env.load_slice_action[i] = 0
            env.B_slice_action[i] = 0
        else:
            if env.load_slice_action[i] == 0:
                env.B_slice_action[i] = 0

    # B_slice_action normalize
    temp = sum(env.B_slice_action)

    if temp != 0:
        env.B_slice_action = [env.B_slice_action[i] / temp for i in range(env.M)]

    env.action = env.offload_action + env.load_slice_action + env.B_slice_action
    pass


def get_ppo_action():
    # print(env.action)
    i = len(env.action) // 3
    j = len(env.action) // 3 * 2
    env.offload_action = env.action[:i]
    env.load_slice_action = env.action[i:j]
    env.B_slice_action = env.action[j:]

    # Convert continuous variables to binary variables
    env.offload_action = [0 if env.offload_action[i] < env.mu[i] else 1 for i in range(env.M)]

    # If task is not offloaded to the cloud, there is no need to split task and allocate bandwidth to rsu
    # for i in range(env.M):
    #     if env.offload_action[i] == 0:
    #     else:
    #         if env.load_slice_action[i] == 0:
    #             env.B_slice_action[i] = 0


    # Make load_slice_action in the range of [0,1]
    for i in range(env.M):
        if env.offload_action[i] != 0:  # 如果卸载计算
            left = (env.mu[i] - 3 * env.sigma[i])[0][0].item()
            length = (env.sigma[i] * 6)[0][0].item()
            env.load_slice_action[i] = (env.load_slice_action[i] - left) / length
        else:  # 如果本地计算
            env.load_slice_action[i] = 0
            env.B_slice_action[i] = 0


    # Make B_slice_action in the range of [0,1] and normalize
    for i in range(env.M):
        if env.B_slice_action[i] != 0:
            left = (env.mu[i] - 3 * env.sigma[i])[0][0].item()
            length = (env.sigma[i] * 6)[0][0].item()
            env.B_slice_action[i] = (env.B_slice_action[i] - left) / length
            env.B_slice_action[i] = np.clip(env.B_slice_action[i], 0, 1)
        else:
            env.B_slice_action[i] = 0


    temp = sum(env.B_slice_action)
    if temp != 0:
        env.B_slice_action = [env.B_slice_action[i] / temp for i in range(env.M)]

    env.action = env.offload_action + env.load_slice_action + env.B_slice_action
    pass


# --------------------- Environment parameters function ---------------------

def ger_workload(vehicle_number):
    load = []
    env.alive_number = [0] * env.M
    for i in range(env.M):
        temp = 0
        j = 0
        while j < vehicle_number:
            r = random.random()
            if r >= (1-env.rho):
                env.alive_number[i] += 1
                temp += env.task_size
            j += 1
        load.append(temp)
    return load


def get_delay_and_cost(path, record=True):
    # primary cost and cost of each rsu
    global noise_r2b, primary_cost

    # Load of each layer
    load_v = []
    load_r = []
    load_c = []

    # Delay of each layer
    delay_local = []
    delay_v2r = []
    delay_rsu = []
    delay_r2b = []

    rate_r2b = []

    env.load = [float(env.load[i] * 1024 * 8) for i in range(env.M)]  # 确保转换为浮点数
    # env.load = [env.load[i] * 8 for i in range(env.M)]

    for i in range(env.M):
        # local
        if env.offload_action[i] == 0:
            delay_local.append(float(env.load[i]) / env.f_v * env.c)  # 确保使用浮点数除法

            delay_v2r.append(0)
            delay_rsu.append(0)
            delay_r2b.append(0)
            rate_r2b.append(0)

            load_v.append(env.load[i])
            load_c.append(0)
            load_r.append(0)
            continue
        # offload
        else:
            # 1.V2R delay
            delay_local.append(0)
            delay_v2r.append(env.load[i] / env.rate_v2r)  # s
            load_v.append(0)

            # 如果全部RSU计算
            if env.load_slice_action[i] == 0 or env.B_slice_action[i] == 0:

                # ----------- load -----------
                load_r.append(env.load[i])
                load_c.append(0)

                # ----------- delay -----------
                rate_r2b.append(0)
                delay_r2b.append(0)

                # 1.V2R delay
                delay_v2r.append(env.load[i] / env.rate_v2r)  # s

                # 2.RSU local computing delay
                delay_rsu.append(env.load[i] * env.c / env.f_rsu)  # s

            # 如果有部分云计算
            else:
                # ----------- load -----------
                load_c.append(env.load[i] * env.load_slice_action[i])
                load_r.append(env.load[i] - load_c[i])

                # ----------- delay -----------
                rate_r2b_ = env.B_mbs * env.B_slice_action[i] * math.log(
                    1 + env.power_r * env.g_r2b[i] / env.noise_r2b) / env.c  # bit/s

                rate_r2b.append(rate_r2b_)  # bit/s
                delay_r2b.append(load_c[i] / rate_r2b_)
                delay_rsu.append(load_r[i] * env.c / env.f_rsu)  # s

    # ---------------------------------- Comparison algorithm ----------------------------------
    # 1.computed by vehicle totally
    delay_all_vehicle = [env.load[i] / env.f_v * env.c for i in range(env.M)]

    # 2.computed by rsu totally
    delay_all_rsu = [(env.load[i] / env.f_rsu * env.c) + (env.load[i] / env.rate_v2r) for i in range(env.M)]

    # 3.computed by cloud totally
    delay_all_cloud = [(env.load[i] / env.rate_v2r) + (env.load[i] / env.rate_r2b_average) for i in range(env.M)]

    # ---------------------------------- Total delay ----------------------------------
    total_delay = []
    for i in range(env.M):
        if env.offload_action[i] == 0:
            total_delay.append(delay_local[i])
        else:
            total_delay.append(delay_v2r[i] + max(delay_rsu[i], delay_r2b[i]))

    cost = [load_r[i] * env.price_r + load_c[i] * (env.price_c + env.price_B) for i in range(env.M)]

    def record_detail():
        if not path:  # 如果路径为空，不记录日志
            return
        log("load\t\t", env.load, path)
        log("load_veh\t", load_v, path)
        log("load_rsu\t", load_r, path)
        log("load_cloud\t", load_c, path)

        log("rate_v2r", env.rate_v2r, path)
        log("rate_r2b", rate_r2b, path)

        log("offload_action\t\t", env.offload_action, path)
        log("load_slice_action\t", env.load_slice_action, path)
        log("B_slice_action\t\t", env.B_slice_action, path)
        log("f_rsu\t\t", env.f_rsu, path)

        log("delay_local\t", delay_local, path)
        log("delay_v2r\t\t", delay_v2r, path)
        log("delay_rsu\t\t", delay_rsu, path)
        log("delay_r2b\t\t", delay_r2b, path)

        log("------cost\t\t", cost, path)
        log("------total_delay\t\t", total_delay, path)
        log("------delay_all_vehicle\t", delay_all_vehicle, path)
        log("------delay_all_rsu\t\t", delay_all_rsu, path)
        log("------delay_all_cloud\t", delay_all_cloud, path)
        Qos = [env.time_slot * env.alive_number[i] for i in range(env.M)]
        log("------alive_number\t", env.alive_number, path)
        log("------Qos\t", Qos, path)

    if record:  # 只在需要时记录日志
        record_detail()

    return total_delay, cost  # M M


# return:RSU加惩罚的平均处理时延
def get_reward(delay, cost):
    reward = []
    for i in range(env.M):
        # if delay[i] > env.time_slot * env.alive_number[i]:
        #     reward.append(100 + delay[i])
        # else:
        #     reward.append(delay[i])
        if delay[i] > env.time_slot * env.alive_number[i]:
            reward.append(1000 + cost[i])
        else:
            reward.append(cost[i])
    return reward  # M


# --------------------- Record function ---------------------


def record_per_t(path):
    # log("state", env.state, "log.md")
    # log("action\t", env.action, "log.md")
    log("reward\t", env.reward, path)
    # log("next_state\t", env.next_state, "log.md")


def record_per_episode(path):
    log(
        "--------------------------------------------------------- episode ---------------------------------------------------------",
        "", path)
    log("vehicle number", env.v_num, path)
    log("load", env.load, path)
    log("v2r channel gain", env.g_v2r, path)


# def dqn_main(env, mode):

#     agent = DDQN.Agent(env.M * 3, env.M * 3, 1.0)  # -------3M

#     if mode == 'test':
#         for speed in env.speed:
#             agent.load(speed)

#             t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
#             path = "./log/DQN/test/speed_{}-{}.md".format(speed, t)

#             reward_list = []

#             delay_per_vehicle_list = []
#             cost_per_vehicle_list = []
#             reward_per_vehicle_list = []

#             total_alive_number = 0
#             for j in range(max_episode):
#                 total_delay = 0
#                 total_cost = 0
#                 total_reward = 0

#                 # Initialize environment
#                 env.v_num = math.ceil((1 - speed / env.speed_max) * env.n)
#                 env.load = ger_workload(env.v_num)  # M
#                 env.state = np.hstack((env.load, env.g_r2b, [0] * env.M))  # -------3M
#                 cost_list = []
#                 for t in range(env.T):
#                     env.action = agent.select_action(env.state)
#                     env.action = (env.action + np.random.normal(0, DDPG.exploration_noise, size=env.M * 3)).clip(0, 1)  # 3M
#                     action_normalize()

#                     env.delay, env.cost = get_delay_and_cost(path)  # M M
#                     env.reward = get_reward(env.delay, env.cost)  # M
#                     print(env.reward)

#                     total_delay += sum(env.delay)  # 1
#                     total_cost += sum(env.cost)  # 1
#                     total_reward += sum(env.reward)  # 1
#                     total_alive_number += sum(env.alive_number)  # 1

#                     env.load = ger_workload(env.v_num)  # M
#                     env.next_state = np.hstack((env.load, env.g_r2b, env.delay))  # -------3M
#                     env.state = env.next_state

#                 # plot.save_to_csv(cost_list, "time slot", "cost", "result/cost_time slot/" + "/TD3.csv")

#                 delay_per_vehicle_list.append(total_delay / total_alive_number / env.T)
#                 cost_per_vehicle_list.append(total_cost / total_alive_number / env.T)
#                 reward_per_vehicle_list.append(total_reward / env.T)
#                 #
#             # t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
#             # path_a = "./csv/DDPG/test/speed_{}/{}".format(speed, t)
#             # if not os.path.exists(path_a):
#             #     os.makedirs(path_a)

#             # plot.save_to_csv(delay_per_vehicle_list, "episode", "delay", path_a + "/DDPG_delay.csv")
#             # plot.save_to_csv(cost_per_vehicle_list, "episode", "cost", path_a + "/DDPG_cost.csv")
#             # plot.save_to_csv(reward_per_vehicle_list, "episode", "reward", path_a + "/DDPG_reward.csv")
#             # plot.save_to_csv(reward_list, "episode", "reward", "./csv/DDPG/speed_{}-{}.csv".format(speed, t))
#             # print("speed = {}".format(speed))
#             # print("number = {}".format(env.v_num))
#             # print("alive = {}".format(env.alive_number))
#             # print(len(delay_per_vehicle_list))
#             print(np.mean(delay_per_vehicle_list))
#             # print(np.mean(cost_per_vehicle_list))
#             # print(np.mean(reward_per_vehicle_list))

#     elif mode == 'train':
#         for speed in env.speed:
#             t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
#             path = "./log/DDQN/speed_{}-{}.md".format(speed, t)

#             total_step = 0
#             reward_list = []

#             log("speed", speed, path)

#             delay_per_episode_list = []
#             cost_per_episode_list = []
#             reward_per_episode_list = []

#             for j in range(max_episode):
#                 step = 0
#                 total_delay = 0
#                 total_cost = 0
#                 total_reward = 0

#                 # Initialize environment
#                 env.v_num = math.ceil((1 - speed / env.speed_max) * env.n)
#                 env.load = ger_workload(env.v_num)  # M
#                 env.state = np.hstack((env.load, env.g_r2b, [0] * env.M))  # -------3M
#                 record_per_episode(path)

#                 for t in range(env.T):
#                     log("--------------------------------- t -------------------------------", "", path)
#                     env.action = agent.choose_action(env.state)
#                     env.action = (env.action + np.random.normal(0, DDPG.exploration_noise, size=env.M * 3)).clip(0, 1)  # 3M
#                     action_normalize()

#                     env.delay, env.cost = get_delay_and_cost(path)  # M M
#                     env.reward = get_reward(env.delay, env.cost)  # M
#                     # env.reward = sum(env.delay) / env.M  # 1

#                     total_delay += sum(env.delay)  # 1
#                     total_cost += sum(env.cost)  # 1
#                     total_reward += sum(env.reward)  # 1

#                     env.load = ger_workload(env.v_num)  # M
#                     env.next_state = np.hstack((env.load, env.g_r2b, env.delay))  # -------3M
#                     record_per_t(path)

#                     agent.replay_buffer.push(env.state, env.action, env.next_state, sum(env.reward))
#                     env.state = env.next_state
#                     step += 1

#                 delay_per_episode_list.append(total_delay / env.M / env.T)
#                 cost_per_episode_list.append(total_cost / env.M / env.T)
#                 reward_per_episode_list.append(total_reward / env.M / env.T)

#                 total_step += step + 1

#                 average_t_reward = total_reward / env.T / env.M
#                 reward_list.append(average_t_reward)

#                 print("Episode: {}/{}\tAverage reward: {:0.5f}".format(j, max_episode, average_t_reward))

#                 if agent.memory_counter >= DDQN.MEMORY_CAPACITY:
#                     agent.learn()

#                 # if j % DDPG.log_interval == 0:
#                 #     agent.save()

#             # agent.save(speed)  # 保存模型

#             t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
#             path_a = "./csv/DDQN/speed_{}/{}".format(speed, t)
#             if not os.path.exists(path_a):
#                 os.makedirs(path_a)

#             plot.save_to_csv(delay_per_episode_list, "episode", "delay", path_a + "/DDQN_delay.csv")
#             plot.save_to_csv(cost_per_episode_list, "episode", "cost", path_a + "/DDQN_cost.csv")
#             plot.save_to_csv(reward_per_episode_list, "episode", "reward", path_a + "/DDQN_reward.csv")
#             plot.save_to_csv(reward_list, "episode", "reward", "./csv/DDQN/speed_{}-{}.csv".format(speed, t))

#             # print(np.mean(delay_per_episode_list))

#     else:
#         raise NameError("mode wrong!!!")


def dueling_dqn_main(env, mode):

    agent = DDQN.Agent(env.M * 3, env.M * 3, 1.0)  # -------3M

    if mode == 'test':
        for speed in env.speed:
            agent.load(speed)

            t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            path = "./log/DDPG/test/speed_{}-{}.md".format(speed, t)

            reward_list = []

            delay_per_vehicle_list = []
            cost_per_vehicle_list = []
            reward_per_vehicle_list = []

            total_alive_number = 0
            for j in range(max_episode):
                total_delay = 0
                total_cost = 0
                total_reward = 0

                # Initialize environment
                env.v_num = math.ceil((1 - speed / env.speed_max) * env.n)
                env.load = ger_workload(env.v_num)  # M
                env.state = np.hstack((env.load, env.g_r2b, [0] * env.M))  # -------3M
                cost_list = []
                for t in range(env.T):
                    env.action = agent.select_action(env.state)
                    env.action = (env.action + np.random.normal(0, DDPG.exploration_noise, size=env.M * 3)).clip(0, 1)  # 3M
                    action_normalize()

                    env.delay, env.cost = get_delay_and_cost(path)  # M M
                    env.reward = get_reward(env.delay, env.cost)  # M
                    print(env.reward)

                    total_delay += sum(env.delay)  # 1
                    total_cost += sum(env.cost)  # 1
                    total_reward += sum(env.reward)  # 1
                    total_alive_number += sum(env.alive_number)  # 1

                    env.load = ger_workload(env.v_num)  # M
                    env.next_state = np.hstack((env.load, env.g_r2b, env.delay))  # -------3M
                    env.state = env.next_state

                # plot.save_to_csv(cost_list, "time slot", "cost", "result/cost_time slot/" + "/TD3.csv")

                delay_per_vehicle_list.append(total_delay / total_alive_number / env.T)
                cost_per_vehicle_list.append(total_cost / total_alive_number / env.T)
                reward_per_vehicle_list.append(total_reward / env.T)
                #
            # t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            # path_a = "./csv/DDPG/test/speed_{}/{}".format(speed, t)
            # if not os.path.exists(path_a):
            #     os.makedirs(path_a)

            # plot.save_to_csv(delay_per_vehicle_list, "episode", "delay", path_a + "/DDPG_delay.csv")
            # plot.save_to_csv(cost_per_vehicle_list, "episode", "cost", path_a + "/DDPG_cost.csv")
            # plot.save_to_csv(reward_per_vehicle_list, "episode", "reward", path_a + "/DDPG_reward.csv")
            # plot.save_to_csv(reward_list, "episode", "reward", "./csv/DDPG/speed_{}-{}.csv".format(speed, t))
            # print("speed = {}".format(speed))
            # print("number = {}".format(env.v_num))
            # print("alive = {}".format(env.alive_number))
            # print(len(delay_per_vehicle_list))
            print(np.mean(delay_per_vehicle_list))
            # print(np.mean(cost_per_vehicle_list))
            # print(np.mean(reward_per_vehicle_list))

    elif mode == 'train':
        for speed in env.speed:
            t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            path = "./log/Dueling_DQN/speed_{}-{}.md".format(speed, t)

            total_step = 0
            reward_list = []

            log("speed", speed, path)

            delay_per_episode_list = []
            cost_per_episode_list = []
            reward_per_episode_list = []

            for j in range(max_episode):
                step = 0
                total_delay = 0
                total_cost = 0
                total_reward = 0

                # Initialize environment
                env.v_num = math.ceil((1 - speed / env.speed_max) * env.n)
                env.load = ger_workload(env.v_num)  # M
                env.state = np.hstack((env.load, env.g_r2b, [0] * env.M))  # -------3M
                record_per_episode(path)

                for t in range(env.T):
                    log("--------------------------------- t -------------------------------", "", path)
                    env.action = agent.choose_action(env.state)
                    env.action = (env.action + np.random.normal(0, DDPG.exploration_noise, size=env.M * 3)).clip(0, 1)  # 3M
                    action_normalize()

                    env.delay, env.cost = get_delay_and_cost(path)  # M M
                    env.reward = get_reward(env.delay, env.cost)  # M
                    # env.reward = sum(env.delay) / env.M  # 1

                    total_delay += sum(env.delay)  # 1
                    total_cost += sum(env.cost)  # 1
                    total_reward += sum(env.reward)  # 1

                    env.load = ger_workload(env.v_num)  # M
                    env.next_state = np.hstack((env.load, env.g_r2b, env.delay))  # -------3M
                    record_per_t(path)

                    agent.replay_buffer.push(env.state, env.action, env.next_state, sum(env.reward))
                    env.state = env.next_state
                    step += 1

                delay_per_episode_list.append(total_delay / env.M / env.T)
                cost_per_episode_list.append(total_cost / env.M / env.T)
                reward_per_episode_list.append(total_reward / env.M / env.T)

                total_step += step + 1

                average_t_reward = total_reward / env.T / env.M
                reward_list.append(average_t_reward)

                print("Episode: {}/{}\tAverage reward: {:0.5f}".format(j, max_episode, average_t_reward))

                agent.update(64)

                # if j % DDPG.log_interval == 0:
                #     agent.save()

            # agent.save(speed)  # 保存模型

            t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            path_a = "./csv/Dueling_DQN/speed_{}/{}".format(speed, t)
            if not os.path.exists(path_a):
                os.makedirs(path_a)

            plot.save_to_csv(delay_per_episode_list, "episode", "delay", path_a + "/Dueling_DQN_delay.csv")
            plot.save_to_csv(cost_per_episode_list, "episode", "cost", path_a + "/Dueling_DQN_cost.csv")
            plot.save_to_csv(reward_per_episode_list, "episode", "reward", path_a + "/Dueling_DQN_reward.csv")
            plot.save_to_csv(reward_list, "episode", "reward", "./csv/Dueling_DQN/speed_{}-{}.csv".format(speed, t))

            # print(np.mean(delay_per_episode_list))

    else:
        raise NameError("mode wrong!!!")


def ddpg_main(env, mode):

    agent = DDPG.Agent(env.M * 3, env.M * 3, 1.0)  # -------3M

    if mode == 'test':
        for speed in env.speed:
            agent.load(speed)
            t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            path = "./log/DDPG/test/speed_{}-{}.md".format(speed, t)

            reward_list = []

            delay_per_vehicle_list = []
            cost_per_vehicle_list = []
            reward_per_vehicle_list = []

            total_alive_number = 0
            for j in range(max_episode):
                total_delay = 0
                total_cost = 0
                total_reward = 0

                # Initialize environment
                env.v_num = math.ceil((1 - speed / env.speed_max) * env.n)
                env.load = ger_workload(env.v_num)  # M
                env.state = np.hstack((env.load, env.g_r2b, [0] * env.M))  # -------3M
                cost_list = []
                for t in range(env.T):
                    env.action = agent.select_action(env.state)
                    env.action = (env.action + np.random.normal(0, DDPG.exploration_noise, size=env.M * 3)).clip(0, 1)  # 3M
                    action_normalize()

                    env.delay, env.cost = get_delay_and_cost(path)  # M M
                    env.reward = get_reward(env.delay, env.cost)  # M
                    print(env.reward)

                    total_delay += sum(env.delay)  # 1
                    total_cost += sum(env.cost)  # 1
                    total_reward += sum(env.reward)  # 1
                    total_alive_number += sum(env.alive_number)  # 1

                    env.load = ger_workload(env.v_num)  # M
                    env.next_state = np.hstack((env.load, env.g_r2b, env.delay))  # -------3M
                    env.state = env.next_state

                plot.save_to_csv(cost_list, "time slot", "cost", "result/cost_time slot/" + "/TD3.csv")

                delay_per_vehicle_list.append(total_delay / total_alive_number / env.T)
                cost_per_vehicle_list.append(total_cost / total_alive_number / env.T)
                reward_per_vehicle_list.append(total_reward / env.T)
                #
            # t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            # path_a = "./csv/DDPG/test/speed_{}/{}".format(speed, t)
            # if not os.path.exists(path_a):
            #     os.makedirs(path_a)

            # plot.save_to_csv(delay_per_vehicle_list, "episode", "delay", path_a + "/DDPG_delay.csv")
            # plot.save_to_csv(cost_per_vehicle_list, "episode", "cost", path_a + "/DDPG_cost.csv")
            # plot.save_to_csv(reward_per_vehicle_list, "episode", "reward", path_a + "/DDPG_reward.csv")
            # plot.save_to_csv(reward_list, "episode", "reward", "./csv/DDPG/speed_{}-{}.csv".format(speed, t))
            # print("speed = {}".format(speed))
            # print("number = {}".format(env.v_num))
            # print("alive = {}".format(env.alive_number))
            # print(len(delay_per_vehicle_list))
            print(np.mean(delay_per_vehicle_list))
            # print(np.mean(cost_per_vehicle_list))
            # print(np.mean(reward_per_vehicle_list))

    elif mode == 'train':
        for speed in env.speed:
            t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            path = "./log/DDPG/speed_{}-{}.md".format(speed, t)

            total_step = 0
            reward_list = []

            log("speed", speed, path)

            delay_per_episode_list = []
            cost_per_episode_list = []
            reward_per_episode_list = []

            for j in range(max_episode):
                step = 0
                total_delay = 0
                total_cost = 0
                total_reward = 0

                # Initialize environment
                env.v_num = math.ceil((1 - speed / env.speed_max) * env.n)
                env.load = ger_workload(env.v_num)  # M
                env.state = np.hstack((env.load, env.g_r2b, [0] * env.M))  # -------3M
                record_per_episode(path)

                for t in range(env.T):
                    log("--------------------------------- t -------------------------------", "", path)
                    env.action = agent.select_action(env.state)
                    env.action = (env.action + np.random.normal(0, DDPG.exploration_noise, size=env.M * 3)).clip(0, 1)  # 3M
                    action_normalize()

                    env.delay, env.cost = get_delay_and_cost(path)  # M M
                    env.reward = get_reward(env.delay, env.cost)  # M
                    # env.reward = sum(env.delay) / env.M  # 1

                    total_delay += sum(env.delay)  # 1
                    total_cost += sum(env.cost)  # 1
                    total_reward += sum(env.reward)  # 1

                    env.load = ger_workload(env.v_num)  # M
                    env.next_state = np.hstack((env.load, env.g_r2b, env.delay))  # -------3M
                    record_per_t(path)

                    agent.memory.push((env.state, env.action, env.next_state, sum(env.reward)))
                    env.state = env.next_state
                    step += 1

                delay_per_episode_list.append(total_delay / env.M / env.T)
                cost_per_episode_list.append(total_cost / env.M / env.T)
                reward_per_episode_list.append(total_reward / env.M / env.T)

                total_step += step + 1

                average_t_reward = total_reward / env.T / env.M
                reward_list.append(average_t_reward)

                print("Episode: {}/{}\tAverage reward: {:0.5f}".format(j, max_episode, average_t_reward))

                agent.update()

                # if j % DDPG.log_interval == 0:
                #     agent.save()

            # agent.save(speed)  # 保存模型

            t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            path_a = "./csv/DDPG/speed_{}/{}".format(speed, t)
            if not os.path.exists(path_a):
                os.makedirs(path_a)

            plot.save_to_csv(delay_per_episode_list, "episode", "delay", path_a + "/DDPG_delay.csv")
            plot.save_to_csv(cost_per_episode_list, "episode", "cost", path_a + "/DDPG_cost.csv")
            plot.save_to_csv(reward_per_episode_list, "episode", "reward", path_a + "/DDPG_reward.csv")
            plot.save_to_csv(reward_list, "episode", "reward", "./csv/DDPG/speed_{}-{}.csv".format(speed, t))

            # print(np.mean(delay_per_episode_list))

    else:
        raise NameError("mode wrong!!!")


# def ppo_main(env):
#     global reward_list
#     # clear_log()
#     agent = PPO.Agent()

#     for speed in env.speed:
#         # if DDPG.load:
#         #     agent.load()
#         total_step = 0
#         reward_list = []
#         t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
#         path = "./log/PPO/speed_{}-{}.md".format(speed, t)

#         log("speed", speed, path)
#         for j in range(max_episode):

#             # begin of each episode
#             log("------------------------------------------------------  episode", j, path)
#             total_reward = 0
#             total_delay = 0
#             step = 0

#             # initialize first state
#             env.v_num = math.ceil((1 - speed / env.speed_max) * env.n)
#             env.load = ger_workload(env.v_num)  # M
#             env.state = np.hstack((env.load, env.g_r2b, [0] * env.M))  # 3M
#             record_per_episode(path)

#             for t in range(env.T):
#                 log("--------------------------------- t -------------------------------", "", path)

#                 # get PPO action
#                 env.action = []
#                 env.action_log_prob = []
#                 env.mu = []
#                 env.sigma = []

#                 for i in range(3 * env.M):
#                     a, a_prob, m, s = agent.select_action(env.state)
#                     env.action.append(a)
#                     env.action_log_prob.append(a_prob)
#                     env.mu.append(m)
#                     env.sigma.append(s)

#                 get_ppo_action()

#                 # ------------------------------- DRL奖励值 -------------------------------
#                 env.delay, env.cost = get_delay_and_cost(path)  # M M
#                 env.reward = get_reward(env.delay, env.cost)  # M RSU加惩罚的平均成本
#                 # env.reward = sum(env.delay) / env.M  # 1
#                 log("reward", env.reward, "./log/PPO/reward.md")

#                 env.load = ger_workload(env.v_num)  # M
#                 env.next_state = np.hstack((env.load, env.g_r2b, env.delay))  # 3M

#                 record_per_t(path)
#                 # print(len(env.action))

#                 if agent.store(PPO.Transition(env.state, env.action, sum(env.action_log_prob), sum(env.reward),
#                 env.next_state)):
#                     agent.update()
#                     print("PPO update")

#                 total_reward += sum(env.reward)
#                 # total_delay += sum(env.delay)

#                 env.state = env.next_state
#                 step += 1

#             total_step += step + 1

#             # average_t_delay = total_delay / env.T / env.M
#             average_t_reward = total_reward / env.T / env.M
#             reward_list.append(average_t_reward)
#             print("Episode: {}/{} \taverage reward: {:0.5f}".format(j, max_episode, average_t_reward))

#         t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
#         plot.save_to_csv(reward_list, "episode", "reward", "./csv/PPO/speed_{}-{}.csv".format(speed, t))


def td3_main(env, mode='train', collect_data=False):
    state_dim = env.M * 3  # 应该是12 (4 * 3)
    action_dim = env.M * 3  # 应该是12 (4 * 3)
    agent = TD3.Agent(state_dim=state_dim, action_dim=action_dim, max_action=1.0)
    
    if collect_data:
        costs = []
        agent.actor.eval()
        
        try:
            cost_per_episode_list = []
            env.v_num = env.n

            for j in range(max_episode):
                total_cost = 0
                # 确保每个部分都是正确的维度
                env.load = ger_workload(env.v_num)  # 应该是M维
                if len(env.load) != env.M:
                    env.load = np.zeros(env.M)  # 如果维度不对，初始化为0
                
                if len(env.g_r2b) != env.M:
                    env.g_r2b = np.zeros(env.M)  # 确保g_r2b是M维
                
                # 构建完整的12维状态向量
                state = np.zeros(env.M * 3)  # 初始化12维向量
                state[:env.M] = env.load  # 前M维是负载
                state[env.M:2*env.M] = env.g_r2b  # 中间M维是信道增益
                state[2*env.M:] = np.zeros(env.M)  # 最后M维是初始延迟
                
                # 打印调试信息
                print(f"State shape: {state.shape}")
                
                for t in range(env.T):
                    # 确保状态维度正确
                    current_state = state.reshape(1, -1)  # 转换为1x12
                    
                    # 选择动作
                    env.action = agent.select_action(current_state)
                    env.action = (env.action + np.random.normal(0, TD3.exploration_noise, size=env.M * 3)).clip(0, 1)
                    action_normalize()

                    # 获取成本
                    _, cost = get_delay_and_cost("", record=False)
                    total_cost += sum(cost)

                    # 更新状态
                    env.load = ger_workload(env.v_num)
                    if len(env.load) != env.M:
                        env.load = np.zeros(env.M)
                    
                    # 更新状态向量
                    state = np.zeros(env.M * 3)
                    state[:env.M] = env.load
                    state[env.M:2*env.M] = env.g_r2b
                    state[2*env.M:] = env.delay if hasattr(env, 'delay') else np.zeros(env.M)

                cost_per_episode_list.append(total_cost / env.M / env.T)
            
            costs.append(np.mean(cost_per_episode_list))
            return costs
        except Exception as e:
            print(f"Error in td3_main: {e}")
            print(f"Current state shape: {state.shape if 'state' in locals() else 'unknown'}")
            return [0] * len(env.speed)
    
    agent.actor.train()
    agent.actor_target.train()
    agent.critic_1.train()
    agent.critic_2.train()
    agent.critic_1_target.train()
    agent.critic_2_target.train()
    w_glob=[agent.actor.state_dict(), agent.actor_target.state_dict(),  agent.critic_1.state_dict(),
            agent.critic_2.state_dict(), agent.critic_1_target.state_dict(), agent.critic_2_target.state_dict()]
    print(w_glob)

    # for speed in env.speed:
    #     t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    #     path = "./log/TD3/speed_{}-{}.md".format(speed, t)
    #
    #     total_step = 0
    #     reward_list = []
    #
    #     delay_per_episode_list = []
    #     cost_per_episode_list = []
    #     reward_per_episode_list = []
    #
    #
    #     # for j in range(5):
    #     for j in range(max_episode):
    #
    #         total_delay = 0
    #         total_cost = 0
    #         total_reward = 0
    #         step = 0
    #
    #         # Initialize environment
    #         env.v_num = math.ceil((1 - speed / env.speed_max) * env.n)
    #         env.load = ger_workload(env.v_num)  # M
    #         env.state = np.hstack((env.load, env.g_r2b, [0] * env.M))  # -------3M
    #         # env.state = np.hstack((env.load, env.g_r2b))  # -------2M
    #         record_per_episode(path)
    #
    #         for t in range(T):
    #             log("--------------------------------- t -------------------------------", "", path)
    #             env.action = agent.select_action(env.state)
    #             env.action = (env.action + np.random.normal(0, TD3.exploration_noise, size=env.M * 3)).clip(0, 1)  # 3M
    #             action_normalize()
    #             env.delay, env.cost = get_delay_and_cost(path)  # M M
    #             env.reward = get_reward(env.delay, env.cost)  # M
    #             # env.reward = sum(env.delay) / env.M  # 1
    #
    #             env.load = ger_workload(env.v_num)  # M
    #             env.next_state = np.hstack((env.load, env.g_r2b, env.delay))  # -------3M
    #             # env.next_state = np.hstack((env.load, env.g_r2b))  # -------2M
    #             record_per_t(path)
    #
    #             agent.memory.push((env.state, env.next_state, env.action, sum(env.reward), [0] * env.M))
    #             if len(agent.memory.storage) >= TD3.capacity - 1:
    #                 agent.update(3)
    #                 # print("TD3 update")
    #
    #             env.state = env.next_state
    #             step += 1
    #
    #             total_delay += sum(env.delay)  # 1
    #             total_cost += sum(env.cost)  # 1
    #             total_reward += sum(env.reward)  # 1
    #
    #         delay_per_episode_list.append(total_delay / env.M / env.T)
    #         cost_per_episode_list.append(total_cost / env.M / env.T)
    #         reward_per_episode_list.append(total_reward / env.M / env.T)
    #
    #         total_step += step + 1
    #         average_t_reward = total_reward / env.T / env.M
    #         reward_list.append(average_t_reward)
    #
    #         print("Episode: {}/{} \tAverage reward: {:0.5f}".format(j, max_episode, average_t_reward))
    #
    #
    #     t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    #     path_a = "./csv/TD3/speed_{}/{}".format(speed, t)
    #     if not os.path.exists(path_a):
    #         os.makedirs(path_a)
    #
    #     plot.save_to_csv(delay_per_episode_list, "episode", "delay", path_a + "/TD3_delay.csv")
    #     plot.save_to_csv(cost_per_episode_list, "episode", "cost", path_a + "/TD3_cost.csv")
    #     plot.save_to_csv(reward_per_episode_list, "episode", "reward", path_a + "/TD3_reward.csv")
    #
    #     t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    #     plot.save_to_csv(reward_list, "episode", "reward", "./csv/TD3/speed_{}-{}.csv".format(speed, t))


def all_rsu_main(env, collect_data=False):
    costs = []
    for speed in env.speed:
        cost_per_episode_list = []
        env.v_num = env.n  # 直接使用设定的车辆数
        
        for j in range(max_episode):
            total_cost = 0
            for t in range(env.T):
                env.load = ger_workload(env.v_num)
                env.load = [env.load[i] * 1024 * 8 for i in range(env.M)]
                cost = [env.load[i] * env.price_r for i in range(env.M)]
                total_cost += sum(cost)
            
            cost_per_episode_list.append(total_cost / env.M / env.T)
        
        costs.append(np.mean(cost_per_episode_list))
    
    if collect_data:
        return costs
    print(np.mean(costs))


def all_cloud_main(env, collect_data=False):
    costs = []
    for speed in env.speed:
        cost_per_episode_list = []
        env.v_num = env.n  # 直接使用设定的车辆数

        for j in range(max_episode):
            total_cost = 0
            for t in range(env.T):
                env.load = ger_workload(env.v_num)  # M
                env.load = [env.load[i] * 1024 * 8 for i in range(env.M)]
                
                cost = [env.load[i] * (env.price_r + env.price_B) for i in range(env.M)]  # M
                total_cost += sum(cost)
            
            cost_per_episode_list.append(total_cost / env.M / env.T)
        
        costs.append(np.mean(cost_per_episode_list))
    
    if collect_data:
        return costs
    print(np.mean(costs))


def all_local_main(env):

    for speed in env.speed:

        delay_per_episode_list = []
        alive_per_episode_list = []
        cost_per_episode_list = []
        reward_per_episode_list = []

        env.v_num = math.ceil((1 - speed / env.speed_max) * env.n)

        for j in range(max_episode):
            total_delay = 0
            total_alive = 0
            total_cost = 0
            total_reward = 0
            for t in range(env.T):

                # Initialize environment
                env.load = ger_workload(env.v_num)  # M
                env.load = [env.load[i] * 1024 * 8 for i in range(env.M)]

                delay = [env.load[i] * env.c / env.f_v for i in range(env.M)]  # M
                # cost = [0] * env.M  # M
                # reward = get_reward(delay, cost)  # M

                total_delay += sum(delay)  # 1
                total_alive += sum(env.alive_number) # 1
                # total_cost += sum(cost)  # 1
                # total_reward += sum(reward)  # 1

            delay_per_episode_list.append(total_delay / env.M / env.T)
            alive_per_episode_list.append(total_alive / env.M / env.T)
            # cost_per_episode_list.append(total_cost / env.M / env.T)
            # reward_per_episode_list.append(total_reward / env.M / env.T)

        # t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        # path = "./csv/Local/speed_{}/{}".format(speed, t)
        # if not os.path.exists(path):
        #     os.makedirs(path)

        # plot.save_to_csv(delay_per_episode_list, "episode", "delay", path+"/local_delay.csv")
        # plot.save_to_csv(cost_per_episode_list, "episode", "cost", path+"/local_cost.csv")
        # plot.save_to_csv(reward_per_episode_list, "episode", "reward", path+"/local_reward.csv")
        print(np.mean(delay_per_episode_list) / np.mean(alive_per_episode_list))#np.mean平均值


def random_main(env, collect_data=False):
    costs = []
    for speed in env.speed:
        cost_per_episode_list = []
        env.v_num = env.n

        for j in range(max_episode):
            total_cost = 0
            env.load = ger_workload(env.v_num)

            for t in range(env.T):
                env.action = np.random.uniform(0, 1, size=12)
                env.action = (env.action + np.random.normal(0, DDPG.exploration_noise, size=env.M * 3)).clip(0, 1)
                action_normalize()

                _, cost = get_delay_and_cost("", record=False)  # 不记录日志
                total_cost += sum(cost)

            cost_per_episode_list.append(total_cost / env.M / env.T)
        
        costs.append(np.mean(cost_per_episode_list))
    
    if collect_data:
        return costs
    print(np.mean(costs))


def collect_comparison_data(env):
    results = {
        'Edge': [],
        'Cloud': [],
        'Random': [],
        'CORA': []
    }
    
    test_points = [10, 15, 20, 25, 30, 35, 40]
    
    for num_vehicles in test_points:
        print(f"Testing with {num_vehicles} vehicles...")
        env.n = int(num_vehicles)
        
        try:
            # 收集边缘计算结果
            edge_costs = all_rsu_main(env, collect_data=True)
            results['Edge'].append(float(np.mean(edge_costs)) if edge_costs else 0)
            
            # 收集云计算结果
            cloud_costs = all_cloud_main(env, collect_data=True)
            results['Cloud'].append(float(np.mean(cloud_costs)) if cloud_costs else 0)
            
            # 收集随机调度结果
            random_costs = random_main(env, collect_data=True)
            results['Random'].append(float(np.mean(random_costs)) if random_costs else 0)
            
            # 收集CORA结果
            cora_costs = td3_main(env, mode='test', collect_data=True)
            results['CORA'].append(float(np.mean(cora_costs)) if cora_costs else 0)
            
        except Exception as e:
            print(f"Error at {num_vehicles} vehicles: {e}")
            # 如果出错，为所有策略添加默认值
            results['Edge'].append(0)
            results['Cloud'].append(0)
            results['Random'].append(0)
            results['CORA'].append(0)
    
    return results, test_points


def plot_comparison(results, vehicle_numbers):
    import matplotlib.pyplot as plt
    
    # 验证数据
    if not all(len(results[key]) == len(vehicle_numbers) for key in results):
        raise ValueError("Data length mismatch with vehicle numbers")
    
    width = 0.2
    plt.figure(figsize=(10, 6))
    
    # 设置柱状图位置
    x = np.arange(len(vehicle_numbers))
    
    # 确保数据是浮点数并处理无效值
    edge_data = [float(v) if v is not None else 0 for v in results['Edge']]
    cloud_data = [float(v) if v is not None else 0 for v in results['Cloud']]
    random_data = [float(v) if v is not None else 0 for v in results['Random']]
    cora_data = [float(v) if v is not None else 0 for v in results['CORA']]
    
    # 绘制柱状图
    plt.bar(x - 1.5*width, edge_data, width, label='Edge computing', color='coral')
    plt.bar(x - 0.5*width, cloud_data, width, label='Cloud computing', color='lightgreen')
    plt.bar(x + 0.5*width, random_data, width, label='Random scheduling', color='skyblue')
    plt.bar(x + 1.5*width, cora_data, width, label='CORA', color='plum')
    
    # 设置图表属性
    plt.xlabel('Number of vehicles')
    plt.ylabel('System cost')
    plt.title('System cost with different number of vehicles')
    plt.xticks(x, vehicle_numbers)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 保存图表
    plt.savefig('system_cost_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    env = env()
    
    try:
        print("Collecting data...")
        print(f"Environment dimensions: M={env.M}")
        results, vehicle_numbers = collect_comparison_data(env)
        
        print("Plotting comparison...")
        plot_comparison(results, vehicle_numbers)
        print("Done! Results saved to system_cost_comparison.png")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
