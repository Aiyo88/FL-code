# 建立模型，创建自己的环境体，pdqn的agent应该是不用改的，自己去做适配
# pdqn结果输出来之后，需要再到d3qn进行跑
# 有没有可能保存pdqn的结果，然后在训练d3qn，将他们分开跑呢 - - 这样比一次性的跑是不是快些
'''
env : 确定状态空间、动作空间、奖励机制等行为
1. init初始化环境体的参数:状态、动作、奖励等
2. step
3. reset
'''
"""
状态(State):
状态应该包含边缘服务器的存储空间和计算资源的当前状态，以及可能的服务质量(QoS)指标,如延迟或任务队列长度。
您可能需要一个状态向量来表示每个边缘服务器的状态，包括它的最大存储和计算能力，以及当前正在处理的任务类型和数量。

动作(Action):
动作可以是一个多部分的结构，一部分决定任务是否卸载到边缘服务器(离散动作），另一部分决定资源分配的比例(连续动作参数）。
您可以使用PDQN的ParamActor来输出这些连续动作参数,例如分配给每个服务的计算资源比例。

奖励(Reward):
奖励函数应该反映出任务响应时间的最小化目标。例如，任务成功执行和快速响应时间可以给予正奖励，而任务失败、执行超时或资源浪费可以给予负奖励。

环境模型(Environment Model):
环境模型需要模拟任务生成、服务部署和任务卸载的过程。这包括根据当前状态和动作来更新环境的状态，并计算下一状态和奖励。
您需要模拟任务到达的过程，包括它们的类型和服务需求，以及它们是如何被分配到各个边缘服务器上的
"""

import numpy as np
from pprint import pprint
import gym
from gym.spaces import Discrete
from gym.spaces import Box

class OffLoadingEnv():
    # 因为统计的是每个时隙t的所以，所以在命名变量的时候可以将上标t去掉
    def __init__(self,numbers_edge_server=10, numbers_service=5, task_d_numbers=3,lambda_s=0.04,theta_s=0.1):
        super(OffLoadingEnv, self).__init__()
        #初始化一些参数之类的
        # 这里的状态空间得是1维的才行
        low = [0.0] * (numbers_edge_server * task_d_numbers) # 代表状态空间中每个维度的最低值
        high = [1.0] * (numbers_edge_server * task_d_numbers)
        self.observation_space  = Box(low=np.array(low), high=np.array(high), dtype=np.float32)
        # self.action_space = np.zeros((numbers_edge_server,task_d_numbers))
        self.action_space = Discrete(numbers_edge_server * task_d_numbers) # 动作空间
        # self.observation_space  = np.zeros(numbers_edge_server * task_d_numbers)
        # self.action_space = np.zeros((numbers_edge_server,task_d_numbers))

        '''
            一般变量
        '''

        self.service_i_numbers = numbers_service # 服务类型的个数
        self.edgeserver_numbers = numbers_edge_server # 边缘服务器的个数 
        self.task_d_numbers = task_d_numbers # 任务的数量 d的个数
        self.lambda_s = lambda_s # 任务的数量 d的个数
        self.theta_s = theta_s # 任务的数量 d的个数

        self.cc_pro = 15 # 单位处理成本
        # self.cc_pro = 0.3 # 单位处理成本
        
        '''
            状态相关的是啥
        '''
        # ？？？
        self.f_dj_t = [] # 服务器j给任务d分配的计算资源


        '''
            计算资源相关变量
        '''
        # 这个是否要从pdqn中继承过来呢

        self.f_j_max = [] # 服务器j的最大计算资源


        '''
            任务相关的
        '''
        self.c_d = [] # 任务的大小
        self.cpu_task_d = [] # 任务所需要的计算资源


        '''
            action space
        '''
        self.a_ij_t = [] # 服务器j上是否卸载服务类型i
        self.h_ij_t = [] # 给服务器j给服务类型i分配多少计算资源

        self.b_dj_t = [] # 任务的卸载决策 是否在服务器J中放置任务d 、 0与1
        # r_di_t 可以不用在意 d任务就是i类型的
        # self.r_di_t = [] # 任务类型判断 判断任务d属于那个服务类型i 、 0与1


        '''
            一些基本条件的值
        '''
        self.cost_placement = 0 # 这算是每轮的成本花费啊？所有的epoch加起来才行

        '''
            故障率相关
        '''

        self.lambda_j = [] # 故障率
        self.theta_j = [] # 修复率



    

    '''
        判断动作空间是否满足约束规则、否则给予惩罚(reward)
        我只计算惩罚、 对于不符合约束的值我不要管、不要给予纠正
    '''
    def meets_action_constraints(self):
        #
        # pdqn中不满足的约束是否也要予以惩罚呢？不用吧，毕竟是继承过来的，继承过来的时候已经给予惩罚了
        #

        '''
            一个约束的惩罚只用加一次就够了，不用循环的加
        '''
        penalty = 0 # 惩罚

        # 但是这里的一个问题是 a h 这两个动作空间并不是符合之前的约束的啊，怎么半呢？
        # 这里最好直接 a h 对应0的地方直接给映射起来，自己先处理掉吧
        self.h_ij_t *= self.a_ij_t

        '''
            判断约束1 
        '''
        # 列优先遍历
        for d in range(self.task_d_numbers):
            sum_d = 0 
            for j in range(self.edgeserver_numbers):
                sum_d += self.b_dj_t[j,d]
            if sum_d > 1:
                penalty += 5
                break

        '''
            约束2
        '''
        constraints1 = False

        for j in range(self.edgeserver_numbers):
            for i in range(self.service_i_numbers):
                sum_d_2 = 0
                for d in range(self.task_d_numbers):
                    sum_d_2 += self.f_dj_t[j,d] * self.b_dj_t[j,d]
                # print("sum_d_2",sum_d_2)
                # print("self.h_ij_t[j,i] * self.f_j_max[j]:",self.h_ij_t[j,i] * self.f_j_max[j])
                if sum_d_2 > self.h_ij_t[j,i] * self.f_j_max[j]:
                    penalty += 10
                    constraints1 = True
                    break
            if constraints1:
                break


        return penalty
    


    '''
        reset初始化所有的参数,变量,状态等
        然后agent-pdqn执行act,agent根据当前的state选择出来action
        然后将action传入环境体,环境体根据当前的状态 以及 选择出来的action, 执行相应的逻辑,比较惩罚，奖励进行碰撞
        得到当前action+state下的下一个next_state
        如此反复以往,直到时间用尽或者reward趋于平稳done结束
    '''
    def step(self, action, pdqn_action):
        # 根据动作和环境当前的状态更新环境，计算奖励、返回当前动作到达的下一个状态以及当前动作对应的奖励
        """ 
        return :
            next_state
            reward
            done 
            info
        """

        # 动作就是self.action_a 与 self.action_h

    
        # 首先是判断当前的action是否符合约束等
        # 先讲当前的action进行赋值
        '''
            注意这里进行传入的参数action的替换
        '''
        # self.action_a_ij = np.eye(self.edgeserver_numbers , self.service_i_numbers)
        # self.action_h_ij = np.zeros((self.edgeserver_numbers,self.service_i_numbers))
        # 在函数中反解析

        def parse_pdqn_action(action):
            action1_size = self.edgeserver_numbers * self.service_i_numbers
            action1 = np.array(action[:action1_size])
            action2 = np.array(action[action1_size:])
            return action1.reshape((self.edgeserver_numbers,self.service_i_numbers)), action2.reshape((self.edgeserver_numbers,self.service_i_numbers))
        
        def parse_action(action):
            # d3qn的好像只有一个 就是 b r不用管不在意
            action1_size = self.edgeserver_numbers * self.task_d_numbers
            action1 = np.array(action[:action1_size])
            # action2 = action[action1_size:]
            return action1.reshape((self.edgeserver_numbers,self.task_d_numbers)) # , action2.reshape((self.edgeserver_numbers,self.task_d_numbers))
        
        '''
            需要将action 与 pdqn_action 的值进行数值区间的映射
        '''

        # pdqn_action是映射好的
        # action是d3qn输出的 
        self.a_ij_t, self.h_ij_t = parse_pdqn_action(pdqn_action)


        self.b_dj_t = parse_action(action)
        # print("b_dj_t",self.b_dj_t)
        # 搞d3qn的输出
        for j in range(self.edgeserver_numbers):
            for d in range(self.task_d_numbers):
                self.b_dj_t[j,d] = 1 if self.b_dj_t[j,d] > 0.5 else 0

        # print("b_dj_t",self.b_dj_t)

        # 判断是否满足约束条件
        action_penalty = self.meets_action_constraints()
        # print("penaly:",action_penalty)

        '''
            实现目标以及优化函数
            成本
        '''
        # 服务器故障的处理时间
        theta_dj_t =  np.zeros((self.edgeserver_numbers,self.task_d_numbers))


        '''
            无故障处理时间
        '''
        epus_dj_t = np.zeros((self.edgeserver_numbers,self.task_d_numbers))

        for j in range(self.edgeserver_numbers):
            for d in range(self.task_d_numbers):
                epus_dj_t[j,d] = self.c_d[d] / self.f_dj_t[j,d]



        for j in range(self.edgeserver_numbers):
            for d in range(self.task_d_numbers):
                temp1 = self.lambda_j[j] * self.theta_j[j] * epus_dj_t[j,d]
                temp2 = self.lambda_j[j] + self.theta_j[j]
                temp3 = np.exp( self.lambda_j[j] * epus_dj_t[j,d] ) + np.exp(-1 * self.lambda_j[j] * epus_dj_t[j,d]) - 2
                temp4 = self.lambda_j[j] * self.theta_j[j] * (2 - np.exp(-1 * self.lambda_j[j] * epus_dj_t[j,d])  )
                epsilon = 1e-10  # 小数值
                theta_dj_t[j,d] = (temp1 + temp2 * temp3) / (temp4 + epsilon)

        # 计算总的时间
        T_t = 0 
        for j in range(self.edgeserver_numbers):
            for i in range(self.service_i_numbers):
                for d in range(self.task_d_numbers):
                    T_t += self.a_ij_t[j,i] * self.b_dj_t[j,d] * theta_dj_t[j,d]

        '''
            这里的T_t怎么有的是负的呢，不太对啊 ，应该是正的才行，哪部分的计算有问题
        '''
        
        # print("T_t" , T_t)
        # print("action_penalty" , action_penalty)

        # 总的处理成本
        # 这里的太小了好像
        # 
        # 
        self.cost_placement = self.cc_pro * T_t
                
                

        reward = -(self.cost_placement + action_penalty)
        done = False

        # 在返回reward和cost之前
        if np.isnan(reward):
            reward = 0.0  # 或其他默认值
        if np.isnan(self.cost_placement):
            self.cost_placement = 0.0  # 或其他默认值



        '''
            状态之间的转移是啥呢？状态之间如何转移的呢？
            state = state+action
            最好是按照比例来设定
            带宽就不管他了，带宽就忽略掉了
            a_ij: [
                [1,0,0]
                [0,1,0]
                [0,0,1]
            ]
            h_ij: [
                [0.1, 0,   0]
                [0,   0.3, 0]
                [0,   0,   0.4]
            ]
        '''

        '''
        #
        # 这里有一个问题，计算资源一直是减小的吗？不能增加吗？减小到0咋办
        # 
        '''

        for j in range(self.edgeserver_numbers):
            for d in range(self.task_d_numbers):
                if self.b_dj_t[j,d] == 1:
                    self.f_dj_t[j,d] -= self.cpu_task_d[d]





        self.episode_rewards += reward

        
        # print("reward",reward)
        return np.array(self.f_dj_t.ravel()) , reward , self.cost_placement
    

    def get_episode_rewards(self):
        return self.episode_rewards
    

    def rewardFunction(self):
        # 定义任务目标定义奖励函数

        return self.reward
    
    '''
        状态是啥？
    '''
    def get_combined_state(self):
        return self.f_dj_t # j * d大小的

    def reset(self):

        '''
            action的初始化
        '''
        self.b_dj_t = np.zeros((self.edgeserver_numbers,self.task_d_numbers))


        '''
            基本变量一般变量的初始化
        '''
        self.cost_placement = 0 
        self.cc_tra = 0.03 # 单位传输成本

        '''
            奖励相关
        '''
        self.episode_rewards = 0

        '''
            存储资源计算资源做约束
        '''
        # 
        # 需要及时调整f-dj-t的大小，给予惩罚和奖励的时候才能合适
        #
        self.f_dj_t =np.random.uniform(5, 10, size=(self.edgeserver_numbers , self.task_d_numbers)) # 服务器j分配给任务d的计算资源
        self.f_j_max = np.random.uniform(15, 20, size=self.edgeserver_numbers) # 服务器j的最大计算资源

        '''
            任务相关
        '''
        self.c_d = np.random.uniform(0.5, 1, size=self.task_d_numbers)  # 任务大小相关

        #
        # 每个任务d所需要的计算资源，不然都没法完成状态之间的转移
        #
        self.cpu_task_d = np.random.uniform(0.5, 1.5, size=self.task_d_numbers)

        '''
            服务器故障时间
        '''
        self.lambda_j = [self.lambda_s] * self.edgeserver_numbers
        self.theta_j = [self.theta_s] * self.edgeserver_numbers
        # self.lambda_j = [0.01] * self.edgeserver_numbers
        # self.theta_j = [0.1] * self.edgeserver_numbers

        # 二维变成一维
        return np.array(self.f_dj_t.ravel())
    
    def get_observation(self):
        return {'action_a_ij': self.action_a_ij,
                 'action_h_ij': self.action_h_ij}
    

    def render(self):
        return None
    
    def close(self):
        return None
    
    def seed(self):
        return None


