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
from gym import spaces

class EdgeComputingEnv():
    # 因为统计的是每个时隙t的所以，所以在命名变量的时候可以将上标t去掉
    def __init__(self,numbers_edge_server=10, numbers_service=5):
        super(EdgeComputingEnv, self).__init__()
        # self.observation_space = spaces.Tuple([
        #     spaces.Box(low=0, high=1, shape=(numbers_edge_server * 2,), dtype=np.float32),
        # ])
        # ai修改过
        self.observation_space = spaces.Box(low=0, high=1, shape=(numbers_edge_server * 2,), dtype=np.float32)
        
        self.action_space = spaces.Tuple([
            spaces.MultiDiscrete([2] * numbers_edge_server * numbers_service), # 
            spaces.Box(low=0, high=1, shape=(numbers_edge_server, numbers_service), dtype=np.float32),  
        ])

        '''
            一般变量
        '''
        self.service_i_numbers = numbers_service # 服务类型的个数
        # self.f_dj = [] # action中 f_dj 任务d卸载到服务器j上所需要的计算资源
        # self.s_j_max = [] # 服务器j的最大存储空间
        # self.f_j_max = [] # 服务器j的最大计算空间(资源)


        self.c_i  = [] # 服务i所需要的放置成本
        self.c_i_max = [] # 服务i放置的最大成本
        self.service_s_i = [] # 服务i所需要的存储空间
        self.server_f_max = [] # 服务器j的最大计算资源

        '''
            时间、传输计算上的变量
        '''
        self.rate_j_t  = [] # 云服务器到边缘服务器的下行链路传输速率
        self.cc_tra = 0.02 # 单位传输成本
        self.transtime_beta_ij_t = [] # 传输时间 每个time下的每个ij

        '''
            state space
        '''
        self.edgeserver_numbers = numbers_edge_server # 边缘服务器的个数  # 这个好像没有作为状态空间传入，但是是空间的一个属性值，可以传入，但是不涉及到任何操作
        # self.state_B_j = None # j个[] 上传链路的带宽 update
        self.state_ss_j = None #  每个边缘服务器的存储当量,服务器j所剩余的存储空间
        self.state_ff_j = None #  每个边缘服务器的计算当量 - notes: 是想说计算资源总量吗

        '''
            action space
        '''
        # i、j、t、d
        # i是服务i
        # j是服务器j
        # d是任务d
        # t是每个时隙、实验可以忽略
        self.action_a_ij = None  # 服务资源放置的决策 a_ij 0/1
        self.action_h_ij = None # 计算资源的分配决策 h_ij [0-1]

        # if self.h = self.a | aij=0 => hij=0

        '''
            一些基本条件的值
        '''
        self.cost_placement = 0 # 这算是每轮的成本花费啊？所有的epoch加起来才行
        # self.service_B = [] # 每个服务所占用的带宽？
        self.episode_rewards = 0


        '''
            轮流放置的标志位
        '''
        self.index = 0
    
    
    '''
        判断状态空间是否满足、否则给予惩罚(reward)
    ''' 
    def meets_state_constraints(self, state):
        print("adjust state")

    '''
        判断动作空间是否满足约束规则、否则给予惩罚(reward)
        我只计算惩罚、 对于不符合约束的值我不要管、不要给予纠正
    '''
    def meets_action_constraints(self):
        
        '''
            一个约束的惩罚只用加一次就够了，不用循环的加
        '''
        penalty = 0 # 惩罚
        '''
            判断约束0 就是a=0 对应的h也应该为0
        '''
        constraints0 = False
        for j in range(self.edgeserver_numbers):
            for i in range(self.service_i_numbers):
                if self.action_a_ij[j,i] == 0 and self.action_h_ij[j,i] != 0:
                    penalty += 10 
                    constraints0 = True
                    break
            if constraints0:
                break

        '''
            判断约束1 放置决策
            判断约束2 放置成本决策
        '''
        constraints1 = False
        constraints2 = False

        cost_i = [1] * self.service_i_numbers
        # 判断服务放置约束是否满足条件
        for i in range(self.service_i_numbers):
            sum_j = 0
            service_place = []
            for j in range(self.edgeserver_numbers):
                sum_j+= self.action_a_ij[j,i]
                if self.action_a_ij[j,i] == 1 :
                    service_place.append(j) # 代表j服务器放置了
            if sum_j == 0 and constraints1 == False:
                # 如果某一列的service不满足约束1
                penalty += 15
                constraints1 = True


            # sum_j 其实代表了放置的个数
            cost_i[i] = sum_j * self.c_i[i]

            if cost_i[i] > self.c_i_max[i] and constraints2 == False:
                penalty += 20
                constraints2 = True
            
            if constraints1 and constraints2:
                break


        '''
            判断约束3 服务器剩余存储空间决策
        '''
        constraints3 = False
        for j in range(self.edgeserver_numbers):
            storge = 0
            for i in range(self.service_i_numbers):
                storge += self.action_a_ij[j,i] * self.service_s_i[i]
            if storge > self.state_ss_j[j] : 
                # 如果当前需要使用的空间大小 超出本身的存储空间了 给予惩罚
                penalty += 25
                constraints3 = True

            if constraints3:
                break

        '''
            判断约束4 服务器剩余计算资源决策
        '''
        constraints4 = False
        for j in range(self.edgeserver_numbers):
            computeresource = 0
            for i in range(self.service_i_numbers):
                computeresource += self.action_h_ij[j,i] * self.server_f_max[j]
            if computeresource > self.state_ff_j[j]:
                penalty += 15
                constraints4 = True
            if constraints4:
                break

    
        return penalty


    '''
        reset初始化所有的参数,变量,状态等
        然后agent-pdqn执行act,agent根据当前的state选择出来action
        然后将action传入环境体,环境体根据当前的状态 以及 选择出来的action, 执行相应的逻辑,比较惩罚，奖励进行碰撞
        得到当前action+state下的下一个next_state
        如此反复以往,直到时间用尽或者reward趋于平稳done结束
    '''
    def step(self, action):
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

        def parse_action(action):
            '''
                记得验证这样的拆分是否是正确的
            '''
            action1_size = self.edgeserver_numbers * self.service_i_numbers
            action1 = action[:action1_size]
            action2 = action[action1_size:]
            return action1.reshape((self.edgeserver_numbers,self.service_i_numbers)), action2.reshape((self.edgeserver_numbers,self.service_i_numbers))
        
        '''
            需要将action的值进行数值区间的映射
        '''
        action_a_ij_temp, self.action_h_ij = parse_action(action)

        for j in range(self.edgeserver_numbers):
            for i in range(self.service_i_numbers):
                self.action_a_ij[j,i] = 1 if action_a_ij_temp[j,i] > 0.5 else 0

        """
            自定义资源分配的决策
        """
        # 统计每个服务器上的任务的数量

        

        # 判断是否满足约束条件
        action_penalty = self.meets_action_constraints()
        # print("penaly:",action_penalty)

        '''
            实现目标以及优化函数
        '''

        ## 计算放置的成本

        # 传输成本
        # 服务器j上的服务i的传输时间 [j,i]
        for i in range(self.service_i_numbers):
            for j in range(self.edgeserver_numbers):
                self.transtime_beta_ij_t[j,i] = self.service_s_i[i] / self.rate_j_t[j]
        # print("transtime_beta_ij_t:",self.transtime_beta_ij_t)

        for i in range(self.service_i_numbers):
            for j in range(self.edgeserver_numbers):
                self.cost_placement += self.action_a_ij[j,i] * (self.c_i[i] + self.cc_tra * self.transtime_beta_ij_t[j,i]) 

        # 但是这个cost_placement是对于所有的时间而言的

        reward = -(self.cost_placement + action_penalty)
        done = False



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

        # 根据决策决定存储资源与计算资源的变化，变化就是新的状态，状态之间的转移

        temp_h_ij = self.action_h_ij
        temp_h_ij *= self.action_a_ij

        for j in range(self.edgeserver_numbers):
            ss_j = 0 # 部署对应的i的存储空间
            ff_j = 0 # 部署对应的i的计算资源消耗
            for i in range(self.service_i_numbers):
                if temp_h_ij[j,i] != 0:
                    # !=0 代表这个服务i要被卸载到服务器j中
                    # service i 占用的存储空间
                    ss_j += self.service_s_i[i]
                    ff_j += temp_h_ij[j,i]
            # 对于服务器j而言，要在j上放置的各种服务所消耗的存储以及计算资源都计算完成了
            if self.state_ss_j[j] > ss_j and self.state_ff_j[j] > ff_j:
                # 如果消耗的资源都小于剩余的资源，那就进行状态的转移？否则不进行状态的转移
                self.state_ss_j[j] = self.state_ss_j[j] - ss_j
                self.state_ff_j[j] = self.state_ff_j[j] - ff_j
            else :
                # 否则 不进行状态的转移？
                None
        

        self.episode_rewards += reward

        
        # print("reward",reward)
        return self.get_combined_state() , reward , self.cost_placement
    

    def get_episode_rewards(self):
        return self.episode_rewards
    

    def rewardFunction(self):
        # 定义任务目标定义奖励函数

        return self.reward
    
    def get_combined_state(self):
        # 将四种状态变量拼接成一个状态向量
        # 先讲带宽这个东西进行忽略吧
        combined_state = np.concatenate([self.state_ff_j, self.state_ss_j])
        # combined_state = np.concatenate([self.state_B_j, self.state_ff_j, self.state_ss_j])
        return combined_state

    def reset(self):
        # 重置环境状态
        '''
            # 对于初始状态就将该置为0的全都置为0
            初始化参数的范围要规定一下在哪个范围内行动
        '''
        # 存储空间
        self.state_ss_j = np.random.randint(20, 40, size=self.edgeserver_numbers)
        # 计算资源
        self.state_ff_j = np.random.randint(20, 60, size=self.edgeserver_numbers)

        '''
            action的初始化
        '''
        self.action_a_ij = np.eye(self.edgeserver_numbers , self.service_i_numbers)
        self.action_h_ij = np.zeros((self.edgeserver_numbers,self.service_i_numbers))

        '''
            和服务决策相关变量的初始化
            # 因为前边每列只有一个1,所以就可以保证这个放置成本
            c_i | c_i_max
        '''
        self.c_i = np.random.uniform(10, 20, size=self.service_i_numbers)
        self.c_i_max = self.c_i + np.random.uniform(10, 20, size=self.service_i_numbers)        
        
        '''
            每种服务的放置约束、以及服务类型的成本约束
        '''

        '''
            基本变量一般变量的初始化
        '''

        self.rate_j_t = np.random.uniform(10, 20, size=self.edgeserver_numbers)
        self.cost_placement = 0 
        self.cc_tra = 0.02 # 单位传输成本
        self.transtime_beta_ij_t = np.zeros((self.edgeserver_numbers,self.service_i_numbers)) # 传输时间 每个time下的每个ij
        '''
            奖励相关
        '''
        self.episode_rewards = 0
        

        '''
            存储资源计算资源做约束
        '''
        # 服务i所需要的存储空间, 服务类型i的大小
        ### 这里的服务类型的大小 是否具有一些隐含的条件呢- 比如说和服务器的存储空间相比的话，在初始化的时候就要小于这些值呢

        # 服务i所需要的存储空间
        self.service_s_i = np.random.uniform(1, 5, size=self.service_i_numbers)
        # 服务器j的最大计算资源
        self.server_f_max = np.random.uniform(1, 5, size=self.edgeserver_numbers)

        return self.get_combined_state()
    
    def get_observation(self):
        # 返回当前观测值，可以包括 self.action_a_ij 和 self.action_h_ij
        return {'action_a_ij': self.action_a_ij,
                 'action_h_ij': self.action_h_ij}
    

    def render(self):
        return None
    
    def close(self):
        return None
    
    def seed(self):
        return None


        
if __name__ == '__main__':
    numbers_edge_server = 6
    numbers_service = 3
    env = EdgeComputingEnv(numbers_edge_server=6, numbers_service=3)
    state = env.reset()
    print("state", state)

    print(env.get_observation())
    action1 = np.eye(numbers_edge_server, numbers_service)
    action2 = np.zeros((numbers_edge_server,numbers_service))
    action = np.concatenate([action1,action2])


    next_state, reward, done, _ = env.step(action)

    print("next_state:",next_state)
    print("reward",reward)


    print("observation_space",env.observation_space.shape)
    print("observation_space",env.observation_space.shape[0])
    # print("observation_space",env.observation_space.shape[1])
    print("action_sapce",env.action_space.shape)
