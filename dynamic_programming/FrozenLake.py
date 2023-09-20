import gym

from dynamic_programming import PolicyIteration, ValueIteration
from PolicyIteration import PolicyIteration
from ValueIteration import ValueIteration
from dynamic_programming.main import print_agent


#env = gym.make("FrozenLake-v1", render_mode="rgb_array")  # 创建环境
env = gym.make("FrozenLake-v1", render_mode="human")  # 创建环境
env.reset()
env = env.unwrapped  # 解封装才能访问状态转移矩阵P
env.render()  # 环境渲染,通常是弹窗显示或打印出可视化的环境

holes = set()
ends = set()
for s in env.P:
    for a in env.P[s]:
        for s_ in env.P[s][a]:
            if s_[2] == 1.0:  # 获得奖励为1,代表是目标
                ends.add(s_[1])
            if s_[3] == True:
                holes.add(s_[1])
holes = holes - ends
print("冰洞的索引:", holes)
print("目标的索引:", ends)

for a in env.P[14]:  # 查看目标左边一格的状态转移信息
    print(env.P[14][a])


print("策略迭代方法")
action_meaning = ['<', 'v', '>', '^']
theta = 1e-5
gamma = 0.9
agent = PolicyIteration(env, theta, gamma)
agent.policy_iteration()
print_agent(agent, action_meaning, holes, ends)


print("价值迭代方法")
agent = ValueIteration(env, theta, gamma)
agent.value_iteration()
print_agent(agent, action_meaning, holes, ends)