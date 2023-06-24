import math

import numpy as np
import random


# 定义MKP问题的参数和约束条件

num_particles = 120  # 粒子数量
max_iterations = 1200  # 最大迭代次数
# penalty_factor = 2000  # 罚函数的惩罚因子
c1 = 1.5 # 学习因子1
c2 = 1.5 # 学习因子2


# def generate_matrix(x, y):
#     matrix = np.zeros((x, y))  # 创建一个全零矩阵
#     matrix[:, :y//10] = 1      # 将前 y/3 列赋值为 1
#     return matrix

def generate_matrix(x, y, m):
    m = int(m)
    matrix = np.zeros((x, y))  # 创建一个全零矩阵
    columns = np.random.choice(y, m, replace=False)  # 随机选择 m 个不重复的列索引
    matrix[:, columns] = 1  # 将选择的列赋值为 1
    return matrix


# 定义MKP问题的目标函数和约束条件
def objective_function(position,max_weight, weights, values):
    # 计算目标函数值最大化价值总和
    ans = 0
    for i in range(len(position)):
        if position[i] == 1 or position[i] == 0:
            ans += values[i]*position[i]
    # if not constraint_function(position,max_weight,weights,values):
    #     ans = ans - penalty_factor
    return ans


def constraint_function(position, max_weight, weights, values):
    # 判断是否满足约束条件背包的承重不能超过最大承重
    weight = np.sum(position * weights)
    # print(weight)
    if weight > max_weight:
        return False
    return True


def mkp_bpso(weights, values, max_weight):
    con = []
    # 初始化粒子的位置和速度
    num_items = len(values)  # 物品数量
    # particles = np.random.rand(num_particles, num_items)
    # particles = np.random.randint(2, size=(num_particles, num_items))
    velocities = np.zeros((num_particles, num_items), dtype=float)
    # velocities = np.random.uniform(-1, 1, size=(num_particles, num_items))

    # 初始化粒子的最佳位置和全局最佳位置
    best_positions = np.zeros((num_particles, num_items), dtype=int)
    # global_best_position = particles[0].copy()
    # particles = np.random.randint(2, size=(num_particles, num_items))
    # particles = np.zeros((num_particles, num_items), dtype=int)
    particles = generate_matrix(num_particles,num_items,num_items/50)
    global_best_position = particles[0].copy()
    particles = np.random.randint(2, size=(num_particles, num_items))
    if not constraint_function(global_best_position,max_weight,weights,values):
        # global_best_fitness = objective_function(global_best_position, max_weight,weights, values)- penalty_factor
        global_best_fitness = 0
    else:
        global_best_fitness = objective_function(global_best_position, max_weight,weights, values)

    # 迭代优化过程
    for iteration in range(max_iterations):
        for i in range(num_particles):
            # 更新速度和位置
            velocities[i] = velocities[i] + c1*random.random() * (best_positions[i] - particles[i]) + c2*random.random() * (global_best_position - particles[i])
            for temp in range(len(velocities[i])):
                p = 1/(1+math.exp(-velocities[i][temp]))
                if random.random() <= p:
                    particles[i][temp] = 1
                else:
                    particles[i][temp] = 0
            # particles[i] = np.where(velocities[i] > 0.5, 1, 0)  # 根据速度更新位置
            # 根据速度更新位置
            # particles[i] = particles[i] + velocities[i]
            # 处理约束条件
            if not constraint_function(particles[i],max_weight,weights,values):
                # 引入罚函数
                # fitness = objective_function(particles[i], max_weight,weights, values) - penalty_factor
                fitness = 0
            else:
                fitness = objective_function(particles[i], max_weight,weights, values)

            if not constraint_function(best_positions[i],max_weight,weights,values):
                # 引入罚函数
                # best_fitness = objective_function(best_positions[i], max_weight,weights, values) - penalty_factor
                best_fitness = 0
            else:
                best_fitness = objective_function(best_positions[i], max_weight,weights, values)

            # 更新粒子的最佳位置和全局最佳位置
            if fitness > best_fitness:
                best_positions[i] = particles[i].copy()
                if fitness > global_best_fitness:
                    global_best_position = particles[i].copy()
                    global_best_fitness = fitness
            elif best_fitness > global_best_fitness:
                global_best_position = best_positions[i].copy()
                global_best_fitness = best_fitness
        con.append(global_best_fitness)
        # print("Iteration:", iteration, "Global Best Fitness:", global_best_fitness)

    # 输出最优解
    print("Best Solution:", global_best_position)
    print("Best Fitness:", global_best_fitness)
    return global_best_fitness, global_best_position
