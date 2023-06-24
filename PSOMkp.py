import math
import numpy as np
import random


# 定义MKP问题的参数和约束条件

num_particles = 700  # 粒子数量
max_iterations = 1200  # 最大迭代次数
# penalty_factor = 2000  # 罚函数的惩罚因子
c1 = 1.5 # 学习因子1
c2 = 1.5 # 学习因子2


# 定义MKP问题的目标函数和约束条件
def objective_function(position,max_weight, weights, values,m):
    # 计算目标函数值最大化价值总和
    ans = 0
    for i in range(len(position)):
        if position[i] > m:
            ans += values[i]*1
            position[i] = 1
    return ans, position


def constraint_function(position, max_weight, weights, values,m):
    # 判断是否满足约束条件背包的承重不能超过最大承重
    weight = 0
    for i in range(len(position)):
        if position[i] > m:
            weight += weights[i]
    # weight = np.sum(position * weights)
    # print(weight)
    if weight > max_weight:
        return False
    return True


def mkp_pso(weights, values, max_weight):
    con = []
    # 初始化粒子的位置和速度
    num_items = len(values)  # 物品数量
    velocities = np.random.uniform(-1, 1, size=(num_particles, num_items))
    # 初始化粒子的最佳位置和全局最佳位置
    best_positions = np.zeros((num_particles, num_items), dtype=float)
    best_fitness = 0
    particles = np.random.uniform(0, 1, size=(num_particles, num_items))
    global_best_position = particles[1].copy()
    global_best_fitness = 0

    # c1_min = 1
    # c1_max = 2
    # c2_min = 1
    # c2_max = 2
    w_min = 0
    w_max = 1
    # c1_range = c1_max - c1_min
    # c2_range = c2_max - c2_min
    w_range = w_max - w_min
    # c1 = np.random.uniform(c1_min, c1_max, size=num_particles)
    # c2 = np.random.uniform(c2_min, c2_max, size=num_particles)
    w = np.random.uniform(w_min, w_max, size=num_particles)

    # 迭代优化过程
    count = 0
    for iteration in range(max_iterations):
        # count = count + 1
        # print(count)
        m = 0
        for i in range(num_particles):
            # 更新速度和位置
            velocities[i] = velocities[i] + c1*random.random() * (best_positions[i] - particles[i]) + c2* random.random() * (global_best_position - particles[i])
            # 根据速度更新位置
            min_v = np.min(velocities[i])
            max_v = np.max(velocities[i])
            velocities[i] = (velocities[i] - min_v) * 2 /(max_v - min_v)-1
            particles[i] = particles[i] + velocities[i]
            particles[i][particles[i] > 1] = 1
            particles[i][particles[i] < 0] = 0
            # particles[i] = (particles[i] + 1) / 3
            m = random.uniform(0.35, 0.6)
            # m = random.random()
            # 处理约束条件

            if not constraint_function(particles[i], max_weight, weights, values, m):
                # 引入罚函数
                # fitness = objective_function(particles[i], max_weight,weights, values) - penalty_factor
                fitness = 0
            else:
                fitness, particles[i] = objective_function(particles[i], max_weight, weights, values, m)

            # 更新粒子的最佳位置和全局最佳位置
            if fitness > best_fitness:
                best_positions[i] = particles[i].copy()
                best_fitness = fitness
            if best_fitness > global_best_fitness:
                global_best_position = best_positions[i].copy()
                global_best_fitness = best_fitness

        # sum_best_fitness = 0
        # for p in best_positions:
        #     sum_best_fitness += objective_function(p,max_weight,weights,values,m)[0]
        # mean_best_fitness = sum_best_fitness / num_particles
        # mean_best_fitness = np.mean([objective_function(p, max_weight, weights, values, m)[0] for p in best_positions])
        # mean_global_best_fitness = objective_function(global_best_position, max_weight, weights, values, m)[0]
        # #
        # # # 自适应调整学习因子和惯性权重
        # # print(w)
        # for i in range(num_particles):
        #     if best_fitness > mean_best_fitness:
        #         # c1[i] = c1[i] + c1_range * (best_fitness - mean_best_fitness) / (
        #         #             mean_global_best_fitness - mean_best_fitness)
        #         # c2[i] = c2[i] + c2_range * (best_fitness - mean_best_fitness) / (
        #         #             mean_global_best_fitness - mean_best_fitness)
        #         w[i] = w[i] + w_range * (best_fitness - mean_best_fitness) / (
        #                     mean_global_best_fitness - mean_best_fitness)
        #     else:
        #         # c1[i] = c1[i] - c1_range * (best_fitness - mean_best_fitness) / (
        #         #             mean_global_best_fitness - mean_best_fitness)
        #         # c2[i] = c2[i] - c2_range * (best_fitness - mean_best_fitness) / (
        #         #             mean_global_best_fitness - mean_best_fitness)
        #         w[i] = w[i] - w_range * (best_fitness - mean_best_fitness) / (
        #                     mean_global_best_fitness - mean_best_fitness)
        # print("Iteration:", iteration, "Global Best Fitness:", global_best_fitness)
        con.append(global_best_fitness)
    # 输出最优解
    global_best_position[global_best_position >= 1 ] = 1
    global_best_position[global_best_position < 1] = 0
    # print("Best Solution:", global_best_position)
    print("Best Fitness:", global_best_fitness)
    return global_best_fitness, global_best_position
