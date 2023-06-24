import random
import numpy as np

# 遗传算法的参数
population_size = 36  # 种群大小
# mutation_rate = 0.003  # 变异率
cross_rate = 0.8
num_generations = 2500  # 迭代次数
num_local_search_steps = 13  # 局部搜索步数
num_crossover_points = 6 # 多点交叉算子的交叉点数量，4个交叉点，交叉两段
con = []

def penatly(chromosome, weights, capacity):
    total_weights =sum([weights[i] for i in range(len(chromosome)) if chromosome[i] == 1])
    if total_weights > capacity:
        return False
    else:
        return True


def fitness(chromosome, values, weights, capacity):
    # 计算染色体的适应度（总价值）
    total_value = 0
    if penatly(chromosome,weights,capacity):
        total_value = sum([values[i] for i in range(len(chromosome)) if chromosome[i] == 1])
    return total_value


# def crossover(parent1, parent2):
#     # 单点交叉产生子代
#     if random.random() < cross_rate:
#         crossover_point1 = random.randint(1, len(parent1)-1)
#         crossover_point2 = abs(int(len(parent1)/10) - crossover_point1)
#         if crossover_point2 > crossover_point1:
#             child1 = parent1[:crossover_point1] + parent2[crossover_point1: crossover_point2] + parent1[crossover_point2:]
#             child2 = parent2[:crossover_point1] + parent1[crossover_point1: crossover_point2] + parent2[crossover_point2:]
#         else:
#             child1 = parent1[:crossover_point2] + parent2[crossover_point2: crossover_point1] + parent1[crossover_point1:]
#             child2 = parent2[:crossover_point2] + parent1[crossover_point2: crossover_point1] + parent2[crossover_point1:]
#         return child1, child2
#     else:
#         return parent1, parent2

def crossover(parent1, parent2):
    # 多点交叉产生子代
    crossover_points = sorted(random.sample(range(len(parent1)), num_crossover_points))
    child1 = parent1.copy()
    child2 = parent2.copy()
    for i in range(0, len(crossover_points), 2):
        start = crossover_points[i]
        end = crossover_points[i + 1] if i + 1 < len(crossover_points) else len(parent1)
        child1[start:end], child2[start:end] = child2[start:end], child1[start:end]
    return child1, child2


def mutate(chromosome,weights,mutation_rate):
    # 突变操作
    # mutation_rate = 100/len(weights)
    # if mutation_rate > 0.1:
    #     mutation_rate = 0.1
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]
    return chromosome


def generate_individual(chromosome_length):
    chromosome = [0] * chromosome_length
    step = int(chromosome_length/5)
    for i in range(0, chromosome_length, step):
        start_range = i
        end_range = i + step
        if end_range >= chromosome_length:
            break
        gene = random.randint(start_range, end_range)
        # print(gene)
        chromosome[gene] = 1

    return chromosome


def generate_initial_population(weights, capacity):
    # 生成初始种群
    population = []
    possbile_weight = [9.99, 0.01]
    i = 0
    count = 0
    while i < population_size:
        count = count + 1
        # chromosome1 = [random.randint(0, 1) for _ in range(len(weights))]
        # 随机采样
        chromosome = random.choices((0, 1), weights=possbile_weight, k=len(weights))
        # 均匀采样
        # chromosome = generate_individual(len(weights))
        # print(count)
        if penatly(chromosome,weights,capacity):
            population.append(chromosome)
            i = i + 1
        else:
            indices = [i for i, value in enumerate(chromosome) if value == 1]
            while 1:
                random_index = random.choice(indices)
                indices.remove(random_index)
                chromosome[random_index] = 0
                if penatly(chromosome, weights, capacity):
                    population.append(chromosome)
                    i = i + 1
                    break
    return population


def select_parents(population, values, weights, capacity):
    # 锦标赛选择父代
    tournament_size = 7
    parents = []
    for _ in range(2):
        candidates = random.sample(population, tournament_size)
        mmax = -1
        temp_parent = candidates[0]
        for i in range(tournament_size):
            tempfit = fitness(candidates[i], values, weights, capacity)
            if tempfit > mmax:
                mmax = tempfit
                temp_parent = candidates[i]
        parents.append(temp_parent)
    return parents


def select_best(population, values, weights, capacity):
    mmax = -1
    for i in range(len(population)):
        tempfit = fitness(population[i], values, weights, capacity)
        if tempfit > mmax:
            mmax = tempfit
            temp_parent = population[i]
    return temp_parent


def select_worst(population, values, weights, capacity):
    mmin = 10000
    for i in range(len(population)):
        tempfit = fitness(population[i], values, weights, capacity)
        if tempfit < mmin:
            mmin = tempfit
            temp_parent = population[i]
    return temp_parent


# 局部搜索（爬山算法）
def local_search(chromosome,values,weights,capacity):
    current_fitness = fitness(chromosome,values,weights,capacity)
    for _ in range(num_local_search_steps):
        index = random.randint(0, len(chromosome)-1)
        new_chromosome = chromosome.copy()
        new_chromosome[index] = 1 - new_chromosome[index]
        new_fitness = fitness(new_chromosome,values,weights,capacity)
        if new_fitness > current_fitness:
            chromosome = new_chromosome
            current_fitness = new_fitness
    return chromosome


# 使用线性插值计算动态变异率
def calculate_mutation_rate(generation, mutation_rate_initial, mutation_rate_final):
    t = generation / num_generations
    mutation_rate = (1 - t) * mutation_rate_initial + t * mutation_rate_final
    return mutation_rate


def genetic_algorithm(weights, values, capacity):
    # 初始化种群
    population = generate_initial_population(weights, capacity)

    global_best_soultion = population[0]
    global_best_fitness = fitness(population[0], values, weights, capacity)
    count = 0
    # 进化循环
    for generation in range(num_generations):
        new_population = []
        # 生成新的种群
        # 精英保留机制
        new_population.append(global_best_soultion)
        m = calculate_mutation_rate(generation, 0.2, 0.01)
        while len(new_population) < population_size:
            parent1, parent2 = select_parents(population, values, weights, capacity)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, weights, m)
            child2 = mutate(child2, weights, m)
            if penatly(child1, weights, capacity):
                new_population.append(child1)
            else:
                indices = [i for i, value in enumerate(child1) if value == 1]
                while 1:
                    random_index = random.choice(indices)
                    indices.remove(random_index)
                    child1[random_index] = 0
                    if penatly(child1,weights,capacity):
                        new_population.append(child1)
                        break

            if penatly(child2, weights, capacity):
                new_population.append(child2)
            else:
                indices = [i for i, value in enumerate(child2) if value == 1]
                while 1:
                    random_index = random.choice(indices)
                    indices.remove(random_index)
                    child2[random_index] = 0
                    if penatly(child2, weights, capacity):
                        new_population.append(child2)
                        break

        for i in range(population_size):
            new_population[i] = local_search(new_population[i],values,weights,capacity)

        population = new_population
        best_solution = select_best(population,values,weights,capacity)
        # worst_solution = select_worst(population, values, weights, capacity)
        best_fitness = fitness(best_solution,values,weights,capacity)
        if best_fitness > global_best_fitness:
            global_best_fitness = best_fitness
            global_best_soultion = best_solution
        # population.remove(worst_solution)
        # population.append(global_best_soultion)
        if best_fitness == global_best_fitness:
            count = count + 1
        if count == 1200:
            break
        # print("best_fitness",generation, ": ", global_best_fitness)
        con.append(global_best_fitness)
    # 选择最优解
    # best_solution =select_best(population,values,weights,capacity)
    # best_fitness = fitness(best_solution,values,weights,capacity)
    return global_best_soultion, global_best_fitness


def GaMkp(weights, values, capacity):
    # 运行遗传算法
    best_solution, best_fitness = genetic_algorithm(weights, values, capacity)

    print("最优解：", best_solution)
    print("最优适应度：", best_fitness)

    return best_fitness, best_solution,con

