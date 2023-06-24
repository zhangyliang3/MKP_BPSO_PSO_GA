import numpy as np


# 动态规划求解mkp，动态规划的时候对容量进行了一个取整操作，方便构建数组
def mkp_dp(ids, values, weights, capacities):
    num = len(ids)
    dp = np.zeros((num + 1, int(capacities + 1)), dtype=float)
    rec = np.zeros((num + 1, int(capacities + 1)), dtype=float)
    show_item = []
    # 利用动态规划求解过程，rec用来记录是否选取商品。
    for i in range(1, num + 1):
        for c in range(1, int(capacities + 1)):
            weight_temp = int(weights[i])
            val = float(values[i])
            # dp核心判断条件
            if weight_temp <= c and dp[i - 1, c - weight_temp] + val > dp[i - 1][c]:
                    dp[i][c] = dp[i - 1, c - weight_temp] + val
                    rec[i][c] = 1
            else:
                rec[i][c] = 0
                dp[i][c] = dp[i - 1][c]
    # 回溯法，输出最优解决方案,本步我们将其放入list，方便输出到out文件
    k = int(capacities)
    for i in range(num, 0, -1):
        if rec[i, k] == 1:
            # print(ids[i-1])
            show_item.append(ids[i-1])
            k = k - int(weights[i])
    return round(dp[num, capacities], 4), show_item
