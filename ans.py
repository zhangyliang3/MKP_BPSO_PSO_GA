from os.path import basename, normpath
import glob
import DpMkp
import numpy
import time
import logging
import BPSOMkp
import GAMkp
import PSOMkp


def read_input_file(path):
    kind = 0
    bag_capacity = 0
    list = []
    with open(path, "r") as fil:
        n = fil.readline().strip()
        assert n.isdigit()
        kind = int(n)

        bag_capacity = float(fil.readline().strip())
        for line in fil:
            tokens = line.strip().split(" ")
            temp = []
            assert len(tokens) == 4
            for i in range(4):
                temp.append(tokens[i])
            list.append(temp)
    return kind, bag_capacity, list


if __name__ == '__main__':
    # inputs = glob.glob('temp/*')
    # 测试用
    inputs = glob.glob('dataset/*')
    # 配置日志文件和获取时间戳
    logging.basicConfig(filename='logFile/PSOLog.log', level=logging.INFO)
    timestamp1 = time.time()
    timestamp3 = time.time()
    logging.info(f'Total startTimestamp: {timestamp1}')

    anss = []
    files = []
    count = 0
    for input_path in inputs:
        itemname = basename(normpath(input_path))[:-3]
        # if itemname.__contains__("ooo") or itemname.__contains__("xxxx"):
        #     print("含有")
        #     continue
        # elif count > 168:
        #     continue
        print(itemname)
        count = count + 1
        kind_num, bag_cap, items = read_input_file(input_path)
        ids = []
        values = [0]
        weights = [0]
        for item in items:
            for i in range(int(item[1])):
                ids.append(item[0])
                values.append(item[2])
                weights.append(float(item[3])*100)
        # dp求解
        # ans, best_solve = DpMkp.mkp_dp(ids, numpy.array(values), numpy.array(weights), int(bag_cap*100))
        # output_path = 'DpOutput/' + basename(normpath(input_path))[:-3] + '.out'
        # BPSO\GA求解
        values.pop(0)
        weights.pop(0)
        v = numpy.array(values)
        v = v.astype(float)
        w = numpy.array(weights)
        w = w.astype(float)
        # BPSO求解
        # ans, best_solve = BPSOMkp.mkp_bpso(w, v, int(bag_cap*100))
        # GA求解，1061.34，1110.83
        # ans, best_solve = GAMkp.GaMkp(w, v, int(bag_cap*100))
        # PSO求解，6255.92，678.49，1110.83
        ans, best_solve = PSOMkp.mkp_pso(w, v, int(bag_cap*100))
        anss.append(ans)
        files.append(itemname)
        output_path = 'PSOOutput/' + basename(normpath(input_path))[:-3] + '.out'
        # BPSO
        with open(output_path, "w") as fw:
            fw.write(f"{ans}\n")
            for i in best_solve:
                fw.write(f"{i}\n")
        fw.close()
        timestamp = time.time()
        # logging.info(f'{itemname}endTimestamp: {timestamp}')
        print(itemname, "已完成")
        # if count == 243:
        #     count = 0
        #     timetemp = time.time()
        #     ttt = timetemp - timestamp3
        #     timestamp3 = timetemp
        #     logging.info(f'each cost time: {ttt}')

    # 将时间戳写入log文件
    timestamp2 = time.time()
    logging.info(f'Total endTimestamp: {timestamp2}')
    t = timestamp2 - timestamp1
    logging.info(f'cost all time: {t}')

    # 打开txt文件进行写入
    with open("PSOOutput.txt", "w") as file:
        # 使用zip函数将文件名称和数值组合起来
        for file_name, value in zip(files, anss):
            # 将文件名和数值按指定格式写入到txt文件中
            file.write(f"{file_name}: {value}\n")
