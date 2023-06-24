from os.path import basename, normpath
import glob


def process_input_file(path):
    with open(path, "r") as fo:
        n = fo.readline().strip()
        assert n.isdigit()
        n = int(n)

        m = float(fo.readline().strip())

        data = {}  # 用于存储合并后的数据

        for line in fo:
            tokens = line.strip().split(" ")
            assert len(tokens) == 4

            id = int(tokens[0])  # 第一列作为id
            if id not in data:
                data[id] = [0, 0, 0]  # 初始化数据列表 [数量, 第三列合计, 第四列合计]

            count = int(tokens[1])  # 第二列作为数量
            data[id][0] += 1

            col3 = float(tokens[2])  # 第三列数据按数量加权合计
            data[id][1] += col3

            col4 = float(tokens[3])  # 第四列数据按数量加权合计
            data[id][2] += col4

    # 将处理后的数据写入新文件
    output_path = 'dataset/' + basename(normpath(input_path))[:-3] + '.in'
    with open(output_path, "w") as fo:
        fo.write(f"{n}\n")
        fo.write(f"{m}\n")
        for id, values in data.items():
            count = values[0]
            col3_avg = values[1] / count
            col3_avg = round(col3_avg, 2)
            col4_avg = values[2] / count
            col4_avg = round(col4_avg, 2)
            fo.write(f"{id} {count} {col3_avg} {col4_avg}\n")
    return output_path


if __name__ == '__main__':
    inputs = glob.glob('old-dataset/*')
    for input_path in inputs:
        output_path = 'dataset/' + basename(normpath(input_path))[:-3] + '.in'
        process_input_file(input_path)