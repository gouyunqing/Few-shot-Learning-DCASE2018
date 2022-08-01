if __name__ == '__main__':
    new_lines = []
    with open('D:\毕设相关\第一次尝试\Train_Scp.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            new_line = line.split('\\', 1)[0] + '_FSL\\' + line.split('\\', 1)[1]
            new_lines.append(new_line)

    with open('D:\毕设相关\第二次尝试\Train_Scp.txt', 'w', encoding='utf-8') as f:
        for line in new_lines:
            f.write(line)
            f.write('\n')