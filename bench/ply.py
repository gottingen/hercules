import codecs
import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import csv
import matplotlib.pyplot as plt

exampleFile = open('b1.csv')  # 打开csv文件
exampleReader = csv.reader(exampleFile)  # 读取csv文件
exampleData = list(exampleReader)  # csv数据转换为列表
length_zu = len(exampleData)  # 得到数据行数
length_yuan = len(exampleData[0])  # 得到每行长度

# for i in range(1,length_zu):
#     print(exampleData[i])

x = list()
y = list()

for i in range(1, length_zu):  # 从第二行开始读取
    x.append(exampleData[i][0])  # 将第一列数据从第二行读取到最后一行赋给列表x
    y.append(exampleData[i][1])  # 将第二列数据从第二行读取到最后一行赋给列表

plt.plot(x, y)  # 绘制x,y的折线图
plt.show()  # 显示折线图

