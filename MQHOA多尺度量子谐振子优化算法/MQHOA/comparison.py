# coding=gbk

import time

import numpy as np

from mqhoaAs import MqhoaAs

func_num = 1
D = 30  # 测试函数维度
Xmin = [-100, -5.12, -32, -10, 0, -500, -5.12, -5.12, -65.536, -100, -1, -5, -5, -10]  # 针对特定函数，不同的上下边界
Xmax = [100, 5.12, 32, 10, 10, 500, 5.12, 5.12, 65.536, 100, 1, 10, 10, 10]
# 特别注意，在修改func.m中函数顺序之后，如果不是通用的定义域，一定修改定义域

pop_size = 20  # 种群数
iter_max = 1e4 * D  # 最大迭代次数
runs = 5  # 重复次数
funcnum = 2
# 测试到第几个函数
# opt1=[-1400:100:-100]
# opt2=[100:100:1400]
# fopt = [opt1, opt2]  # 目标函数的最优值数组拼接，使用CEC13的时候用,注意在使用普通函数的时候ERR是无用的
Algstr = {'MQA_SS05', 'BBFWA', 'CLPSO', 'QPSO', 'SPSO2011', 'mqhoaAs', 'mqhoaA'}
Algnum = 6  # 选取使用的测试算法(上表)
tot_time = np.zeros(runs)
xbest = np.zeros((funcnum, D))
fopt = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600, -500, -400, -300, -200,
         -100, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]
fbest = np.zeros((funcnum, runs))
err = np.zeros((funcnum, runs))

# matlab代码是从2到3，在这里funcnum是0和1。因为matlab数组是从1开始，python中是从0开始，因此后面Xmin[i+1]
for i in range(funcnum):
    func_num = i
    for j in range(runs):
        start_time = time.time()
        if Algnum == 6:
            gbest, gbestval, FES = MqhoaAs(D, pop_size, iter_max, Xmin[i+1], Xmax[i+1], func_num).MQHOA()
        print(f'Func=f{i},No.{j} run,DIM={D},Global minimum={gbestval},Fes={FES},time={time.time() - start_time}')
        print(f'Xmin={Xmin[i+1]},Xmax={Xmax[i+1]}')
        tot_time[j] = time.time() - start_time
        xbest[i, :] = gbest
        fbest[i, j] = gbestval
        err[i, j] = gbestval - fopt[i+1]

    # 测试算法迭代结束,打印相关信息
    print('---------------------------------------------------------------')
    print(f'Repeat={runs}, Mean FE={np.mean(FES)},Meantime={np.mean(tot_time)}')
    print(f'MeanValue={np.mean(fbest[i, :])},BestValue={np.min(fbest[i, :])},Std={np.std(fbest[i, :])}')
    print(f'MeanErr={np.mean(err[i,:])}, BestErr={np.min(err[i,:])}, StdErr={np.std(err[i,:])}')
    # 打印格式化时间
    print(time.strftime('%Y-%m-%d %H:%M:%S \n', time.localtime(time.time())))


