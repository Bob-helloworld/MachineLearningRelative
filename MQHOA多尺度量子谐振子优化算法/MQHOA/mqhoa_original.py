# coding=gbk
import time

import numpy as np

from func import function


class MqhoaOriginal:
    def __init__(self, Dimension, Particle_Number, Max_Gen, VRmin, VRmax, varargin):
        """
        :param Dimension: 测试函数维度
        :param Particle_Number:种群数
        :param Max_Gen:最大迭代次数
        :param VRmin:定义域下界
        :param VRmax:定义域上界
        :param varargin:调用func函数ID
        """
        self.repeat = 2  # 重复计算次数
        self.DIM = Dimension  # 目标函数维度
        self.sigmaMin = 0.000001  # 最小尺度
        self.groupNum = Particle_Number  # 高斯采样区域数目，即k值
        self.minDomain = VRmin  # 目标函数定义域下界
        self.maxDomain = VRmax  # 目标函数定义域上界
        self.maxFE = Max_Gen  # 最大迭代次数
        self.funcNum = varargin  # 调用调用func函数ID

        self.gbestV = np.zeros((1, self.repeat))  # 存储每次计算的最优函数值
        self.gfe = np.zeros((1, self.repeat))  # 存储每次计算的函数进化次数
        self.tot_time = np.zeros((1, self.repeat))  # 存储每次计算的时间

    def MQHOA(self):
        start_time = time.time()  # 开始计时
        func = function()  # 初始化函数
        funcV = np.zeros((1, self.groupNum))  # 存储k个采样点的函数值（向量）
        samplePos = np.zeros((self.DIM, self.groupNum))  # 存储当前k个采样点坐标（矩阵）
        sigma = self.maxDomain - self.minDomain  # 当前尺度（标量）
        stdPre = np.zeros((self.DIM, 1))  # 上一次每个维度上采样点的标准差（向量）
        stdNow = np.zeros((self.DIM, 1))  # 当前每个维度上采样点的标准差（向量）
        # 存储当前k个最优解的坐标（矩阵）, np.random.uniform生成dim×groupNum的随机均匀分布的数组
        optimalSolution = np.random.uniform(self.minDomain, self.maxDomain, [self.DIM, self.groupNum])
        evolutiontime = 0  # 函数进化次数计数(标量)
        global_min = np.min(funcV)
        index_min = np.argmin(funcV)
        for i in range(self.repeat):
            for k in range(self.groupNum):
                # optimalSolution[:, k]表示第k列的所有值
                x = optimalSolution[:, k]
                # 获取第funcNum个方法
                funcV[0, k] = func.get_function(self.funcNum)(x, self.DIM)
                evolutiontime = evolutiontime + 1

            # 尺度迭代开始
            while evolutiontime < self.maxFE:
                # 谐振子迭代开始
                while evolutiontime < self.maxFE:
                    # 能级稳定收敛迭代开始
                    while evolutiontime < self.maxFE:
                        change_flag = 0  # op_solution更新判断标志
                        for k in range(self.groupNum):
                            # 采用Box-Muller方法生成DIM维新的正态分布采样点, np.random.uniform生成dim×1的0到1区间的随机均匀分布的数组
                            theat = 2 * np.pi * np.random.uniform(0, 1, [self.DIM, 1])
                            R = np.sqrt(-2.0 * np.log(np.random.uniform(0, 1, [self.DIM, 1])))
                            gaosiRand = R * np.cos(theat)
                            samplePos[:, k] = optimalSolution[:, k] + sigma * gaosiRand[:, 0]
                            # 对于越界采样点采用最近边界坐标替代
                            for d in range(self.DIM):
                                if samplePos[d, k] > self.maxDomain:
                                    samplePos[d, k] = self.minDomain + np.random.random() * (
                                                self.maxDomain - self.minDomain)
                                elif samplePos[d, k] < self.minDomain:
                                    samplePos[d, k] = self.minDomain + np.random.random() * (
                                                self.maxDomain - self.minDomain)
                            # 对比两次采样之间函数值的大小，并替换差解
                            sampleValue = func.get_function(self.funcNum)(samplePos[:, k], self.DIM)
                            evolutiontime = evolutiontime + 1  # 函数进化次数记录
                            if sampleValue < funcV[0, k]:
                                funcV[0, k] = sampleValue
                                optimalSolution[:, k] = samplePos[:, k]
                                change_flag = 1
                        # 对标志位change_flag进行判断，查询能级稳定过程中是否对op_solution进行过修改
                        if change_flag == 0:
                            break
                    # 能级稳定迭代结束
                    # 能级下降操作，均值替换
                    meanPos = np.mean(optimalSolution, axis=1)  # 是包含每一行均值的列向量
                    index_max = np.argmax(funcV)  # 取得最大值的序号index_max
                    optimalSolution[:, index_max] = meanPos  # 用平均坐标替换最大值对应坐标
                    funcV[0, index_max] = func.get_function(self.funcNum)(meanPos, self.DIM)
                    evolutiontime = evolutiontime + 1
                    stdPre = np.std(optimalSolution, axis=1, ddof=0)  # 新解标准差（计算每一行的标准差,并按照按观测值数量进行归一化）
                    # 尺度下降判据
                    if np.max(stdPre) < sigma:
                        break
                # 谐振子迭代结束
                sigma = sigma / 2.0  # 尺度下降操作
                # 精度判据
                if sigma <= self.sigmaMin:
                    break
                print(
                    f'sigma={sigma}, best={np.min(funcV)}, evolution time={evolutiontime}, time={time.time() - start_time} ')
            # 尺度迭代结束
            self.tot_time[0, i] = time.time() - start_time
            global_min = np.min(funcV)
            index_min = np.argmin(funcV)
            self.gbestV[0, i] = global_min
            self.gfe[0, i] = evolutiontime
            print(
                f'Func=f{self.funcNum},No.{i} run,DIM={self.DIM},Global minimum={global_min},Fes={evolutiontime},time={time.time() - start_time}')
        # 重复计算
        fitcount = evolutiontime
        gbestval = global_min
        gbest = optimalSolution[:, index_min]
        return gbest, gbestval, fitcount


DIM = 20
sigmaMin = 0.000001
groupNum = 40
minDomain = -100
maxDomain = 100
maxFE = (1e4 * DIM)
funNo = 3
mqhoa_original = MqhoaOriginal(DIM, groupNum, maxFE, minDomain, maxDomain, funNo)
mqhoa_original.MQHOA()
