# coding=gbk
import time

import numpy as np

from func import function


class MqhoaOriginal:
    def __init__(self, Dimension, Particle_Number, Max_Gen, VRmin, VRmax, varargin):
        """
        :param Dimension: ���Ժ���ά��
        :param Particle_Number:��Ⱥ��
        :param Max_Gen:����������
        :param VRmin:�������½�
        :param VRmax:�������Ͻ�
        :param varargin:����func����ID
        """
        self.repeat = 2  # �ظ��������
        self.DIM = Dimension  # Ŀ�꺯��ά��
        self.sigmaMin = 0.000001  # ��С�߶�
        self.groupNum = Particle_Number  # ��˹����������Ŀ����kֵ
        self.minDomain = VRmin  # Ŀ�꺯���������½�
        self.maxDomain = VRmax  # Ŀ�꺯���������Ͻ�
        self.maxFE = Max_Gen  # ����������
        self.funcNum = varargin  # ���õ���func����ID

        self.gbestV = np.zeros((1, self.repeat))  # �洢ÿ�μ�������ź���ֵ
        self.gfe = np.zeros((1, self.repeat))  # �洢ÿ�μ���ĺ�����������
        self.tot_time = np.zeros((1, self.repeat))  # �洢ÿ�μ����ʱ��

    def MQHOA(self):
        start_time = time.time()  # ��ʼ��ʱ
        func = function()  # ��ʼ������
        funcV = np.zeros((1, self.groupNum))  # �洢k��������ĺ���ֵ��������
        samplePos = np.zeros((self.DIM, self.groupNum))  # �洢��ǰk�����������꣨����
        sigma = self.maxDomain - self.minDomain  # ��ǰ�߶ȣ�������
        stdPre = np.zeros((self.DIM, 1))  # ��һ��ÿ��ά���ϲ�����ı�׼�������
        stdNow = np.zeros((self.DIM, 1))  # ��ǰÿ��ά���ϲ�����ı�׼�������
        # �洢��ǰk�����Ž�����꣨����, np.random.uniform����dim��groupNum��������ȷֲ�������
        optimalSolution = np.random.uniform(self.minDomain, self.maxDomain, [self.DIM, self.groupNum])
        evolutiontime = 0  # ����������������(����)
        global_min = np.min(funcV)
        index_min = np.argmin(funcV)
        for i in range(self.repeat):
            for k in range(self.groupNum):
                # optimalSolution[:, k]��ʾ��k�е�����ֵ
                x = optimalSolution[:, k]
                # ��ȡ��funcNum������
                funcV[0, k] = func.get_function(self.funcNum)(x, self.DIM)
                evolutiontime = evolutiontime + 1

            # �߶ȵ�����ʼ
            while evolutiontime < self.maxFE:
                # г���ӵ�����ʼ
                while evolutiontime < self.maxFE:
                    # �ܼ��ȶ�����������ʼ
                    while evolutiontime < self.maxFE:
                        change_flag = 0  # op_solution�����жϱ�־
                        for k in range(self.groupNum):
                            # ����Box-Muller��������DIMά�µ���̬�ֲ�������, np.random.uniform����dim��1��0��1�����������ȷֲ�������
                            theat = 2 * np.pi * np.random.uniform(0, 1, [self.DIM, 1])
                            R = np.sqrt(-2.0 * np.log(np.random.uniform(0, 1, [self.DIM, 1])))
                            gaosiRand = R * np.cos(theat)
                            samplePos[:, k] = optimalSolution[:, k] + sigma * gaosiRand[:, 0]
                            # ����Խ��������������߽��������
                            for d in range(self.DIM):
                                if samplePos[d, k] > self.maxDomain:
                                    samplePos[d, k] = self.minDomain + np.random.random() * (
                                                self.maxDomain - self.minDomain)
                                elif samplePos[d, k] < self.minDomain:
                                    samplePos[d, k] = self.minDomain + np.random.random() * (
                                                self.maxDomain - self.minDomain)
                            # �Ա����β���֮�亯��ֵ�Ĵ�С�����滻���
                            sampleValue = func.get_function(self.funcNum)(samplePos[:, k], self.DIM)
                            evolutiontime = evolutiontime + 1  # ��������������¼
                            if sampleValue < funcV[0, k]:
                                funcV[0, k] = sampleValue
                                optimalSolution[:, k] = samplePos[:, k]
                                change_flag = 1
                        # �Ա�־λchange_flag�����жϣ���ѯ�ܼ��ȶ��������Ƿ��op_solution���й��޸�
                        if change_flag == 0:
                            break
                    # �ܼ��ȶ���������
                    # �ܼ��½���������ֵ�滻
                    meanPos = np.mean(optimalSolution, axis=1)  # �ǰ���ÿһ�о�ֵ��������
                    index_max = np.argmax(funcV)  # ȡ�����ֵ�����index_max
                    optimalSolution[:, index_max] = meanPos  # ��ƽ�������滻���ֵ��Ӧ����
                    funcV[0, index_max] = func.get_function(self.funcNum)(meanPos, self.DIM)
                    evolutiontime = evolutiontime + 1
                    stdPre = np.std(optimalSolution, axis=1, ddof=0)  # �½��׼�����ÿһ�еı�׼��,�����հ��۲�ֵ�������й�һ����
                    # �߶��½��о�
                    if np.max(stdPre) < sigma:
                        break
                # г���ӵ�������
                sigma = sigma / 2.0  # �߶��½�����
                # �����о�
                if sigma <= self.sigmaMin:
                    break
                print(
                    f'sigma={sigma}, best={np.min(funcV)}, evolution time={evolutiontime}, time={time.time() - start_time} ')
            # �߶ȵ�������
            self.tot_time[0, i] = time.time() - start_time
            global_min = np.min(funcV)
            index_min = np.argmin(funcV)
            self.gbestV[0, i] = global_min
            self.gfe[0, i] = evolutiontime
            print(
                f'Func=f{self.funcNum},No.{i} run,DIM={self.DIM},Global minimum={global_min},Fes={evolutiontime},time={time.time() - start_time}')
        # �ظ�����
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
