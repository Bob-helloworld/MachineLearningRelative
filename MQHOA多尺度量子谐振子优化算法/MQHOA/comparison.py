# coding=gbk

import time

import numpy as np

from mqhoaAs import MqhoaAs

func_num = 1
D = 30  # ���Ժ���ά��
Xmin = [-100, -5.12, -32, -10, 0, -500, -5.12, -5.12, -65.536, -100, -1, -5, -5, -10]  # ����ض���������ͬ�����±߽�
Xmax = [100, 5.12, 32, 10, 10, 500, 5.12, 5.12, 65.536, 100, 1, 10, 10, 10]
# �ر�ע�⣬���޸�func.m�к���˳��֮���������ͨ�õĶ�����һ���޸Ķ�����

pop_size = 20  # ��Ⱥ��
iter_max = 1e4 * D  # ����������
runs = 5  # �ظ�����
funcnum = 2
# ���Ե��ڼ�������
# opt1=[-1400:100:-100]
# opt2=[100:100:1400]
# fopt = [opt1, opt2]  # Ŀ�꺯��������ֵ����ƴ�ӣ�ʹ��CEC13��ʱ����,ע����ʹ����ͨ������ʱ��ERR�����õ�
Algstr = {'MQA_SS05', 'BBFWA', 'CLPSO', 'QPSO', 'SPSO2011', 'mqhoaAs', 'mqhoaA'}
Algnum = 6  # ѡȡʹ�õĲ����㷨(�ϱ�)
tot_time = np.zeros(runs)
xbest = np.zeros((funcnum, D))
fopt = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600, -500, -400, -300, -200,
         -100, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]
fbest = np.zeros((funcnum, runs))
err = np.zeros((funcnum, runs))

# matlab�����Ǵ�2��3��������funcnum��0��1����Ϊmatlab�����Ǵ�1��ʼ��python���Ǵ�0��ʼ����˺���Xmin[i+1]
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

    # �����㷨��������,��ӡ�����Ϣ
    print('---------------------------------------------------------------')
    print(f'Repeat={runs}, Mean FE={np.mean(FES)},Meantime={np.mean(tot_time)}')
    print(f'MeanValue={np.mean(fbest[i, :])},BestValue={np.min(fbest[i, :])},Std={np.std(fbest[i, :])}')
    print(f'MeanErr={np.mean(err[i,:])}, BestErr={np.min(err[i,:])}, StdErr={np.std(err[i,:])}')
    # ��ӡ��ʽ��ʱ��
    print(time.strftime('%Y-%m-%d %H:%M:%S \n', time.localtime(time.time())))


