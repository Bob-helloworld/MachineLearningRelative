# coding=gbk
import math


class function:

    def get_function(self, case):
        fun_name = "func_" + str(case)
        method = getattr(self, fun_name, self.func_other)
        return method

    # 对应matlab func.m中case 2
    def func_0(self, x, dimension):
        s = 0
        for j in range(dimension):
            s = s + (x[j] ** 2 - 10 * math.cos(2 * math.pi * x[j]))
        y = 10 * dimension + s
        return y

    # 对应matlab func.m中case 3
    def func_1(self, x, dimension):
        n = dimension
        a = 20
        b = 0.2
        c = 2 * math.pi
        s1 = 0
        s2 = 0
        for i in range(dimension):
            s1 = s1 + x[i] ** 2
            s2 = s2 + math.cos(c * x[i])
        y = -a * math.exp(-b * math.sqrt(1 / n * s1)) - math.exp(1 / n * s2) + a + math.exp(1)
        return y

    # 对应matlab func.m中case 1
    def func_2(self, x, dimension):
        s = 0
        k = 1
        for i in range(dimension):
            s = x[i] ** 2 + s
            k = math.cos(x[i]/math.sqrt(i+1)) * k
        s = s / 4000
        y = s - k + 1
        return y

    def func_3(self, x, dimension):
        s = 0
        for j in range(dimension):
            s += x[j] ** 2
        return s

    def func_other(self, x, dimension):
        print("function other")
