import matplotlib.pyplot as plt
from RK import *
import numpy as np
from numpy import trapz
from scipy.integrate import simpson
from scipy import signal
from random import randrange

leftCoefs = [0.0000063, 0.0071, 0.12, 0]
rightCoefs = [2]
initState = [0, 0, 0]
inputFunc = 1
step = 0.001
regTime = 2
regStructList = [1, 0, 0]


# my_RK = Runge_Kutta(leftCoefs, rightCoefs, initState, inputFunc, step, 1.5 * regTime,regP, None, None, Tr = regTime / 5)
# my_RK.solve()
# my_RK.plot_solution()


def abs_stability_criteria(regStructList, reg, Hp, Hi, Hd, tolerance):
    if regStructList[0] == 1:
        K1p = reg[0][0][1] / reg[0][0][0]
        K2p = (reg[0][1][1] - reg[0][0][1]) / (reg[0][1][0] - reg[0][0][0])
        KpMax = max(K1p, K2p)
    else:
        KpMax = 0
    if regStructList[1] == 1:
        K1i = reg[1][0][1] / reg[1][0][0]
        K2i = (reg[1][1][1] - reg[1][0][1]) / (reg[1][1][0] - reg[1][0][0])
        KiMax = max(K1i, K2i)
    else:
        KiMax = 0
    if regStructList[2] == 1:
        K1d = reg[2][0][1] / reg[2][0][0]
        K2d = (reg[2][1][1] - reg[2][0][1]) / (reg[2][1][0] - reg[2][0][0])
        KdMax = max(K1d, K2d)
    else:
        KdMax = 0
    for i in range(len(Hp.real)):
        tmp = np.matrix([[1 + KpMax * Hp.real[i], KpMax * Hp.real[i], KpMax * Hp.real[i]],
                         [KiMax * Hi.real[i], 1 + KiMax * Hi.real[i], KiMax * Hi.real[i]],
                         [KdMax * Hd.real[i], KdMax * Hd.real[i], 1 + KdMax * Hd.real[i]]])
        res = np.linalg.det(tmp)
        if abs(res) <= tolerance:
            return -1
    return 0


def func(leftCoefs, rightCoefs, initState, inputFunc, step, regTime, Tr, tolerance, reg):
    satLevel = reg[1][1]
    plantTF = signal.TransferFunction(rightCoefs, leftCoefs)
    wp, Hp = signal.freqresp(plantTF)
    tmp = leftCoefs[:]
    tmp.extend([0, ])
    plantIntgrTF = signal.TransferFunction(rightCoefs, tmp)
    wi, Hi = signal.freqresp(plantIntgrTF)
    tmp = rightCoefs[:]
    tmp.extend([0, ])
    plantDrvtTF = signal.TransferFunction(tmp, leftCoefs)
    wd, Hd = signal.freqresp(plantDrvtTF)

    regsList = []

    dList = [0, 0, 0]
    cList = [0, 0, 0]
    bList = [0, 0, 0]
    aList = [0, 0, 0]

    if leftCoefs[-1] != 0:
        if regStructList[1] == 1:
            dList[1] = random.uniform(inputFunc * leftCoefs[-1] / (satLevel * rightCoefs[-1]),
                                      2 * inputFunc * leftCoefs[-1] / (satLevel * rightCoefs[-1]))
            temp = 1 - dList[1]
        if regStructList[0] == 1:
            dList[0] = random.uniform(0, temp)
        if regStructList[2] == 1:
            dList[2] = random.uniform(0, 0.1)
    else:
        if regStructList[0] == 1:
            dList[0] = random.uniform(0.5, 1)
        if regStructList[1] == 1:
            dList[1] = random.uniform(0, 0.4)
        if regStructList[2] == 1:
            dList[2] = random.uniform(0, 0.1)

    # Normalize this values to have sum of saturation
    sumDList = sum(dList)
    for i in range(len(dList)):
        dList[i] = dList[i] * satLevel / sumDList

    for i in range(len(regStructList)):
        if regStructList[i] == 1:
            bList[i] = random.uniform(0, dList[i])

    if regStructList[0] == 1:
        cList[0] = random.uniform(inputFunc, 2 * inputFunc)
    if regStructList[1] == 1:
        cList[1] = random.uniform(inputFunc * timeConst, 5 * inputFunc * timeConst)
    if regStructList[2] == 1:
        cList[2] = random.uniform(inputFunc / timeConst, 5 * inputFunc / timeConst)

    if regStructList[0] == 1:
        aList[0] = random.uniform(0, inputFunc)
    if regStructList[1] == 1:
        aList[1] = random.uniform(0, inputFunc * timeConst)
    if regStructList[2] == 1:
        aList[2] = random.uniform(0, inputFunc / timeConst)

    # Check if Yackubovich condition is met
    tmpReg = []
    if regStructList[0] == 1:
        tmpReg.append([[aList[0], bList[0]], [cList[0], dList[0]]])
    else:
        tmpReg.append(None)
    if regStructList[1] == 1:
        tmpReg.append([[aList[1], bList[1]], [cList[1], dList[1]]])
    else:
        tmpReg.append(None)
    if regStructList[2] == 1:
        tmpReg.append([[aList[2], bList[2]], [cList[2], dList[2]]])
    else:
        tmpReg.append(None)

    res = abs_stability_criteria(regStructList, reg, Hp, Hi, Hd, 0.001)

    return res


def simps(y, step):
    return simpson(y, dx=step)


alpha = 1  # коэффициент отражения alpha 1
beta = 0.5  # коэффициент сжатия beta  выбирается равным 0.5
gamma = 2  # коэффициент растяжения gamma 2


# Waycro check stability
# func(leftCoefs, rightCoefs, initState, inputFunc, step, regTime, reg1, reg2, Tr, step_grad, lr, epochs, tolerance)
def Nelder_Mid(leftCoefs, rightCoefs, initState, inputFunc, step, regTime, Tr, tolerance, reg, alpha, beta, gamma):
    kp_list = list()
    func_list = list()

    kp1 = reg[0]

    a, aelta = kp1[0][0], 3
    b, belta = kp1[0][1], 1
    c, celta = kp1[1][0], 8
    d, delta = kp1[1][1], 130

    kp2 = np.array(
        [[abs(np.random.uniform(a - aelta, a + aelta)), abs(np.random.uniform(b - belta, b + belta))], \
         [abs(np.random.uniform(c - celta, c + celta)), abs(np.random.uniform(d - delta, d + delta))]])

    kp3 = np.array(
        [[abs(np.random.uniform(a - aelta, a + aelta)), abs(np.random.uniform(b - belta, b + belta))], \
         [abs(np.random.uniform(c - celta, c + celta)), abs(np.random.uniform(d - delta, d + delta))]])

    kp4 = np.array(
        [[abs(np.random.uniform(a - aelta, a + aelta)), abs(np.random.uniform(b - belta, b + belta))], \
         [abs(np.random.uniform(c - celta, c + celta)), abs(np.random.uniform(d - delta, d + delta))]])

    kp5 = np.array(
        [[abs(np.random.uniform(a - aelta, a + aelta)), abs(np.random.uniform(b - belta, b + belta))], \
         [abs(np.random.uniform(c - celta, c + celta)), abs(np.random.uniform(d - delta, d + delta))]])

    kp_list = [kp1, kp2, kp3, kp4, kp5]

    for el in kp_list:
        my_RK = Runge_Kutta(leftCoefs, rightCoefs, initState, inputFunc, step, 1.5 * regTime,
                            el, None, None, Tr=regTime / 5)
        my_RK.solve()
        y = my_RK.e0_solve
        e = simpson(abs(np.array(y)), dx=step)
        func_list.append(e)

    return kp_list, func_list,


def Nedler_Mid(kp_list, func_list, leftCoefs, rightCoefs, initState, inputFunc, step, regTime, Tr, tolerance, alpha,
               beta, gamma):
    key_val = zip(kp_list, func_list)

    key_val = sorted(key_val, key=lambda tup: tup[1])

    kp_list = [x[0] for x in key_val]
    func_list = [x[1] for x in key_val]

    kp_center_mass = sum(kp_list[:-1]) / len(kp_list[:-1])

    kp_r = (1 + alpha) * kp_center_mass - alpha * kp_list[4]

    my_RK = Runge_Kutta(leftCoefs, rightCoefs, initState, inputFunc, step, 1.5 * regTime,
                        kp_r, None, None, Tr=regTime / 5)
    my_RK.solve()
    y = my_RK.e0_solve
    e_r = simpson(abs(np.array(y)), dx=step)

    if e_r < func_list[0]:

        kp_e = (1 - gamma) * kp_center_mass + gamma * kp_r
        my_RK = Runge_Kutta(leftCoefs, rightCoefs, initState, inputFunc, step, 1.5 * regTime,
                            kp_e, None, None, Tr=regTime / 5)
        my_RK.solve()
        y = my_RK.e0_solve
        e_e = simpson(abs(np.array(y)), dx=step)

        if e_e < e_r:
            kp_list[4], func_list[4] = kp_e, e_e
            return kp_list, func_list
        else:
            kp_list[4], func_list[4] = kp_r, e_r
            return kp_list, func_list

    elif func_list[0] < e_r < func_list[3]:
        kp_list[4], func_list[4] = kp_r, e_r
        return kp_list, func_list

    elif func_list[3] < e_r < func_list[4]:
        kp_list[4], func_list[4] = kp_r, e_r

    kp_s = beta * kp_list[4] + (1 - beta) * kp_center_mass
    my_RK = Runge_Kutta(leftCoefs, rightCoefs, initState, inputFunc, step, 1.5 * regTime,
                        kp_s, None, None, Tr=regTime / 5)
    my_RK.solve()
    y = my_RK.e0_solve
    e_s = simpson(abs(np.array(y)), dx=step)

    if e_s < func_list[4]:
        kp_list[4], func_list[4] = kp_s, e_s
        return kp_list, func_list

    else:

        for k in range(4):

            if k == 3:
                continue
            else:
                kp_list[k] = (kp_list[3] + (kp_list[k] - kp_list[3])) / 2
                my_RK = Runge_Kutta(leftCoefs, rightCoefs, initState, inputFunc, step, 1.5 * regTime,
                                    kp_list[k], None, None, Tr=regTime / 5)
                my_RK.solve()
                y = my_RK.e0_solve
                e = simpson(abs(np.array(y)), dx=step)
                func_list[k] = e

        return kp_list, func_list


reg = np.array([[[4, 0.4625], [10, 500]], [], []], dtype=object)
params = [leftCoefs, rightCoefs, [0, 0, 0], 1, 0.001, regTime, regTime, 0.001, reg]
kp_list, func_list = Nelder_Mid(*params, alpha, beta, gamma)
epochs = int(input("Enter the amount of iterations: "))
print('\n')

for i in range(epochs):
    kp_list, func_list = Nedler_Mid(kp_list, func_list, *params[:-1], alpha, beta, gamma)

# print(sum(kp_list) / len(kp_list))
# print(func_list)
# print(kp_list)

# print(min(func_list))
# reg = (kp_list[func_list.index(min(func_list))])
reg = sum(kp_list) / len(kp_list)

my_RK = Runge_Kutta(leftCoefs, rightCoefs, initState, inputFunc, step, 1.5 * regTime,
                    reg, None, None, Tr=regTime / 5)
my_RK.solve()
my_RK.plot_solution()
error = simps(abs(np.asarray(my_RK.e0_solve)), step)
print("Error: ", error)
