import matplotlib.pyplot as plt
from RK import *
import numpy as np
from numpy import trapz
from scipy.integrate import simpson
from scipy import signal

leftCoefs = [0.0000063, 0.0071, 0.12, 0]
rightCoefs = [2]
initState = [0, 0, 0]
inputFunc = 1
step = 0.001
regTime = 2
reg = [[[0.9814968953533798, 0.8735947051097981], [1.3020361281861965, 1.9970448053536267]], None, [[0.24236802398950513, 0.002761137278375356], [2.543671286897902, 0.0029551946463733884]]]
regStructList = [1, 0, 0]

#my_RK = Runge_Kutta(leftCoefs, rightCoefs, initState, inputFunc, step, 1.5 * regTime,regP, None, None, Tr = regTime / 5)
#my_RK.solve()
#my_RK.plot_solution()




def abs_stability_criteria( regStructList, reg, Hp, Hi, Hd, tolerance):
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
                        [KdMax * Hd.real[i], KdMax * Hd.real[i], 1 + KdMax * Hd.real[i]]    ])
        res = np.linalg.det(tmp)
        if abs(res) <= tolerance:
            return -1
    return 0





def func(leftCoefs, rightCoefs, initState, inputFunc, step, regTime, reg1, reg2, Tr, step_grad, lr, epochs, tolerance):
    
    satLevel = reg2[0][1][1]
    plantTF = signal.TransferFunction(rightCoefs, leftCoefs)
    wp, Hp = signal.freqresp(plantTF)
    tmp = leftCoefs[:]
    tmp.extend([0,])
    plantIntgrTF = signal.TransferFunction(rightCoefs, tmp)
    wi, Hi = signal.freqresp(plantIntgrTF)
    tmp = rightCoefs[:]
    tmp.extend([0,])
    plantDrvtTF = signal.TransferFunction(tmp, leftCoefs)
    wd, Hd = signal.freqresp(plantDrvtTF)
        
    
    regsList = []
    
    dList = [0, 0, 0]
    cList = [0, 0, 0]
    bList = [0, 0, 0]
    aList = [0, 0, 0]
    
    if leftCoefs[-1] != 0:
        if regStructList[1] == 1:
            dList[1] = random.uniform(inputFunc * leftCoefs[-1] / (satLevel * rightCoefs[-1]), 2 * inputFunc * leftCoefs[-1] / (satLevel * rightCoefs[-1]))
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
    
    res = abs_stability_criteria(regStructList, reg2, Hp, Hi, Hd, 0.001)
    
    return res



def simps(y, step):                 
    return simpson(y, dx=step)


def gradient(kp_1, kp_2, e_1, e_2):    

    gradient_vector = np.zeros((2, 2))
    for j in range(2):
        for i in range(2):
            a = (kp_1[j][i] * (e_1 - e_2)) / (kp_1[j][i] - kp_2[j][i])
            gradient_vector[i][j] = a
    return gradient_vector

def retrain():
    
    s = input('Do you like this plot?: [y/n] ')
    if s == 'n':
        t = int(input('How many steps: '))
        return t
    
    else:
        return 0
print('\n','\n','\n')

def gradient_decent(leftCoefs, rightCoefs, initState, inputFunc, step, regTime, reg1, reg2, Tr, step_grad, Lr, epochs, tolerance,params):

    print('-' * 50)
    
    kp_1 = reg1[0]
    kp_2 = reg2[0]
    lr = Lr
#    print(epochs)
    
    for i in range(int(epochs)):
        
        
        my_RK_1 = Runge_Kutta(leftCoefs, rightCoefs, initState, inputFunc, step, 1.5 * regTime,
                              kp_1, None, None, Tr=regTime / 5)
        my_RK_2 = Runge_Kutta(leftCoefs, rightCoefs, initState, inputFunc, step, 1.5 * regTime,
                              kp_2, None, None, Tr=regTime / 5)

        my_RK_1.solve()
        my_RK_2.solve()
        e_1 = simps(abs(np.asarray(my_RK_1.e0_solve)), step)        # получаем значения ошибки
        e_2 = simps(abs(np.asarray(my_RK_2.e0_solve)), step)        # получаем значения ошибки

        gradient_vector = np.asarray(gradient(kp_1, kp_2, e_1, e_2)) # Поучили вектор производных по каждой компоненте
        
        kp_3 = kp_2 - lr*gradient_vector
        reg2[0] = kp_2 - lr*gradient_vector
        
        if func(leftCoefs, rightCoefs, initState, inputFunc, step, regTime, reg1, reg2, Tr, step_grad, lr, epochs, tolerance) == 0:

            
            
            kp_1 = kp_2
            reg1[0] = kp_1
            kp_2 = kp_2 - lr * gradient_vector
            reg2[0] = kp_2
            lr = Lr
            
            print('absolutely stable:',func(leftCoefs, rightCoefs, initState, inputFunc, step, regTime, reg1, reg2, Tr, step_grad, lr, epochs, tolerance))
            
            
            print('kp_1:', kp_1, '\n')
            print('kp_2', kp_2, '\n')

            my_RK_3 = Runge_Kutta(leftCoefs, rightCoefs, initState, inputFunc, step, 1.5 * regTime,
                                  kp_2, None, None, Tr=regTime / 5)
            my_RK_3.solve()
            e = simps(abs(np.asarray(my_RK_3.e0_solve)), step)
            print('error ', e, '\n', 'step = ', i + 1, '\n')  # Пишем ошибку для получившегося вектора
            print('-' * 50)
            
            
        else:
            lr-=step_grad
            reg2[0] = kp_2
            print('absolutely unstable')
            print('step:', i+1,'\n')
            print('-'*50)
        
    my_RK_3 = Runge_Kutta(leftCoefs, rightCoefs, initState, inputFunc, step, 1.5 * regTime,
                          kp_2, None, None, Tr=regTime / 5)
    my_RK_3.solve()            
    my_RK_3.plot_solution()   
    e = simps(abs(np.asarray(my_RK_3.e0_solve)), step)
    print(reg1)
    print(reg2)
    print(e)
    
    temp = retrain()
    if temp != 0 and temp > 0:
    
        params[11] = temp
        params[6] = reg1
        params[7] = reg2        
        gradient_decent(*params, params)
       
    else:
        
        return (reg2, reg1, e)
        

reg2 = np.array([[[4, 0.4625], [10, 500]], [], []], dtype=object)
reg1 = np.array([[[12, 1], [41, 325]], [], []], dtype=object)

#reg2 = [[[4, 0.4625], [10, 500]], [], []]

params =  [leftCoefs, rightCoefs, [0,0,0], 1, 0.001, regTime*1.5, reg1, reg2, regTime / 5, 0.01, 0.1, 1, 0.001,]

print()
print()
a, b, c = gradient_decent(*params, params)
print('reg2: ',a)
print('-' * 50)    
print('reg1: ',b)
print('-' * 50)
print('e: ', c)





    
    
    
    