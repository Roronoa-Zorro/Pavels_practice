import random
from scipy import signal
import numpy as np

dList = [0, 0, 0]
cList = [0, 0, 0]
bList = [0, 0, 0]
aList = [0, 0, 0]
# Make 3 d values for nonlinear characteristic
# P channel gets the most value
# if plant is static, then integral part must have mininal value that will make zero error
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

def abs_stability_criteria(self, regStructList, reg, Hp, Hi, Hd, tolerance):
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