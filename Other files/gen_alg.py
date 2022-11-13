from operator import index
import numpy as np
import random
import math

class gen_alg:
    def __init__(self, regsList, regStruct, JList, selection_ratio, mutation_ratio, target_function):
        self.inPopulation = regsList
        self.regStruct = regStruct
        self.selPopulation = list()
        self.JList = JList
        self.selection_ratio = selection_ratio
        self.mutation_ratio = mutation_ratio
        self.target_function = target_function

    def selection_v2(self):
        pList = [] # probability list
        sum_inv_coef = 0
        for i in range(len(self.JList)):
            sum_inv_coef += 1/abs(self.JList[i] - self.target_function)
        for i in range(len(self.JList)):
            pList.append((1 / abs(self.JList[i] - self.target_function)) / sum_inv_coef)
        
        i = range(len(self.JList))
        tmp = np.random.choice(a = i, size = math.floor(self.selection_ratio * len(self.JList)), 
            replace = False, p = pList)
        for i in tmp:
            self.selPopulation.append(self.inPopulation[i])

    def crossover_v2(self):
        outPopulation = []

        if len(self.selPopulation) == 1:
            return self.selPopulation

        tmpList = list()
        for i in range(len(self.selPopulation)):
            tmpReg = []
            for j in range(4):
                if self.selPopulation[i][0] is not None:
                    tmpReg.append(self.selPopulation[i][0][int(j/2)][j%2])
                if self.selPopulation[i][1] is not None:
                    tmpReg.append(self.selPopulation[i][1][int(j/2)][j%2])
                if self.selPopulation[i][2] is not None:
                    tmpReg.append(self.selPopulation[i][2][int(j/2)][j%2])
            tmpList.append(tmpReg)

        numOfRegs = math.floor(len(tmpList) / 2)
        for i in range(numOfRegs):
            fParent = random.choice(tmpList)
            # Slice parents at random point
            sliceIndex = random.randint(1, 3 * len(fParent) / 4)
            index = tmpList.index(fParent)
            del tmpList[index]

            sParent = random.choice(tmpList)
            index = tmpList.index(sParent)
            del tmpList[index]

            fChild = fParent[0:sliceIndex]
            fChild.extend(sParent[sliceIndex:])

            sChild = sParent[0:sliceIndex]
            sChild.extend(fParent[sliceIndex:])

            k = 0
            fReg = [    [[], []], 
                        [[], []],
                        [[], []]    ]
            sReg = [    [[], []], 
                        [[], []],
                        [[], []]    ]
            for j in range(4):
                if self.regStruct[0] == 1:
                    fReg[0][int(j/2)].append(fChild[k])
                    sReg[0][int(j/2)].append(sChild[k])
                    k += 1
                else:
                    fReg[0] = None
                    sReg[0] = None
                if self.regStruct[1] == 1:
                    fReg[1][int(j/2)].append(fChild[k])
                    sReg[1][int(j/2)].append(sChild[k])
                    k += 1
                else:
                    fReg[1] = None
                    sReg[1] = None
                if self.regStruct[2] == 1:
                    fReg[2][int(j/2)].append(fChild[k])
                    sReg[2][int(j/2)].append(sChild[k])
                    k += 1
                else:
                    fReg[2] = None
                    sReg[2] = None
            
            outPopulation.append(fReg)
            outPopulation.append(sReg)

        return outPopulation
    
    # remake this function
    def mutation_v2(self, population):
        in_population = population

        if len(self.sel_population) == 1:
            return population

        i = range(len(population))
        tmp = np.random.choice(a = i, size = math.floor(self.mutation_ratio * len(population)), replace = False)
        for i in tmp:
            in_population[i] = [[in_population[i][0][0] + 0.5 * (-1**(random.randint(1,3)))*np.random.random_sample(), in_population[i][0][1]],[in_population[i][1][0], in_population[i][1][1]]]

        return in_population