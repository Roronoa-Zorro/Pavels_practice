def synth_reg(self, input_func_str, left_coefs_str, right_coefs_str, 
                    init_state_str, regStructList, step_str, satLevel_str, 
                    regTime_str, pplSize_str, message):
        # Check parametres for valid input
        res, leftCoefs = self.str_pattern_conv(left_coefs_str)
        if res == -1:
            message.value = "Ошибка формата параметров левой части ОУ"
            return -1
        res, rightCoefs = self.str_pattern_conv(right_coefs_str)
        if res == -1:
            message.value = "Ошибка формата параметров правой части ОУ"
            return -1
        res, initState = self.str_pattern_conv(init_state_str)
        if res == -1:
            message.value = "Ошибка формата начальных условий"
            return -1
        try:
            step = float(step_str)
        except ValueError:
            message.value = "Ошибка формата шага интегрирования"
            return -1
        try:
            satLevel = float(satLevel_str)
        except ValueError:
            message.value = "Ошибка формата уровня насыщения"
            return -1
        try:
            regTime = float(regTime_str)
        except ValueError:
            message.value = "Ошибка формата времени регулирования"
            return -1
        timeConst = regTime / 5
        try:
            pplSize = float(pplSize_str)
        except ValueError:
            message.value = "Ошибка формата размера популяции"
            return -1
        try:
            inputFunc = float(input_func_str)
        except ValueError:
            message.value = "Поддерживается только постоянный вход"
            return -1
        message.value = ""

        # Prepare data 
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
            
        iter = 0
        regsList = []
        bar = Bar('Processing', max=pplSize)
        while iter != pplSize:
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

            '''
            for i in range(len(regStructList)):
                if regStructList[i] == 1:
                    dList[i] = random.random()
            '''

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

            res = self.abs_stability_criteria(regStructList, tmpReg, Hp, Hi, Hd, 0.001)
            if res == -1:
                continue

            message.value = "Прогресс создания начальной популяции: " + str(int(float(iter/pplSize) * 100)) + "%"
            self.r_k_calculator_window.update()

            iter += 1
            bar.next()

            regsList.append(tmpReg)
        bar.finish()

        message.value = "Начальная популяция сформирована. Запущен генетический алгоритм"
        self.r_k_calculator_window.update()

        print(regsList[0])
        print(initState)
        synthedReg, fullRegsnJDict = self.genetic_algorithm(leftCoefs, rightCoefs, initState, 
            inputFunc, step, regTime, regsList, regStructList)
        print("Synthed reg = ", synthedReg)

        res = self.abs_stability_criteria(regStructList, synthedReg, Hp, Hi, Hd, 0.001)
        if res == -1:
            print("Check for absolute stability of final regulator FAILED")
            print('Trying to find the most suitable regulator')
            for i in range(len(fullRegsnJDict)):
                keysList = list(fullRegsnJDict.keys())
                minJDelta = min(keysList)
                res = self.abs_stability_criteria(regStructList, fullRegsnJDict[minJDelta], Hp, Hi, Hd, 0.001)
                if res == -1:
                    del fullRegsnJDict[minJDelta]
                else:
                    print("Absolute stable reg found! Reg params: ", fullRegsnJDict[minJDelta], 
                        " delta J = ", minJDelta)
                    my_RK = Runge_Kutta(leftCoefs, rightCoefs, initState, 
                        inputFunc, step, 1.5 * regTime, fullRegsnJDict[minJDelta][0], 
                        fullRegsnJDict[minJDelta][1], fullRegsnJDict[minJDelta][2], 
                        Tr = regTime / 5)
                    my_RK.solve()
                    my_RK.plot_solution()
                    synthedReg = fullRegsnJDict[minJDelta]
                    break
        else:
            # plot trasient response with final regulator
            print("Check for absolute stability of final regulator SUCCEED")
        print("Finished")
        print("Synthed reg = ", synthedReg)