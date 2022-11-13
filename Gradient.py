import matplotlib.pyplot as plt
from RK import *
import numpy as np
from numpy import trapz
from scipy.integrate import simpson

leftCoefs = [0.0000063, 0.0071, 0.12, 0]
rightCoefs = [2]
initState = [0, 0, 0]
inputFunc = 1
step = 0.001
regTime = 2
# reg = [[[0.9814968953533798, 0.8735947051097981], [1.3020361281861965, 1.9970448053536267]], None, [[0.24236802398950513, 0.002761137278375356], [2.543671286897902, 0.0029551946463733884]]]
#regP = np.array([[4, 0.4625], [10, 500]], dtype=object)
regP = np.array([[2.674475389904081, 5.581632451291929],
 [20.01496883245269, 449.080797674991]], dtype=object)
       # [20.049337763193474, 444.45495792243815]], dtype=object)

my_RK = Runge_Kutta(leftCoefs, rightCoefs, initState, inputFunc, step, 1.5 * regTime,
                   regP, None, None, Tr = regTime / 5)
my_RK.solve()
my_RK.plot_solution()

# plt.plot(np.arange(0, 3, step), my_RK.e0_solve,)
# plt.grid()
# plt.show()
#plt.plot(np.arange(0, 3, step), my_RK.e1_solve,)


# y = my_RK.e0_solve
# area_s = simpson(y, dx=step)
# ? = trapz(y, dx=step)
# print(area_s)

def simps(y, step):                 # функция подсчет интеграла ошибки
    return simpson(y, dx=step)


def gradient(kp_1, kp_2, e_1, e_2):     # функция подсчета градиента
    # на вход подаются: kp-1 - набор точек 1-ого нелинейного элемента, kp_2 - набор точек 2-ого нелинейного элемента
    # e_1 - интеграл ошибки, полученный при использовании 1-ого нелинейного элемента
    # e_2 - интеграл ошибки, полученный при использовании 2-ого нелинейного элемента
    gradient_vector = np.zeros((2, 2))
    for j in range(2):
        for i in range(2):
            a = (kp_1[j][i] * (e_1 - e_2)) / (kp_1[j][i] - kp_2[j][i])
            gradient_vector[i][j] = a
    print('gradient_vector: ', gradient_vector, '\n')
    print('kp_1', kp_1, '\n')
    return gradient_vector


def gradient_decent(kp_1, kp_2, lr, epochs, leftCoefs, rightCoefs, initState, inputFunc, step, regTime, Tr):
    # функция поиска оптимальных значений для нелинейного элемента
    # возвращает: набор точек для оптимального нелинейного элемента и интеграл ошибки
    for i in range(epochs):
        my_RK_1 = Runge_Kutta(leftCoefs, rightCoefs, initState, inputFunc, step, 1.5 * regTime,
                              kp_1, None, None, Tr=regTime / 5)
        my_RK_2 = Runge_Kutta(leftCoefs, rightCoefs, initState, inputFunc, step, 1.5 * regTime,
                              kp_2, None, None, Tr=regTime / 5)

        my_RK_1.solve()
        my_RK_2.solve()
        e_1 = simps(abs(np.asarray(my_RK_1.e0_solve)), step)        # получаем значения ошибки
        e_2 = simps(abs(np.asarray(my_RK_2.e0_solve)), step)        # получаем значения ошибки

        gradient_vector = np.asarray(gradient(kp_1, kp_2, e_1, e_2)) # Поучили вектор производных по каждой компоненте

        kp_1 = kp_2   #  Перезаписываем K_p1  на K_p2
        kp_2 = kp_2 - lr * gradient_vector  # Корректируем K_p2 (условно получил kp_3)
        print('kp_2:', kp_1)
        print('kp_3', kp_2, '\n')

        my_RK_3 = Runge_Kutta(leftCoefs, rightCoefs, initState, inputFunc, step, 1.5 * regTime,
                              kp_2, None, None, Tr=regTime / 5)
        my_RK_3.solve()
        e = simps(abs(np.asarray(my_RK_3.e0_solve)), step)
        print('error ', e, '\n', 'step = ', i + 1, '\n')  # Пишем ошибку для получившегося вектора
        print('-' * 10)

    return kp_2, e 


epochs = 10
kp_1 = np.array([[4, 0.4625], [10, 500]], dtype=object)     # набор точек 1-ого нелинейного элемента
kp_2 = np.array([[3, 5], [20, 450]], dtype=object)          # набор точек 2-ого нелинейного элемента
lr = 1                                                      # шаг сходимости

print(gradient_decent(kp_1, kp_2, lr, epochs, leftCoefs, rightCoefs, initState, inputFunc, step, 1.5 * regTime, regTime / 5))