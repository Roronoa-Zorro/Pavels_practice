import numpy as np
class fuzzy_reg_K:
    def __init__(self,points, e):

        self.e = e
        self.points = points
        self.K1 = 0
        self.K2 = 0

    def calc_out(self):
        U = 0

        for i in range(len(self.points)):
            if abs(self.e) <= self.points[i][0]:
                if i == 0:
                    K1 = self.points[0][1] / self.points[0][0]
                    U = abs(self.e) * K1
                else:
                    K = (self.points[i][1] - self.points[i-1][1])/(self.points[i][0] - self.points[i-1][0])
                    U = K*(abs(self.e) - self.points[i-1][0]) + self.points[i-1][1]
                
                if self.e < 0:
                    return -U
                return U
        if self.e < 0:
                return -self.points[-1][-1]       
        return self.points[-1][-1]
                        
'''
        index = self.define_e_pos() 
        if self.points[index][0] == self.points[0][0]:
            K1 = self.points[0][1] / self.points[0][0]
            U = self.e * K1
        elif self.points[index][0] == self.points[-1][0]:
            U = self.points[-1][-1]
        else:
            K = (self.points[index][1] - self.points[index-1][1])/(self.points[index][0] - self.points[index-1][0])
            U = K*(self.e - self.points[index-1][0]) + self.points[index-1][1]
        return U
'''
'''
arr = np.random.rand(1,2)
print(arr)
fuz = fuzzy_reg_K(arr,  e = 0.6)
print(fuz.define_e_pos())
print(fuz.define_K())
'''
