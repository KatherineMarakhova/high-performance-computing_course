# $$A = B^3 + C^2 \cdot E + B \cdot D $$ 
import numpy as np


def readmatrix(filename):
    with open(filename) as file:
        array = np.array([row.strip().split(' ') for row in file]).astype(float)
        return array
    
def check_answer(check, corr):
    for i in range(corr.shape[0]):
        for j in range(corr.shape[0]):
            if check[i][j] != corr[i][j]:
                
                print(corr)
                return 0
    print('Correct answer!')
        

B = readmatrix("B.txt")  
C = readmatrix("C.txt")
D = readmatrix("D.txt")
E = readmatrix("E.txt")


B3 = B.dot((B.dot(B)))
C2 = C.dot(C)
C2E = C2.dot(E)
BD = B.dot(D)
corr_result = (B3+C2E+BD).round(2)                  #правильное решение, округленное
check_result = readmatrix("result.txt").round(2)    #считываем матрицу параллельного алгоритма из файла, округляем

check_answer(check_result, corr_result)
