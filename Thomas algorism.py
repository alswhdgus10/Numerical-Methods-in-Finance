# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 21:09:48 2018

@author: MinMax

수치해석학 과목의 목표는 ELS 상품의 PRICING 및 델타, 감마 헷지
목표를 달성하기 위한 과정으로 알고리즘을 익혀야 한다.
그 중 하나인 system of linear equations을 풀 수 있도록 접근. \
matrix A를 U(upper)와 L(lower)의 곱으로 나타내어,
Ax = U*L*x = U*y = d  (L*x = y)
이 중에서 특별한 matrix인 tridiagonal matrix를 통해 Thomas algorithm 을  만들고자 한다.
수식은 lecture note2참고 (p13)
a'_n = (a_n - b_n * (c_n-1/a'_n-1))
d'_n = (d_n - b_n * (d'_n-1/a'_n-1))

Convergence 확인 하는 방법
|a_ii| > sum|(a_iJ)| for all i 이면 strictly diagonally dominant한다.
따라서 tridiagonal matrix가 strictly diagonally dominant하다면 이는 Thomas algorism을 이용하여 항상 풀 수 있다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import scipy as si

#%% 
#inverse matrix를 이용하여 구하는 방법
def thomas_al2(diagonal,l_diagonal,u_diagonal,d): return np.linalg.solve(np.matrix(np.diag(diagonal)+np.diag(l_diagonal,k=-1)+np.diag(u_diagonal,k=1)),d).round(3)

emp = []
for alpha in np.arange(0.5,5.1,0.1):
    a = np.ones(10) * (1+alpha) #diagonal
    b = np.ones(9) * (-alpha/2) #lower diagonal
    c = np.ones(9) * (-alpha/2) #upper diagonal
    d = np.zeros(10)
    d[0] =0
    d[-1] = 1
    d = d*alpha/2
    z = thomas_al2(a,b,c,d)
    emp.append(z)
solution_inv = pd.DataFrame(emp, index = np.arange(0.5,5.1,0.1).round(1), columns = [ 'x1','x2','x3','x4','x5','x6','x7','x8','x9','x10'])
print ("using inverse matrix (d=1)")

print ("="*30)

emp = []
for alpha in np.arange(0.5,5.1,0.1):
    a = np.ones(10) * (1+alpha) #diagonal
    b = np.ones(9) * (-alpha/2) #lower diagonal
    c = np.ones(9) * (-alpha/2) #upper diagonal
    d = np.zeros(10)
    d[0] =0
    d[-1] = 5
    d = d*alpha/2
    z = thomas_al2(a,b,c,d)
    emp.append(z)    
solution_inv2 = pd.DataFrame(emp, index = np.arange(0.5,5.1,0.1).round(1),columns = [ 'x1','x2','x3','x4','x5','x6','x7','x8','x9','x10'])
print ("using inverse matrix (d=5)")
print (solution_inv2)
print ("="*30)

#%%
#Thomas algorism을 통해 구한 방법
alpha = 0.5
diagonal = np.ones(10) * (1+alpha)
l_diagonal = np.ones(10) * (-alpha/2)
u_diagonal = np.ones(10) * (-alpha/2)
d = np.array([0,0,0,0,0,0,0,0,0,1]) * (alpha/2)


def cal(diagonal,l_diagonal,u_diagonal,d):
    a1 =[diagonal[0]]
    d1 =[d[0]]
    x = []
    for i in range(len(diagonal)-1):
        a_prime = diagonal[i+1] - (l_diagonal[i+1] * u_diagonal[i]/a1[i])
        d_prime = d[i+1] - (l_diagonal[i+1] * d1[i]/a1[i])
        a1.append(a_prime)
        d1.append(d_prime)
    
    x.append(d1[-1]/a1[-1])
    
    for i in range(9,0,-1):
        x.append((-u_diagonal[i]/a1[i])*x[9-i])
    x.reverse()
    x = [x]
    x= pd.DataFrame(x, columns = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']).round(3)
    return x
x = cal(diagonal, l_diagonal, u_diagonal, d)


solution_thomas = pd.DataFrame()
for alpha in np.arange(0.5,5.1,0.1):
    diagonal = np.ones(10) * (1+alpha)
    l_diagonal = np.ones(10) * (-alpha/2)
    u_diagonal = np.ones(10) * (-alpha/2)
    d = np.array([0,0,0,0,0,0,0,0,0,1]) * (alpha/2)
    solution_thomas = solution_thomas.append(cal(diagonal, l_diagonal, u_diagonal,d))
solution_thomas.index = np.arange(0.5,5.1,0.1).round(2)

solution_thomas2 = pd.DataFrame()
for alpha in np.arange(0.5,5.1,0.1):
    diagonal = np.ones(10) * (1+alpha)
    l_diagonal = np.ones(10) * (-alpha/2)
    u_diagonal = np.ones(10) * (-alpha/2)
    d = np.array([0,0,0,0,0,0,0,0,0,5]) * (alpha/2)
    solution_thomas2 = solution_thomas2.append(cal(diagonal, l_diagonal, u_diagonal,d))
solution_thomas2.index = np.arange(0.5,5.1,0.1).round(2)
