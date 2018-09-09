# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 21:09:48 2018

@author: MinMax
날짜: 2018.09.09
과목: 수치해석학

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

def thomas_al2(diagonal,l_diagonal,u_diagonal,d): return np.linalg.solve(np.matrix(np.diag(diagonal)+np.diag(l_diagonal,k=-1)+np.diag(u_diagonal,k=1)),d).round(3)

emp = []  #appending 위해서 만든 list
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
sol_3 = pd.DataFrame(emp, index = np.arange(0.5,5.1,0.1).round(1), columns = [ 'x1','x2','x3','x4','x5','x6','x7','x8','x9','x10'])
print ("using inverse matrix (d=1)")
print (sol_3)
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
sol_4 = pd.DataFrame(emp, index = np.arange(0.5,5.1,0.1).round(1),columns = [ 'x1','x2','x3','x4','x5','x6','x7','x8','x9','x10'])
print ("using inverse matrix (d=5)")
print (sol_4)
print ("="*30)
def cal(a,b,c,d):
    a1 =[a[0]]
    d1 =[d[0]] 
    for i in range(len(c)):
        x = a[i+1] - (b[i+1] * c[i]/a1[i])
        y = d[i+1] - (b[i+1] * d1[i]/a1[i])
        a1.append(x)
        d1.append(y)
    return a1, d1

##Thomas algorism에서 a prime 과 d prime이 이떻게 되는지 계산 해봄
emp_1 = []
emp_2 = []
emp_3 = []
for alpha in np.arange(0.5,5.1,0.1):
    a= np.ones(10) * (1+alpha)
    b = np.ones(10) * (-alpha/2)
    b[0] = 0
    c = np.ones(9) * (-alpha/2)
    d = np.zeros(10)
    d[0] =0
    d[-1] = 1
    d = d*alpha/2
    a1, d1 = cal(a,b,c,d)
    emp_1.append(a1)
    emp_2.append(d1)
    emp_3.append(c)

# Thomas Algorithm d=1
calculation_a = pd.DataFrame(emp_1, index = np.arange(0.5,5.1,0.1).round(1), columns= ['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10'])
calculation_d = pd.DataFrame(emp_2, index = np.arange(0.5,5.1,0.1).round(1), columns= ['d1','d2','d3','d4','d5','d6','d7','d8','d9','d10'])
calculation_d5 = calculation_d * 5
calculation_c = pd.DataFrame(emp_3, index = np.arange(0.5,5.1,0.1).round(1), columns= ['c1','c2','c3','c4','c5','c6','c7','c8','c9'])
d_result = pd.DataFrame(index = np.arange(0.5,5.1,0.1).round(1), columns= ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10'])

d10 = calculation_d['d10']/calculation_a['a10']
d_result['x10'] = d10

for i in range(9,0,-1):
    if i ==9:
        d_result['x'+str(i)]=(calculation_d['d'+str(i)]/calculation_a['a'+str(i)])-(calculation_c['c'+str(i)]/calculation_a['a'+str(i)])*d10
    else:
        d_result['x'+str(i)]=(calculation_d['d'+str(i)]/calculation_a['a'+str(i)])-(calculation_c['c'+str(i)]/calculation_a['a'+str(i)])*d_result['x'+str(i+1)]

print ("using thomas algorithm (d=1)")
print (d_result.round(3))

# Thomas Algorithm d=5
d10 = calculation_d5['d10']/calculation_a['a10']
d_result['x10'] = d10

for i in range(9,0,-1):
    if i ==9:
        d_result['x'+str(i)]=(calculation_d5['d'+str(i)]/calculation_a['a'+str(i)])-(calculation_c['c'+str(i)]/calculation_a['a'+str(i)])*d10
    else:
        d_result['x'+str(i)]=(calculation_d5['d'+str(i)]/calculation_a['a'+str(i)])-(calculation_c['c'+str(i)]/calculation_a['a'+str(i)])*d_result['x'+str(i+1)]

print ("using thomas algorithm (d=5)")
print (d_result.round(3))