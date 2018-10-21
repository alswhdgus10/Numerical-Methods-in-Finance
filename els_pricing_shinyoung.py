# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 19:03:56 2018
신영 ELS
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy

##기본 파라미터##
S1max = 7000;
S1min = 0;
S2max = 6000;
S2min = 0;  # Max and min price of the underlying asset 2
mu1 = 0.1;  # mean return of the underlying asset 1
mu2 = 0.1;  # mean return of the underlying asset 2
sig1 = 0.25;  # Volatility of the underlying asset 1
sig2 = 0.2;  # Volatility of the underlying asset 2
q1 = 0
q2 = 0
rho = 0.4;  # Correlation between prices of the two assets
r = 0.02;  # Interest rate
K0 = [3500, 3000];  # Reference price of each asset
F = 100;  # Face value
T = 3;  # Maturation of contract
c = [0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375];  # Rate of return on each early redemption date
K = [0.85, 0.80, 0.75, 0.70, 0.70, 0.70];  # Exercise price on each early redemption date
KI = 0.60;  # Knock-In barrier level
n_Steps = 300;
Nx = 100;
Ny = 100;
dt = T / n_Steps;
hx = (S1max - S1min) / Nx;
hy = (S2max - S2min) / Ny;


# Thomas algorism

def TDMAsolver(alpha, beta, gamma, f):
    a = list(alpha)
    b = list(beta)
    c = list(gamma)
    d = list(f)

    nf = len(d)  # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d))  # copy arrays
    for it in range(1, nf):
        mc = ac[it - 1] / bc[it - 1]  # mc = alpha)(n) / beta_prime_n-1
        bc[it] = bc[it] - mc * cc[it - 1]
        dc[it] = dc[it] - mc * dc[it - 1]
        xc = copy.deepcopy(bc)
        xc[-1] = dc[-1] / bc[-1]

    for il in range(nf - 2, -1, -1):
        xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]
    return xc


# %%
# 기본값 설정.
# x,y 그리드 설정
# 100개로 짤랐으므로 노드는 101개
x = np.linspace(S1min, S1max, Nx + 1)
y = np.linspace(S2min, S2max, Ny + 1)
# %%
# 가격 매트릭스 형성 및 alpha, beta, gamma 기본값 설정
u = np.zeros((Nx + 1, Ny + 1))
u2 = np.zeros((Nx + 1, Ny + 1))
k = np.zeros((Nx + 1, Ny + 1))
k2 = np.zeros((Nx + 1, Ny + 1))

# %%
# initial condition 설정
# 1번 위치
# no hit
for i in range(Nx + 1):
    for j in range(Ny + 1):
        if (x[i] >= K0[0] * K[-1] and y[j] >= K0[1] * K[-1]):  # 1번 케이스 두 자산 모두 85프로 이상일 때
            u[i, j] = F * (1 + c[-1])

        elif (x[i] > K0[0] * KI and y[j] >= K0[1] * KI) and (x[i] < K[-1] * K0[0] or y[j] < K0[1] * K[-1]):
            u[i, j] = F * (1 + c[-1])

        elif min(x[i] / K0[0], y[j] / K0[1]) < KI:  # 3번 케이스. KI 베리어 쳤을 때
            u[i, j] = F * min(x[i] / K0[0], y[j] / K0[1])
# %%
for i in range(Nx + 1):
    for j in range(Ny + 1):
        if (x[i] >= K0[0] * K[-1] and y[j] >= K0[1] * K[-1]):  # 1번 케이스 두 자산 모두 85프로 이상일 때
            k[i, j] = F * (1 + c[-1])

        elif (x[i] > K0[0] * KI and y[j] >= K0[1] * KI) and (x[i] < K[-1] * K0[0] or y[j] < K0[1] * K[-1]):
            k[i, j] = F * min(x[i] / K0[0], y[j] / K0[1])

        elif min(x[i] / K0[0], y[j] / K0[1]) < KI:  # 3번 케이스. KI 베리어 쳤을 때
            k[i, j] = F * min(x[i] / K0[0], y[j] / K0[1])


# %%
def OSMalgorism(u, u2, case):
    for m in range(1, n_Steps + 1):
        # 1ST STEP#
        for j in range(1, Ny):
            # 99개를 만듬
            alpha_x = np.zeros(Nx - 1);
            beta_x = np.zeros(Nx - 1);
            gamma_x = np.zeros(Nx - 1);
            fy = np.zeros(Ny - 1);
            for i in range(1, Nx):
                # 논문에 있는 수식대로 입력
                beta_x[i - 1] = 1 / dt + np.power(sig1 * x[i], 2) / hx ** 2 + r * x[i] / hx + 0.5 * r
                alpha_x[i - 1] = -0.5 * np.power(sig1 * x[i], 2) / hx ** 2
                gamma_x[i - 1] = -0.5 * np.power(sig1 * x[i], 2) / hx ** 2 - r * x[i] / hx
                
                if i == 99:
                    fy[i - 1] = 0.5 * rho * sig1 * sig2 * x[i] * y[j] \
                                * (2 * u[i, j + 1] - u[i - 1, j + 1] - (2 * u[i, j] - u[i - 1, j]) - u[i, j + 1] + u[
                        i, j]) / (hx ** 2) + u[i, j] / dt;
                else:
                    fy[i - 1] = 0.125 * rho * sig1 * sig2 * x[i] * y[j] \
                                * (u[i + 1, j + 1] - u[i + 1, j] - u[i, j + 1] + u[i, j]) / (hx ** 2) + u[i, j] / dt;

            beta_x[0] = beta_x[0] + 2.0 * alpha_x[0];
            gamma_x[0] = gamma_x[0] - alpha_x[0];
            alpha_x[-1] = alpha_x[-1] - gamma_x[-1];
            beta_x[-1] = beta_x[-1] + 2.0 * gamma_x[-1];

            u2[1:Nx, j] = TDMAsolver(alpha_x[1:], beta_x, gamma_x[:-1], fy);
            # 1부터 Nx-1까지 넣어야 하므로, 1:Nx로 인덱싱
        u2[0, 1:Ny] = 2 * u2[1, 1:Ny] - u2[2, 1:Ny];  # 첫행 바운더리
        u2[Nx, 1:Ny] = 2 * u2[Nx - 1, 1:Ny] - u2[Nx - 2, 1:Ny];  # 마지막행 바운더리인데 필요한가? i==100일 때 위에서 했는데
        u2[1:Nx, 0] = 2 * u2[1:Nx, 1] - u2[1:Nx, 2];  # 첫 열 바운더리
        u2[1:Nx, Ny] = 2 * u2[1:Nx, Ny - 1] - u2[1:Nx, Ny - 2];  # 마지막열 바운더리

        # 2ND STEP#
        for i in range(1, Nx):
            alpha_y = np.zeros(Ny - 1);
            beta_y = np.zeros(Ny - 1);
            gamma_y = np.zeros(Ny - 1);
            fx = np.zeros(Nx - 1);
            for j in range(1, Ny):
                beta_y[j - 1] = 1 / dt + np.power(sig2 * y[j], 2) / hy ** 2 + r * y[j] / hy + 0.5 * r
                alpha_y[j - 1] = -0.5 * np.power(sig2 * y[j], 2) / hy ** 2
                gamma_y[j - 1] = -0.5 * np.power(sig2 * y[j], 2) / hy ** 2 - r * y[j] / hy
                if j == 99:
                    fx[j - 1] = 0.5 * rho * sig1 * sig2 * x[i] * y[j] \
                                * (2 * u2[i + 1, j] - u2[i + 1, j - 1] - u2[i + 1, j] - (2 * u2[i, j] - u2[i, j - 1]) +
                                   u2[i, j]) / (hy ** 2) + u2[i, j] / dt;
                else:
                    fx[j - 1] = 0.125 * rho * sig1 * sig2 * x[i] * y[j] \
                                * (u2[i + 1, j + 1] - u2[i + 1, j] - u2[i, j + 1] + u2[i, j]) / (hy ** 2) + u2[
                                    i, j] / dt;

            beta_y[0] = beta_y[0] + 2.0 * alpha_y[0];
            gamma_y[0] = gamma_y[0] - alpha_y[0];
            alpha_y[-1] = alpha_y[-1] - gamma_y[-1];
            beta_y[-1] = beta_y[-1] + 2.0 * gamma_y[-1];

            u[i, 1:Ny] = TDMAsolver(alpha_y[1:], beta_y, gamma_y[:-1], fx);

        u[0, 1:Ny] = 2 * u[1, 1:Ny] - u[2, 1:Ny];  # 첫행 바운더리
        u[Nx, 1:Ny] = 2 * u[Nx - 1, 1:Ny] - u[Nx - 2, 1:Ny];  # 마지막행 바운더리인데 필요한가? i==99일 때 위에서 했는데
        u[1:Nx, 0] = 2 * u[1:Nx, 1] - u[1:Nx, 2];  # 첫 열 바운더리
        u[1:Nx, Ny] = 2 * u[1:Nx, Ny - 1] - u[1:Nx, Ny - 2];  # 마지막열 바운더리

        # No hit 일때, 조기상환 처리
        if case == 'no':
            if m == 50:  # 잔존 만기가 6개월 남았을 때, 즉 2.5년 시점에서
                for i in range(Nx + 1):
                    for j in range(Ny + 1):
                        if min(x[i] / K0[0], y[j] / K0[1]) > K[4]:
                            u   [i, j] = (1 + c[4]) * F
            if m == 100:  # 잔존 만기가 1년 남았을 때, 2년 시점에서
                for i in range(Nx + 1):
                    for j in range(Ny + 1):
                        if min(x[i] / K0[0], y[j] / K0[1]) > K[3]:
                            u[i, j] = (1 + c[3]) * F
            if m == 150:  # 잔존만기 1.5년 남았을 떄 1.5년 시점에서
                for i in range(Nx + 1):
                    for j in range(Ny + 1):
                        if min(x[i] / K0[0], y[j] / K0[1]) > K[2]:
                            u[i, j] = (1 + c[2]) * F
            if m == 200:  # 잔존 만기가 2년 남았을 떄/ 1년 시점에서
                for i in range(Nx + 1):
                    for j in range(Ny + 1):
                        if min(x[i] / K0[0], y[j] / K0[1]) > K[1]:
                            u[i, j] = (1 + c[1]) * F
            if m == 250:  # 잔존만기가 6개월 남았을 떄. 6개월 시점에서
                for i in range(Nx + 1):
                    for j in range(Ny + 1):
                        if min(x[i] / K0[0], y[j] / K0[1]) > K[0]:
                            u[i, j] = (1 + c[0]) * F
    return u;


# %%
u = OSMalgorism(u, u2, 'no')
k = OSMalgorism(k, k2, 'knock-in')
p = 0.132  # 시뮬레이션에 의한 확률 1000번 10번씩 돌려서 나온 확률의 평균
tt = u * (1 - p) + k * p
# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
[x, y] = np.meshgrid(x, y)
ax.plot_surface(x, y, u)
plt.show()
# f2 = figure;
# mesh(x, y, u);
fdm_price = tt[50, 50]
print ("FDM price : ",fdm_price)