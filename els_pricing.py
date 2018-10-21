# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

Max = [7000, 6000]  # Underlying 1,2 의 Max Price
Min = [0, 0]  # Underlying 1,2 의 Min Price
rho = 0.4  # Correlation between two underlying
r = 0.022 # Interest Rate
K0 = [3500,3000]  # Reference price of each Underlying
F = 100  # Face value
T = 3 # Maturity of ELS
Nx = 100  # number of X node
Ny = 100  # number of Y node
c = [0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375] # Return of each early redemption
K = [0.85, 0.80, 0.75, 0.70, 0.70, 0.70]  # Exercise price on each early redemption
sig1 = 0.25  # sigma of the Underlying 1
sig2 = 0.2  # sigma of the Underlying 2
KI = 0.60  # Knock-In barrier
p = 0.17  # 이 상품의 Knock in hit할 확률
n_Steps = 300

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

def osm_algo(Max, Min, sig1, sig2, rho, r, K0, F, T, c, K, KI, n_Steps, Nx, Ny, p):
    dt = T / n_Steps  # delta tau
    hx = (Max[0] - Min[0]) / Nx
    hy = (Max[1] - Min[1]) / Ny  # delta x, delta y
    x = np.linspace(Min[0], Max[0], Nx + 1)
    y = np.linspace(Min[1], Max[1], Ny + 1)  # 최소 최대 지정
    u = np.zeros((Nx + 1, Ny + 1))
    u2 = np.zeros((Nx + 1, Ny + 1))  # 낙인 안칠 때 u(i,j) 및 v(i,j) 사이즈 설정
    k = np.zeros((Nx + 1, Ny + 1))
    k2 = np.zeros((Nx + 1, Ny + 1))  # 낙인 칠 때 u(i,j) 및 v(i,j) 사이즈 설정
    
    for i in range(Nx + 1):
        for j in range(Ny + 1):
            if (x[i] >= K0[0] * K[-1] and y[j] >= K0[1] * K[-1]):  # 1번 케이스 두 자산 모두 만기 수익 받을 때
                u[i, j] = F * (1 + c[-1])

            elif (x[i] > K0[0] * KI and y[j] >= K0[1] * KI) and (x[i] < K[-1] * K0[0] or y[j] < K0[1] * K[-1]):
                u[i, j] = F * (1 + c[-1])  # 2번 케이스: KI 안쳤다고 가정

            elif min(x[i] / K0[0], y[j] / K0[1]) < KI:  # 3번 케이스. KI 베리어 쳤을 때
                u[i, j] = F * min(x[i] / K0[0], y[j] / K0[1])
    
    for i in range(Nx + 1):
        for j in range(Ny + 1):
            if (x[i] >= K0[0] * K[-1] and y[j] >= K0[1] * K[-1]):  # 1번 케이스 두 자산 모두 80프로 이상일 때
                k[i, j] = F * (1 + c[-1])

            elif (x[i] > K0[0] * KI and y[j] >= K0[1] * KI) and (x[i] < K[-1] * K0[0] or y[j] < K0[1] * K[-1]):
                k[i, j] = F * min(x[i] / K0[0], y[j] / K0[1])  # 2번 케이스: KI 쳤다고 가정

            elif min(x[i] / K0[0], y[j] / K0[1]) < KI:  # 3번 케이스. KI 베리어 쳤을 때
                k[i, j] = F * min(x[i] / K0[0], y[j] / K0[1])

    for m in range(1, n_Steps + 1):
        # 1ST STEP#
        for j in range(1, Ny):
            alpha_x = np.zeros(Nx - 1)
            beta_x = np.zeros(Nx - 1)
            gamma_x = np.zeros(Nx - 1)
            fy = np.zeros(Ny - 1)
            for i in range(1, Nx):
                beta_x[i - 1] = 1 / dt + np.power(sig1 * x[i], 2) / hx ** 2 + r * x[i] / hx + 0.5 * r
                alpha_x[i - 1] = -0.5 * np.power(sig1 * x[i], 2) / hx ** 2
                gamma_x[i - 1] = -0.5 * np.power(sig1 * x[i], 2) / hx ** 2 - r * x[i] / hx
                if i == Nx - 1:
                    fy[i - 1] = 0.125 * rho * sig1 * sig2 * x[i] * y[j] \
                                * (2 * u[i, j + 1] - u[i - 1, j + 1] - (2 * u[i, j] - u[i - 1, j]) - u[i, j + 1] + u[
                        i, j]) / (hx ** 2) + u[i, j] / dt
                else:
                    fy[i - 1] = 0.125 * rho * sig1 * sig2 * x[i] * y[j] \
                                * (u[i + 1, j + 1] - u[i + 1, j] - u[i, j + 1] + u[i, j]) / (hx ** 2) + u[i, j] / dt

            beta_x[0] = beta_x[0] + 2.0 * alpha_x[0]  # 메트릭스 조정. 논문 참고.
            gamma_x[0] = gamma_x[0] - alpha_x[0]
            alpha_x[-1] = alpha_x[-1] - gamma_x[-1]
            beta_x[-1] = beta_x[-1] + 2.0 * gamma_x[-1]

            u2[1:Nx, j] = TDMAsolver(alpha_x[1:], beta_x, gamma_x[:-1], fy)
            # 1부터 Nx-1까지 넣어야 하므로, 1:Nx로 인덱싱
        u2[0, 1:Ny] = 2 * u2[1, 1:Ny] - u2[2, 1:Ny]  # 첫행 바운더리
        u2[Nx, 1:Ny] = 2 * u2[Nx - 1, 1:Ny] - u2[Nx - 2, 1:Ny]  # 마지막행 바운더리인데 필요한가? i==100일 때 위에서 했는데
        u2[1:Nx, 0] = 2 * u2[1:Nx, 1] - u2[1:Nx, 2]  # 첫 열 바운더리
        u2[1:Nx, Ny] = 2 * u2[1:Nx, Ny - 1] - u2[1:Nx, Ny - 2]  # 마지막열 바운더리

        for i in range(1, Nx):
            alpha_y = np.zeros(Ny - 1)
            beta_y = np.zeros(Ny - 1)
            gamma_y = np.zeros(Ny - 1)
            fx = np.zeros(Nx - 1)
            for j in range(1, Ny):
                beta_y[j - 1] = 1 / dt + np.power(sig2 * y[j], 2) / hy ** 2 + r * y[j] / hy + 0.5 * r
                alpha_y[j - 1] = -0.5 * np.power(sig2 * y[j], 2) / hy ** 2
                gamma_y[j - 1] = -0.5 * np.power(sig2 * y[j], 2) / hy ** 2 - r * y[j] / hy
                if j == Ny - 1:
                    fx[j - 1] = 0.125 * rho * sig1 * sig2 * x[i] * y[j] \
                                * (2 * u2[i + 1, j] - u2[i + 1, j - 1] - u2[i + 1, j] - (2 * u2[i, j] - u2[i, j - 1]) +
                                   u2[i, j]) / (hy ** 2) + u2[i, j] / dt
                else:
                    fx[j - 1] = 0.125 * rho * sig1 * sig2 * x[i] * y[j] \
                                * (u2[i + 1, j + 1] - u2[i + 1, j] - u2[i, j + 1] + u2[i, j]) / (hy ** 2) + u2[
                                    i, j] / dt

            beta_y[0] = beta_y[0] + 2.0 * alpha_y[0]
            gamma_y[0] = gamma_y[0] - alpha_y[0]
            alpha_y[-1] = alpha_y[-1] - gamma_y[-1]
            beta_y[-1] = beta_y[-1] + 2.0 * gamma_y[-1]

            u[i, 1:Ny] = TDMAsolver(alpha_y[1:], beta_y, gamma_y[:-1], fx)

        u[0, 1:Ny] = 2 * u[1, 1:Ny] - u[2, 1:Ny]  # 첫행 바운더리
        u[Nx, 1:Ny] = 2 * u[Nx - 1, 1:Ny] - u[Nx - 2, 1:Ny]  # 마지막행 바운더리인데 필요한가? i==99일 때 위에서 했는데
        u[1:Nx, 0] = 2 * u[1:Nx, 1] - u[1:Nx, 2]  # 첫 열 바운더리
        u[1:Nx, Ny] = 2 * u[1:Nx, Ny - 1] - u[1:Nx, Ny - 2]  # 마지막열 바운더리

        if m == n_Steps / (2 * T):  # 잔존 만기가 6개월 남았을 때, 즉 2.5년 시점에서
            for i in range(Nx + 1):
                for j in range(Ny + 1):
                    if min(x[i] / K0[0], y[j] / K0[1]) > K[4]:
                        u[i, j] = (1 + c[4]) * F
        if m == 2 * n_Steps / (2 * T):  # 잔존 만기가 1년 남았을 때, 2년 시점에서
            for i in range(Nx + 1):
                for j in range(Ny + 1):
                    if min(x[i] / K0[0], y[j] / K0[1]) > K[3]:
                        u[i, j] = (1 + c[3]) * F
        if m == 3 * n_Steps / (2 * T):  # 잔존만기 1.5년 남았을 떄 1.5년 시점에서
            for i in range(Nx + 1):
                for j in range(Ny + 1):
                    if min(x[i] / K0[0], y[j] / K0[1]) > K[2]:
                        u[i, j] = (1 + c[2]) * F
        if m == 4 * n_Steps / (2 * T):  # 잔존 만기가 2년 남았을 떄/ 1년 시점에서
            for i in range(Nx + 1):
                for j in range(Ny + 1):
                    if min(x[i] / K0[0], y[j] / K0[1]) > K[1]:
                        u[i, j] = (1 + c[1]) * F
        if m == 5 * n_Steps / (2 * T):  # 잔존만기가 6개월 남았을 떄. 6개월 시점에서
            for i in range(Nx + 1):
                for j in range(Ny + 1):
                    if min(x[i] / K0[0], y[j] / K0[1]) > K[0]:
                        u[i, j] = (1 + c[0]) * F

        # hit할 때 알고리즘
        for j in range(1, Ny):
            alpha_x = np.zeros(Nx - 1)
            beta_x = np.zeros(Nx - 1)
            gamma_x = np.zeros(Nx - 1)
            fy = np.zeros(Ny - 1)
            for i in range(1, Nx):
                beta_x[i - 1] = 1 / dt + np.power(sig1 * x[i], 2) / hx ** 2 + r * x[i] / hx + 0.5 * r
                alpha_x[i - 1] = -0.5 * np.power(sig1 * x[i], 2) / hx ** 2
                gamma_x[i - 1] = -0.5 * np.power(sig1 * x[i], 2) / hx ** 2 - r * x[i] / hx
                if i == Nx - 1:
                    fy[i - 1] = 0.5 * rho * sig1 * sig2 * x[i] * y[j] \
                                * (2 * k[i, j + 1] - k[i - 1, j + 1] - (2 * k[i, j] - u[i - 1, j]) - k[i, j + 1] + k[
                        i, j]) / (hx ** 2) + k[i, j] / dt
                else:
                    fy[i - 1] = 0.125 * rho * sig1 * sig2 * x[i] * y[j] \
                                * (k[i + 1, j + 1] - k[i + 1, j] - k[i, j + 1] + k[i, j]) / (hx ** 2) + k[i, j] / dt

            beta_x[0] = beta_x[0] + 2.0 * alpha_x[0]
            gamma_x[0] = gamma_x[0] - alpha_x[0]
            alpha_x[-1] = alpha_x[-1] - gamma_x[-1]
            beta_x[-1] = beta_x[-1] + 2.0 * gamma_x[-1]

            k2[1:Nx, j] = TDMAsolver(alpha_x[1:], beta_x, gamma_x[:-1], fy)
            # 1부터 Nx-1까지 넣어야 하므로, 1:Nx로 인덱싱
        k2[0, 1:Ny] = 2 * k2[1, 1:Ny] - k2[2, 1:Ny]  # 첫행 바운더리
        k2[Nx, 1:Ny] = 2 * k2[Nx - 1, 1:Ny] - k2[Nx - 2, 1:Ny]  # 마지막행 바운더리인데 필요한가? i==100일 때 위에서 했는데
        k2[1:Nx, 0] = 2 * k2[1:Nx, 1] - k2[1:Nx, 2]  # 첫 열 바운더리
        k2[1:Nx, Ny] = 2 * k2[1:Nx, Ny - 1] - k2[1:Nx, Ny - 2]  # 마지막열 바운더리

        # 2ND STEP#
        for i in range(1, Nx):
            alpha_y = np.zeros(Ny - 1)
            beta_y = np.zeros(Ny - 1)
            gamma_y = np.zeros(Ny - 1)
            fx = np.zeros(Nx - 1)
            for j in range(1, Ny):
                beta_y[j - 1] = 1 / dt + np.power(sig2 * y[j], 2) / hy ** 2 + r * y[j] / hy + 0.5 * r
                alpha_y[j - 1] = -0.5 * np.power(sig2 * y[j], 2) / hy ** 2
                gamma_y[j - 1] = -0.5 * np.power(sig2 * y[j], 2) / hy ** 2 - r * y[j] / hy
                if j == Ny - 1:
                    fx[j - 1] = 0.5 * rho * sig1 * sig2 * x[i] * y[j] \
                                * (2 * k2[i + 1, j] - k2[i + 1, j - 1] - k2[i + 1, j] - (2 * k2[i, j] - k2[i, j - 1]) +
                                   k2[i, j]) / (hy ** 2) + k2[i, j] / dt
                else:
                    fx[j - 1] = 0.125 * rho * sig1 * sig2 * x[i] * y[j] \
                                * (k2[i + 1, j + 1] - k2[i + 1, j] - k2[i, j + 1] + k2[i, j]) / (hy ** 2) + k2[
                                    i, j] / dt

            beta_y[0] = beta_y[0] + 2.0 * alpha_y[0]
            gamma_y[0] = gamma_y[0] - alpha_y[0]
            alpha_y[-1] = alpha_y[-1] - gamma_y[-1]
            beta_y[-1] = beta_y[-1] + 2.0 * gamma_y[-1]

            k[i, 1:Ny] = TDMAsolver(alpha_y[1:], beta_y, gamma_y[:-1], fx)

        k[0, 1:Ny] = 2 * k[1, 1:Ny] - k[2, 1:Ny]  # 첫행 바운더리
        k[Nx, 1:Ny] = 2 * k[Nx - 1, 1:Ny] - k[Nx - 2, 1:Ny]  # 마지막행 바운더리인데 필요한가? i==99일 때 위에서 했는데
        k[1:Nx, 0] = 2 * k[1:Nx, 1] - k[1:Nx, 2]  # 첫 열 바운더리
        k[1:Nx, Ny] = 2 * k[1:Nx, Ny - 1] - k[1:Nx, Ny - 2]  # 마지막열 바운더리

    return u * (1 - p) + k * p

Els_price = osm_algo(Max, Min, sig1, sig2, rho, r, K0, F, T, c, K, KI, n_Steps, Nx, Ny, p)
x = np.linspace(Min[0], Max[0], Nx + 1)
y = np.linspace(Min[1], Max[1], Ny + 1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
[x, y] = np.meshgrid(x, y)
ax.plot_surface(x, y, Els_price)
plt.show()
fdm_price = Els_price[50, 50]

hx = (Max[0] - 0) / Nx
hy = (Max[1] - 0) / Ny
u_x = pd.DataFrame(Els_price)
u_y = pd.DataFrame(Els_price)
delta_x = ((u_x - u_x.shift(axis=1)) / hx)
delta_y = ((u_y - u_y.shift(axis=0)) / hy)

fig2 = plt.figure()
x = np.linspace(Min[0], Max[0], Nx + 1)
y = np.linspace(Min[1], Max[1], Ny + 1)

plt.plot(x[1:], delta_x.iloc[50, 1:])
plt.xlabel('A1')
plt.ylabel('delta')
plt.axvline(x=K0[0] * KI)
plt.show()
fig3 = plt.figure()
plt.plot(y[1:], delta_y.iloc[1:, 50])
plt.xlabel('A2')
plt.ylabel('delta')
plt.axvline(x=K0[1] * KI)
plt.show()

gamma_x = ((delta_x - delta_x.shift(axis=1)) / hx) * 100
gamma_y = ((delta_y - delta_y.shift(axis=0)) / hy) * 100
fig3 = plt.figure()
plt.plot(x[2:], gamma_x.iloc[50, 2:])
plt.xlabel('A1')
plt.ylabel('gamma')
plt.axvline(x=K0[0] * KI)
plt.show()
fig4 = plt.figure()
plt.plot(y[2:], gamma_y.iloc[2:, 50])
plt.xlabel('A2')
plt.ylabel('gamma')
plt.axvline(x=K0[1] * KI)
plt.show()