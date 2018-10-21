# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 19:03:56 2018

@author: 한승표
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy

# 과거 일별 데이터
# underlying_data = pd.read_excel(r'data.xlsx', sheet_name='data', columns=['KSP200', 'EURO50'])  # 기초자산의 데이터
# under_ret = underlying_data.pct_change()  # 일별 수익률 계산
# mu = under_ret.mean() * 365  # 1년 수익률
# sig = under_ret.std() * np.sqrt(365)  # 1년 표준편차
#
# corr = under_ret.corr()  # 두 자산의 상관관계
# =============================================================================
# ##기본 파라미터##
# =============================================================================
Max = [7000, 6000]  # 두 자산의 최대값
Min = [0, 0]  # 두 자산의 최소값
sig1 = 0.25;  # Volatility of the underlying asset 1
sig2 = 0.2;  # Volatility of the underlying asset 2
rho = 0.4;  # Correlation between prices of the two assets
r = 0.022;  # 2014년 10월에 공시된 91일 CD 금리
K0 = [3500,3000];  # 두 자산의 기준 가견(Refefence price)
F = 100;  # Face value
T = 3;  # Maturation of contract
c = [0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375];  # Rate of return on each early redemption date
K = [0.85, 0.80, 0.75, 0.70, 0.70, 0.70];  # Exercise price on each early redemption date
KI = 0.60;  # Knock-In barrier level
n_Steps = 300;  # 시간의 노드
Nx = 100;  # KOSPI200의 노드
Ny = 100;  # HSCEI지수 노드
p = 0.17  # 이 상품의 Knock in hit할 확률


# 직접 돌린 가격 9839(OSM)
# 시뮬레이션으로 한 가격 9852(1000번 시뮬레이션을 10번 시행하여 평균 )
# %%
# =============================================================================
# ##낙인 칠 확률 계산, 같이 돌리면 오래 걸려서 이걸 먼저 시행하여 확률 계산 후 고정
# =============================================================================
# q1 = 0; q2=0;
# n_trials = 1000
# def hit_pro(n_trials,n_Steps,T,rho,sig1,sig2,K0,q1,q2,KI):
#    dt = 0.5 /n_Steps; T = T*2;
#    randn_matrix_1 = np.random.normal(size=(n_trials, n_Steps*T)) #epsilon1
#    randn_matrix_2 = np.random.normal(size=(n_trials, n_Steps*T)) #epsilon2
#    randn_matrix_S1 = randn_matrix_1
#    randn_matrix_S2 =  rho * randn_matrix_S1 + np.sqrt(1 - rho ** 2) * randn_matrix_2
#    s1_matrix = np.zeros((n_trials, n_Steps*T))
#    s1_matrix[:,0] = K0[0]
#    s2_matrix = np.zeros((n_trials, n_Steps*T))
#    s2_matrix[:,0] = K0[1]
#    hit = 0
#    for j in range((n_Steps*T)-1):
#        s1_matrix[:,j+1] = s1_matrix[:,j] * np.exp((r-q1-0.5*sig1**2)*dt + sig1*np.sqrt(dt)*randn_matrix_S1[:,j])
#        s2_matrix[:,j+1] = s2_matrix[:,j] * np.exp((r-q2-0.5*sig2**2)*dt + sig2*np.sqrt(dt)*randn_matrix_S2[:,j])
#
##낙인을 한번이라도 칠 확률
#    for i in range(n_trials):
#        if min(min(s1_matrix[i]/K0[0]), min(s2_matrix[i]/K0[1]))<KI:
#            hit = hit+1
#    KI_pro = hit / n_trials
#
#    return KI_pro
#
# p = 0
# for i in range(10):
#    n_trials = 1000*(1+i)
#    p = p+ hit_pro(n_trials,n_Steps,T,rho,sig1,sig2,K0,q1,q2,KI)
#
# p = p/10
# %%
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
# =============================================================================
# OSM알고리즘 함수 정의
# =============================================================================
def algorism(Max, Min, sig1, sig2, rho, r, K0, F, T, c, K, KI, n_Steps, Nx, Ny, p):
    dt = T / n_Steps;  # delta tau
    hx = (Max[0] - Min[0]) / Nx;
    hy = (Max[1] - Min[1]) / Ny;  # delta x, delta y
    x = np.linspace(Min[0], Max[0], Nx + 1);
    y = np.linspace(Min[1], Max[1], Ny + 1);  # 최소 최대 지정
    u = np.zeros((Nx + 1, Ny + 1));
    u2 = np.zeros((Nx + 1, Ny + 1));  # 낙인 안칠 때 u(i,j) 및 v(i,j) 사이즈 설정
    k = np.zeros((Nx + 1, Ny + 1));
    k2 = np.zeros((Nx + 1, Ny + 1));  # 낙인 칠 때 u(i,j) 및 v(i,j) 사이즈 설정
    # =============================================================================
    # no hit payoff
    # =============================================================================
    for i in range(Nx + 1):
        for j in range(Ny + 1):
            if (x[i] >= K0[0] * K[-1] and y[j] >= K0[1] * K[-1]):  # 1번 케이스 두 자산 모두 만기 수익 받을 때
                u[i, j] = F * (1 + c[-1])

            elif (x[i] > K0[0] * KI and y[j] >= K0[1] * KI) and (x[i] < K[-1] * K0[0] or y[j] < K0[1] * K[-1]):
                u[i, j] = F * (1 + c[-1])  # 2번 케이스: KI 안쳤다고 가정

            elif min(x[i] / K0[0], y[j] / K0[1]) < KI:  # 3번 케이스. KI 베리어 쳤을 때
                u[i, j] = F * min(x[i] / K0[0], y[j] / K0[1])
    # =============================================================================
    # hit payoff
    # =============================================================================
    for i in range(Nx + 1):
        for j in range(Ny + 1):
            if (x[i] >= K0[0] * K[-1] and y[j] >= K0[1] * K[-1]):  # 1번 케이스 두 자산 모두 80프로 이상일 때
                k[i, j] = F * (1 + c[-1])

            elif (x[i] > K0[0] * KI and y[j] >= K0[1] * KI) and (x[i] < K[-1] * K0[0] or y[j] < K0[1] * K[-1]):
                k[i, j] = F * min(x[i] / K0[0], y[j] / K0[1])  # 2번 케이스: KI 쳤다고 가정

            elif min(x[i] / K0[0], y[j] / K0[1]) < KI:  # 3번 케이스. KI 베리어 쳤을 때
                k[i, j] = F * min(x[i] / K0[0], y[j] / K0[1])
    # =============================================================================
    # 알고리즘 no hit 일 때
    # =============================================================================
    for m in range(1, n_Steps + 1):
        # 1ST STEP#
        for j in range(1, Ny):
            alpha_x = np.zeros(Nx - 1);
            beta_x = np.zeros(Nx - 1);
            gamma_x = np.zeros(Nx - 1);
            fy = np.zeros(Ny - 1);
            for i in range(1, Nx):
                beta_x[i - 1] = 1 / dt + np.power(sig1 * x[i], 2) / hx ** 2 + r * x[i] / hx + 0.5 * r
                alpha_x[i - 1] = -0.5 * np.power(sig1 * x[i], 2) / hx ** 2
                gamma_x[i - 1] = -0.5 * np.power(sig1 * x[i], 2) / hx ** 2 - r * x[i] / hx
                if i == Nx - 1:
                    fy[i - 1] = 0.125 * rho * sig1 * sig2 * x[i] * y[j] \
                                * (2 * u[i, j + 1] - u[i - 1, j + 1] - (2 * u[i, j] - u[i - 1, j]) - u[i, j + 1] + u[
                        i, j]) / (hx ** 2) + u[i, j] / dt;
                else:
                    fy[i - 1] = 0.125 * rho * sig1 * sig2 * x[i] * y[j] \
                                * (u[i + 1, j + 1] - u[i + 1, j] - u[i, j + 1] + u[i, j]) / (hx ** 2) + u[i, j] / dt;

            beta_x[0] = beta_x[0] + 2.0 * alpha_x[0];  # 메트릭스 조정. 논문 참고.
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
                if j == Ny - 1:
                    fx[j - 1] = 0.125 * rho * sig1 * sig2 * x[i] * y[j] \
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

        # =============================================================================
        # hit할 때 알고리즘
        # =============================================================================
        for j in range(1, Ny):
            alpha_x = np.zeros(Nx - 1);
            beta_x = np.zeros(Nx - 1);
            gamma_x = np.zeros(Nx - 1);
            fy = np.zeros(Ny - 1);
            for i in range(1, Nx):
                beta_x[i - 1] = 1 / dt + np.power(sig1 * x[i], 2) / hx ** 2 + r * x[i] / hx + 0.5 * r
                alpha_x[i - 1] = -0.5 * np.power(sig1 * x[i], 2) / hx ** 2
                gamma_x[i - 1] = -0.5 * np.power(sig1 * x[i], 2) / hx ** 2 - r * x[i] / hx
                if i == Nx - 1:
                    fy[i - 1] = 0.5 * rho * sig1 * sig2 * x[i] * y[j] \
                                * (2 * k[i, j + 1] - k[i - 1, j + 1] - (2 * k[i, j] - u[i - 1, j]) - k[i, j + 1] + k[
                        i, j]) / (hx ** 2) + k[i, j] / dt;
                else:
                    fy[i - 1] = 0.125 * rho * sig1 * sig2 * x[i] * y[j] \
                                * (k[i + 1, j + 1] - k[i + 1, j] - k[i, j + 1] + k[i, j]) / (hx ** 2) + k[i, j] / dt;

            beta_x[0] = beta_x[0] + 2.0 * alpha_x[0];
            gamma_x[0] = gamma_x[0] - alpha_x[0];
            alpha_x[-1] = alpha_x[-1] - gamma_x[-1];
            beta_x[-1] = beta_x[-1] + 2.0 * gamma_x[-1];

            k2[1:Nx, j] = TDMAsolver(alpha_x[1:], beta_x, gamma_x[:-1], fy);
            # 1부터 Nx-1까지 넣어야 하므로, 1:Nx로 인덱싱
        k2[0, 1:Ny] = 2 * k2[1, 1:Ny] - k2[2, 1:Ny];  # 첫행 바운더리
        k2[Nx, 1:Ny] = 2 * k2[Nx - 1, 1:Ny] - k2[Nx - 2, 1:Ny];  # 마지막행 바운더리인데 필요한가? i==100일 때 위에서 했는데
        k2[1:Nx, 0] = 2 * k2[1:Nx, 1] - k2[1:Nx, 2];  # 첫 열 바운더리
        k2[1:Nx, Ny] = 2 * k2[1:Nx, Ny - 1] - k2[1:Nx, Ny - 2];  # 마지막열 바운더리

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
                if j == Ny - 1:
                    fx[j - 1] = 0.5 * rho * sig1 * sig2 * x[i] * y[j] \
                                * (2 * k2[i + 1, j] - k2[i + 1, j - 1] - k2[i + 1, j] - (2 * k2[i, j] - k2[i, j - 1]) +
                                   k2[i, j]) / (hy ** 2) + k2[i, j] / dt;
                else:
                    fx[j - 1] = 0.125 * rho * sig1 * sig2 * x[i] * y[j] \
                                * (k2[i + 1, j + 1] - k2[i + 1, j] - k2[i, j + 1] + k2[i, j]) / (hy ** 2) + k2[
                                    i, j] / dt;

            beta_y[0] = beta_y[0] + 2.0 * alpha_y[0];
            gamma_y[0] = gamma_y[0] - alpha_y[0];
            alpha_y[-1] = alpha_y[-1] - gamma_y[-1];
            beta_y[-1] = beta_y[-1] + 2.0 * gamma_y[-1];

            k[i, 1:Ny] = TDMAsolver(alpha_y[1:], beta_y, gamma_y[:-1], fx);

        k[0, 1:Ny] = 2 * k[1, 1:Ny] - k[2, 1:Ny];  # 첫행 바운더리
        k[Nx, 1:Ny] = 2 * k[Nx - 1, 1:Ny] - k[Nx - 2, 1:Ny];  # 마지막행 바운더리인데 필요한가? i==99일 때 위에서 했는데
        k[1:Nx, 0] = 2 * k[1:Nx, 1] - k[1:Nx, 2];  # 첫 열 바운더리
        k[1:Nx, Ny] = 2 * k[1:Nx, Ny - 1] - k[1:Nx, Ny - 2];  # 마지막열 바운더리

    return u * (1 - p) + k * p


# %%
# ELS가격 산출
Els_price = algorism(Max, Min, sig1, sig2, rho, r, K0, F, T, c, K, KI, n_Steps, Nx, Ny, p)
x = np.linspace(Min[0], Max[0], Nx + 1);
y = np.linspace(Min[1], Max[1], Ny + 1);
# %%
# ELS 그래프
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
[x, y] = np.meshgrid(x, y)
ax.plot_surface(x, y, Els_price)
plt.show()
fdm_price = Els_price[50, 50]

# %%
# delta 산출 및 그래프
hx = (Max[0] - 0) / Nx
hy = (Max[1] - 0) / Ny
u_x = pd.DataFrame(Els_price)
u_y = pd.DataFrame(Els_price)
delta_x = ((u_x - u_x.shift(axis=1)) / hx)
delta_y = ((u_y - u_y.shift(axis=0)) / hy)

fig2 = plt.figure()
x = np.linspace(Min[0], Max[0], Nx + 1);
y = np.linspace(Min[1], Max[1], Ny + 1);

plt.plot(x[1:], delta_x.iloc[50, 1:]);
plt.xlabel('A1')
plt.ylabel('delta')
plt.axvline(x=K0[0] * KI, color='orange')
plt.show()
fig3 = plt.figure()
plt.plot(y[1:], delta_y.iloc[1:, 50])
plt.xlabel('A2')
plt.ylabel('delta')
plt.axvline(x=K0[1] * KI, color='orange')
plt.show()
# %%
# gamma 산출 및 그래프
gamma_x = ((delta_x - delta_x.shift(axis=1)) / hx) * 100
gamma_y = ((delta_y - delta_y.shift(axis=0)) / hy) * 100
fig3 = plt.figure()
plt.plot(x[2:], gamma_x.iloc[50, 2:]);
plt.xlabel('A1')
plt.ylabel('gamma')
plt.axvline(x=K0[0] * KI, color='orange')
plt.show()
fig4 = plt.figure()
plt.plot(y[2:], gamma_y.iloc[2:, 50])
plt.xlabel('A2')
plt.ylabel('gamma')
plt.axvline(x=K0[1] * KI, color='orange')
plt.show()
# %%
# vega
sig11 = sig1 + 0.01
sig22 = sig2 + 0.01
Els_price2 = algorism(Max, Min, sig11, sig2, rho, r, K0, F, T, c, K, KI, n_Steps, Nx, Ny, p)
sig1_price = Els_price2[50, 50]
Els_price3 = algorism(Max, Min, sig1, sig22, rho, r, K0, F, T, c, K, KI, n_Steps, Nx, Ny, p)
sig2_price = Els_price3[50, 50]

# %%
# rho
rho1 = rho + 0.01
Els_price4 = algorism(Max, Min, sig1, sig2, rho1, r, K0, F, T, c, K, KI, n_Steps, Nx, Ny, p)
rho_price = Els_price4[50, 50]

# %%
# =============================================================================
# 시나리오 분석
# =============================================================================
# 몬텤 카롤로 변수
n_trials = 1000
n_steps = 300
tt = 6
r = 0.022
q1 = 0
q2 = 0
v1 = 0.127
v2 = 0.238
rho = 0.164
dt = 0.5 / n_steps
Sum = 0
s1 = 237.15
s2 = 3088.18
Face_value = 10000
con = [0.90, 0.90, 0.85, 0.85, 0.80, 0.8]
ret = [0.035, 0.07, 0.105, 0.14, 0.175, 0.21]
KI = 0.6


# %%
def monte_els(s1, s2, v1, v2, r, q1, q2, rho, dt, T, n_trials, n_steps, con, ret, KI):
    # process 설정
    randn_matrix_1 = np.random.normal(size=(n_trials, n_steps * T))  # epsilon1
    randn_matrix_2 = np.random.normal(size=(n_trials, n_steps * T))  # epsilon2
    randn_matrix_S1 = randn_matrix_1
    randn_matrix_S2 = rho * randn_matrix_S1 + np.sqrt(1 - rho ** 2) * randn_matrix_2
    s1_matrix = np.zeros((n_trials, n_steps * T))
    s1_matrix[:, 0] = s1
    s2_matrix = np.zeros((n_trials, n_steps * T))
    s2_matrix[:, 0] = s2
    for j in range((n_steps * T) - 1):
        s1_matrix[:, j + 1] = s1_matrix[:, j] * np.exp(
            (r - q1 - 0.5 * v1 ** 2) * dt + v1 * np.sqrt(dt) * randn_matrix_S1[:, j])
        s2_matrix[:, j + 1] = s2_matrix[:, j] * np.exp(
            (r - q2 - 0.5 * v2 ** 2) * dt + v2 * np.sqrt(dt) * randn_matrix_S2[:, j])

    # 조기상환 조건
    exp1 = np.ones((n_trials, T)) * s1
    exp2 = np.ones((n_trials, T)) * s2
    exp_con = con
    sit = np.zeros(7)
    f = np.zeros(n_trials)
    # Payoff 구조
    for i in range(len(exp_con)):
        exp1[:, i] = exp1[:, i] * exp_con[i]
        exp2[:, i] = exp2[:, i] * exp_con[i]

    for i in range(n_trials):
        # 0.5년 때 조기상환
        if s1_matrix[i, n_steps - 1] >= exp1[i, 0] and s2_matrix[i, n_steps - 1] >= exp2[i, 0]:
            f[i,] = Face_value * (1 + ret[0]) * np.exp(-r * 0.5)
            sit[0] = sit[0] + 1

        # 1년 때 조기상환
        elif s1_matrix[i, 2 * n_steps - 1] >= exp1[i, 1] and s2_matrix[i, 2 * n_steps - 1] >= exp2[i, 1]:
            f[i,] = Face_value * (1 + ret[1]) * np.exp(-r * 1)
            sit[1] = sit[1] + 1

        # 1.5년 때 조기상환
        elif s1_matrix[i, 3 * n_steps - 1] >= exp1[i, 2] and s2_matrix[i, 3 * n_steps - 1] >= exp2[i, 2]:
            f[i,] = Face_value * (1 + ret[2]) * np.exp(-r * 1.5)
            sit[2] = sit[2] + 1

        # 2년 때 조기상환
        elif s1_matrix[i, 4 * n_steps - 1] >= exp1[i, 3] and s2_matrix[i, 4 * n_steps - 1] >= exp2[i, 3]:
            f[i,] = Face_value * (1 + ret[3]) * np.exp(-r * 2)
            sit[3] = sit[3] + 1

        # 2.5년 때 조기상환
        elif s1_matrix[i, 5 * n_steps - 1] >= exp1[i, 4] and s2_matrix[i, 5 * n_steps - 1] >= exp2[i, 4]:
            f[i,] = Face_value * (1 + ret[4]) * np.exp(-r * 2.5)
            sit[4] = sit[4] + 1

        # 만기 때 Payoff
        elif s1_matrix[i, -1] >= exp1[i, 5] and s2_matrix[i, -1] >= exp2[i, 5]:
            f[i,] = Face_value * (1 + ret[5]) * np.exp(-r * 3)
            sit[5] = sit[5] + 1

        #        elif  s1_matrix[i,-1] <exp1[i,5] or s2_matrix[i,-1] < exp2[i,5]:
        #            if (len([x for x in s1_matrix.flatten() < s1*KI])==0) and (len([y for y in s2_matrix.flatten() < s2*KI])==0):
        #                f[i,] = Face_value * (1+ret[5]) * np.exp(-r*3)
        #                sit[5] = sit[5] +1
        #
        #            if s1_matrix[:,:] >s1*KI and s2_matrix[:,:] >s2*KI:
        #                f[i,] = Face_value * (1+ret[5]) * np.exp(-r*3)
        #                sit[5] = sit[5] +1
        #                sum_p[5] = Face_value * (1+ret[5]) * np.exp(-r*3)
        else:
            # s1_matrix[:,:] < s1*KI or s2_matrix[:,:] < s2*KI:
            f[i,] = Face_value * np.minimum(s2_matrix[i, -1] / s2_matrix[i, 0],
                                            s1_matrix[i, -1] / s1_matrix[i, 0]) * np.exp(-r * 3)
            sit[6] = sit[6] + 1

    Sum = f.sum()
    probability = np.ones(len(sit))
    for i in range(7):
        probability[i] = sit[i] / n_trials
    values = Sum / n_trials
    err = np.std(f) / np.sqrt(n_trials)
    return values, err, probability

# %%
# =============================================================================
# #변동성, rho에 대한 시나리오 분석. 한번에 돌리면 오래 걸려서 나눠서 돌렸습니다.
# =============================================================================
#sig1_new = np.linspace(0, 0.5, 5)
# sig2_new = np.linspace(0,0.5,5)
 rho_new = np.linspace(-1,1,5)
#ELS_price1 = []
#ELS_price1_1 = []
# ELS_price2 = []
# ELS_price2_1 = []
 ELS_price3 = []
 ELS_price3_1 = []
for i in range(5):
#    case1 = algorism(Max, Min, sig1_new[i], sig2, rho, r, K0, F, T, c, K, KI, n_Steps, Nx, Ny, p)
#    case2 = algorism(Max,Min,sig1,sig2_new[i],rho,r,K0,F,T,c,K,KI,n_Steps,Nx,Ny,p)
    case3 = algorism(Max,Min,sig1,sig2,rho_new[i],r,K0,F,T,c,K,KI,n_Steps,Nx,Ny,p)
#    case1_1 = monte_els(K0[0], K0[1], sig1_new[i], sig2, r, q1, q2, rho, dt, T*2, n_trials, n_Steps, K, c, KI)[0]
#    case2_1 =  monte_els(s1,s2,v1,sig2_new[i],r,q1,q2,rho,dt,tt,n_trials,n_steps,con,ret,KI)[0]
    case3_1 =  monte_els(s1,s2,v1,v2,r,q1,q2,rho_new[i],dt,tt,n_trials,n_steps,con,ret,KI)[0]
#    ELS_price1.append(case1)
#    ELS_price1_1.append(case1_1)
#    ELS_price2.append(case2)
#    ELS_price2_1.append(case2_1)
    ELS_price3.append(case3)
    ELS_price3_1.append(case3_1)
#    print('sig1이 %f 일때 가격  = %f' % (sig1_new[i], ELS_price1[i][50, 50]))
#    print('sig1이 %f 일때 monte가격  = %f' % (sig1_new[i], ELS_price1_1[i]))
#    print('sig2이 %f 일때 가격  = %f' %(sig2_new[i], ELS_price2[i][50,50]))
#    print('sig2이 %f 일때 monte가격  = %f' %(sig2_new[i], ELS_price2_1[i]))
    print('rho가 %f 일때 가격  = %f' %(rho_new[i], ELS_price3[i][50,50]))
    print('rho가 %f 일때 monte가격  = %f' %(rho_new[i], ELS_price3_1[i]))
# %%
# =============================================================================
#     기초자산의 노드가  변할 때 시나리오 분석
# =============================================================================
node1 = [50, 100, 200, 300]
ELS_price1 = []
ELS_price2 = []
for i in range(3):
    Nxx = node1[i]
    Nyy = node1[i]
    case1 = algorism(Max, Min, sig1, sig2, rho, r, K0, F, T, c, K, KI, n_Steps, Nxx, Nyy, p)
    ELS_price1.append(case1)
    print('Nx가 %f 일때 가격  = %f' % (node1[i], ELS_price1[i][50, 50]))

# %%
# =============================================================================
#     Time step이 벼할 때 시나리오 분석
# =============================================================================
n_steps = [100, 300, 500]
ELS_price2 = []
ELS_price3 = []
for i in range(len(n_steps)):
    case2 = algorism(Max, Min, sig1, sig2, rho, r, K0, F, T, c, K, KI, n_steps[i], Nx, Ny, p)
    case = monte_els(s1, s2, v1, v2, r, q1, q2, rho, dt, tt, n_trials, n_steps[i], con, ret, KI)[0]
    ELS_price2.append(case)
    ELS_price3.append(case2)
    print('time steps가 %d 일때 monte가격  = %f' % (n_steps[i], ELS_price2[i]))
    print('time steps가 %d 일때 가격  = %f' % (n_steps[i], ELS_price3[i][50, 50]))