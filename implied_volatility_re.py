# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import scipy as sp
from scipy.interpolate import griddata
from scipy import stats
import os

os.chdir('C:\\Users\\mjh\\Documents\\Numerical-Methods-in-Finance')

row_num = 8
col_num = 25
# call_option 가격 불러오기
buff = pd.read_excel("local_vol.xlsx", skip_footer=4)  # 쓸데 없는 거 버리고..

# 이론적으로 맞지 않는 값들을 제거해줌.. 잔존만기가 길 수록 시장가가 높아야하는데, 그렇지 않은 경우 제거함.
# 즉, 시장의 노이즈 제거
'''for i in range(1,len(buff)):
    buff.iloc[i,:]=(buff.iloc[i,:]>buff.iloc[i-1,:])*buff.iloc[i,:]'''
buff[1:] = buff[1:].replace(0, np.nan)

call_index = range(0, 365 * 3)  # 시간은 3년으로 늘림
call = pd.DataFrame(columns=buff.columns, index=call_index).replace(np.nan, True) * buff
call[1:] = call[1:].replace(0, np.nan)  # 잔존만기가 0일 때의 0값은 True값이므로 제외하고 replace함

# %% volatility 먼저 추정!!

buff.index=buff.index/365
vol=buff.copy()
for i in range(1, row_num):
    for j in range(1,24):
        C_T= (buff.iloc[i,j]-buff.iloc[i-1,j])/(buff.index[i]-buff.index[i-1])
        C_K=(0.0165*buff.columns[j])*(buff.iloc[i,j]-buff.iloc[i,j-1])/2.5
        C_KK=(buff.columns[j]**2)*(buff.iloc[i,j+1]-2*buff.iloc[i,j]+buff.iloc[i,j-1])/(2.5**2)
        if np.round(C_KK,9) != 0:
            vol.iloc[i,j]=np.sqrt(2*(C_T+C_K)/C_KK)
        else:
            vol.iloc[i,j]=np.nan
vol.iloc[0, :] = np.nan
vol.iloc[:, 0] = np.nan
vol.iloc[:, 24] = np.nan

# %%
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


x = []
y = []
z = []
for j in range(25):
    y.append(vol.columns[j])
for i in range(row_num):
    x.append(vol.index[i])
    for j in range(25):
        z.append(vol.iloc[i, j])
#empty = np.isfinite(z)
#x = (pd.Series(x) * empty).replace(0, np.nan).dropna()
#y = (pd.Series(y) * empty).replace(0, np.nan).dropna()
#z = (pd.Series(z) * empty).replace(0, np.nan).dropna()

x, y = np.meshgrid(x, y)

sero = np.arange(25)
garo = np.arange(row_num)
nae = pd.DataFrame(index=sero, columns=garo)
z = pd.DataFrame(np.nan_to_num(np.asarray(z).reshape(25,row_num)))

for col in z.columns:
    z[col] = z[col].replace(0,np.nan)
    z[col] = z[col].interpolate(method='linear',limit_direction='both')

f = sp.interpolate.interp2d(x, y, z, kind='cubic')

fig = plt.figure(figsize=(8, 6))
ax = fig.gca(projection='3d')

xnew = np.linspace(0, 3, 300)
ynew = np.linspace(270, 330, 300)
znew = f(xnew, ynew)
xnew, ynew = np.meshgrid(xnew, ynew)

surf = ax.plot_surface(xnew, ynew, znew, cmap=cm.coolwarm,linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()