# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats
import os

os.chdir('C:\\Users\\mjh\\Documents\\Numerical-Methods-in-Finance')

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
for i in range(1, 8):
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
'''그러나 여전히 음수를 가진 변동성이 계속 존재함... 무시하고 진행하면..'''
''' 음수를 다 nan값으로 바꿔주고 다시 보간법...'''
#
#localvol_index = range(0, 365 * 3)  # 시간은 3년으로 늘림
#localvol = pd.DataFrame(columns=buff.columns, index=localvol_index).replace(np.nan, True) * vol

#empty = localvol.iloc[20, :].dropna()
#x_points = np.array(empty.index, dtype='float64')
#y_points = np.array(empty.values)
#lininter = sp.interpolate.interp1d(x_points, y_points, fill_value='extrapolate')
#localvol.iloc[20, :] = lininter(np.linspace(270, 330, 25))
#
#empty = localvol.iloc[83, :].dropna()
#x_points = np.array(empty.index, dtype='float64')
#y_points = np.array(empty.values)
#lininter = sp.interpolate.interp1d(x_points, y_points, fill_value='extrapolate')
#localvol.iloc[83, :] = lininter(np.linspace(270, 330, 25))
#
#for i in range(0, 25):
#    empty = localvol.iloc[0:, i].dropna()
#    x_points = np.array(empty.index, dtype='float64')
#    y_points = np.array(empty.values)
#    lininter = sp.interpolate.interp1d(x_points, y_points, fill_value='extrapolate')
#    for j in range(0, 1095):
#        localvol.iloc[j, i] = lininter(j)
'''좀 더 깔끔해보이긴 하지만... 그래도 여전히 변동성이 음의 값을 갖는다'''

# %%
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


x = []
y = []
z = []
for i in range(8):
    for j in range(25):
        x.append(vol.index[i])
        y.append(vol.columns[j])
        z.append(vol.iloc[i, j])
        
empty = np.isfinite(z)
x = (pd.Series(x) * empty).replace(0, np.nan).dropna()
y = (pd.Series(y) * empty).replace(0, np.nan).dropna()
z = (pd.Series(z) * empty).replace(0, np.nan).dropna()

f = sp.interpolate.interp2d(x, y, z, kind='linear')

fig = plt.figure(figsize=(8, 6))
ax = fig.gca(projection='3d')

# Make data.
xnew = np.linspace(0.4, 3, 300)
ynew = np.linspace(270, 330, 300)
znew = f(xnew, ynew)
xnew, ynew = np.meshgrid(xnew, ynew)


# Plot the surface.
surf = ax.plot_surface(xnew, ynew, znew, cmap=cm.coolwarm,linewidth=0, antialiased=False)

# Customize the z axis.
# ax.set_zlim(0.0, 4.00)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()