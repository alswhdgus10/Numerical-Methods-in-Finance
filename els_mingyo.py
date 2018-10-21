import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm

sig_x = 0.16404837857869314 # Volatility of EUROSTOXX50
sig_y = 0.23255349796074562 # Volatility of HSCEI

rho = 0.17180201357427863 # Correlation between prices of the two assets

r = 0.022 # Interest rate

S0 = [3340.93,12004.51]  # initial underlying asset value
x0 , y0 = S0 # x0=EUROSTOXX50, y0=HSCEI

xmax = x0*2 # EUROSTOXX50 maximum price
ymax = y0*2 # KOSPI200 maximum price

mu_x = 0.0561105 # mean return of EUROSTOXX50
mu_y = -0.205598 # mean return of HSCEI


F = 10000 # Face value
T=3 # Maturation of contract
c = [0.023, 0.046, 0.069, 0.092, 0.115, 0.118] # Rate of return on each early redemption date

K = [0.90, 0.85, 0.80, 0.80, 0.75,0.7] # Exercise price on each early redemption date
KI = 0.50 # Knock-In barrier level

#pp = 50 # Number of time points in each 6 months
pp = 118 # Number of time points in each 6 months

Nt = (T*2) * pp # Total number of time points (monthly frequency)
Nx = 100 # Number of space(X) points
Ny = 100 # Number of space(Y) points

Nx0 = round(Nx/2) # Number of current price X
Ny0 = round(Ny/2) # Number of current price Y

dt = T/Nt # Time step

dx = xmax/Nx # Space(X) step
dy = ymax/Ny # Space(Y) step

#%%

#=======================#%%=========================================================================================================================================================================================

w = np.zeros((Nt+1,Nx+1,Ny+1)) # KI event does not happen
v = np.zeros((Nt+1,Nx+1,Ny+1)) # KI event happens

# Initial condition

# NO BARRIER TOUCH
w[0,round(x0*KI/dx):,round(y0*KI/dy):] = F*(1+c[5])


# BARRIER TOUCH
for i in np.arange(Nx):
    for j in np.arange(Ny):
        v[0,i,j] = F * min( i*dx/x0, j*dy/y0 )

v[0,round(x0*K[5]/dx):,round(y0*K[5]/dy):] = F*(1+c[5])

# boundary condition for w[k+1]        
for j in np.arange(0,Ny+1,1):
    w[:,0,j] = 2*w[:,1,j] - w[:,2,j]
    w[:,Nx,j] = 2*w[:,Nx-1,j] - w[:,Nx-2,j]
for i in np.arange(0,Nx+1,1):
    w[:,i,0] = 2*w[:,i,1] - w[:,i,2]
    w[:,i,Ny] = 2*w[:,i,Ny-1] - w[:,i,Ny-2]  
    
for j in np.arange(0,Ny+1,1):
    v[:,0,j] = 2*v[:,1,j] - v[:,2,j]
    v[:,Nx,j] = 2*v[:,Nx-1,j] - v[:,Nx-2,j]
for i in np.arange(0,Nx+1,1):
    v[:,i,0] = 2*v[:,i,1] - v[:,i,2]
    v[:,i,Ny] = 2*v[:,i,Ny-1] - v[:,i,Ny-2]  

#================================================================================================================================================================================================================

alpha = np.zeros(Nx)
beta = np.zeros(Nx)
gamma = np.zeros(Nx)

for i in np.arange(0,Nx,1):
    alpha[i] = -0.5*((sig_x*(i-0.5))**2)
    beta[i] = (1/dt) + (sig_x*(i-0.5))**2 + r*(i-0.5) + r/2
    gamma[i] = -0.5*((sig_x*(i-0.5))**2) - r*(i-0.5)


Ax = np.zeros([Nx,Nx])
for i in np.arange(Nx):
    for j in np.arange(Nx):
        if i == j:
            Ax[i,j] = beta[i]
        elif i == j+1:
            Ax[i,j] = alpha[i]
        elif i == j-1:
            Ax[i,j] = gamma[i]            
            
Ax[0,0] = Ax[0,0] + 2*alpha[0]
Ax[0,1] = Ax[0,1] - alpha[0]
Ax[-1,-1] = Ax[-1,-1] + 2*gamma[Nx-1]
Ax[-1,-2] = Ax[-1,-2] - gamma[Nx-1]      

#================================================================================================================================================================================================================

alpha2 = np.zeros(Ny)
beta2 = np.zeros(Ny)
gamma2 = np.zeros(Ny)
      
for j in np.arange(0,Ny,1):
    alpha2[j] = -0.5*((sig_y*(j-0.5))**2)
    beta2[j] = (1/dt) + ( sig_y*(j-0.5))**2 + r*(j-0.5) + r/2
    gamma2[j] = -0.5*((sig_y*(j-0.5))**2) - r*(j-0.5)        
        
Ay = np.zeros([Ny,Ny])
for i in np.arange(Ny):
    for j in np.arange(Ny):
        if i == j:
            Ay[i,j] = beta2[i]
        elif i == j+1:
            Ay[i,j] = alpha2[i]
        elif i == j-1:
            Ay[i,j] = gamma2[i]      
            
Ay[0,0] = Ay[0,0] + 2*alpha2[0]
Ay[0,1] = Ay[0,1] - alpha2[0]
Ay[-1,-1] = Ay[-1,-1] + 2*gamma2[Ny-1]
Ay[-1,-2] = Ay[-1,-2] - gamma2[Ny-1] 


#================================================================================================================================================================================================================

# No Knock-In event case

for k in np.arange(Nt):
        
    f = np.zeros((Nx+1,Ny+1))
    w_hat = np.zeros((Nx+1,Ny+1))
    g = np.zeros((Ny+1,Ny+1))
    if k%pp==0:
        w[k,round(x0*K[-int(k/pp)-1]/dx):,round(y0*K[-int(k/pp)-1]/dy):] = F*(1+c[-int(k/pp)-1])

    for j in np.arange(1,Ny+1,1):
        for i in np.arange(1,Nx+1,1):
            f[i,j] = 0.125 * rho * sig_x * sig_y * (i-0.5)*dx * (j-0.5)*dy * ( w[k,i,j] - w[k,i,j-1]-w[k,i-1,j]+w[k,i-1,j-1] )/(dx*dy) + w[k,i-1,j-1]/dt
        w_hat[:Nx,j-1] = np.linalg.solve(Ax,f[1:,j]) # w[k+0.5] 
        
    #Boundary condition for w_hat (or w[k+0.5])
    for j in np.arange(0,Ny+1,1):
        w_hat[0,j] = 2*w_hat[1,j] - w_hat[2,j]
        w_hat[Nx,j] = 2*w_hat[Nx-1,j] - w_hat[Nx-2,j]
    for i in np.arange(0,Nx+1,1):
        w_hat[i,0] = 2*w_hat[i,1] - w_hat[i,2]
        w_hat[i,Ny] = 2*w_hat[i,Ny-1] - w_hat[i,Ny-2]      
                      
    for i in np.arange(1,Nx+1,1):
        for j in np.arange(1,Ny+1,1):
            g[i,j] = 0.125 * rho * sig_x * sig_y * (i-0.5)*dx * (j-0.5)*dy * ( w_hat[i,j]-w_hat[i,j-1]-w_hat[i-1,j]+w_hat[i-1,j-1] )/(dx*dy) + w_hat[i-1,j-1]/dt
        w[k+1,i-1,:Ny] = np.linalg.solve(Ay,g[i,1:])
         
    # boundary condition for w[k+1]        
    for j in np.arange(0,Ny+1,1):
        w[:,0,j] = 2*w[:,1,j] - w[:,2,j]
        w[:,Nx,j] = 2*w[:,Nx-1,j] - w[:,Nx-2,j]
    for i in np.arange(0,Nx+1,1):
        w[:,i,0] = 2*w[:,i,1] - w[:,i,2]
        w[:,i,Ny] = 2*w[:,i,Ny-1] - w[:,i,Ny-2]        

W0 = w[300]
#================================================================================================================================================================================================================     

# Knock-In event case
        
for k in np.arange(Nt):
    f = np.zeros((Nx+1,Ny+1))
    v_hat = np.zeros((Nx+1,Ny+1))
    g = np.zeros((Ny+1,Ny+1))
    if k%pp==0:
        v[k,round(x0*K[-int(k/pp)-1]/dx):,round(y0*K[-int(k/pp)-1]/dy):] = F*(1+c[-int(k/pp)-1])

    for j in np.arange(1,Ny+1,1):
        for i in np.arange(1,Nx+1,1):
    #       f[i,j] = 0.5 * rho * sig_x * sig_y * i*dx * i*dy * ( w[0,i+1,j+1] - w[0,i+1,j]-w[0,i,j+1]+w[0,i,j] )/((dx**2)) + w[0,i,j]/dt
            f[i,j] = 0.125 * rho * sig_x * sig_y * (i-0.5)*dx * (j-0.5)*dy * ( v[k,i,j] - v[k,i,j-1]- v[k,i-1,j]+ v[k,i-1,j-1] )/(dx*dy) + v[k,i-1,j-1]/dt
        v_hat[:Nx,j-1] = np.linalg.solve(Ax,f[1:,j]) # w[k+0.5]   
        
    #Boundary condition for w_hat (or w[k+0.5])
    for j in np.arange(0,Ny+1,1):
        v_hat[0,j] = 2*v_hat[1,j] - v_hat[2,j]
        v_hat[Nx,j] = 2*v_hat[Nx-1,j] - v_hat[Nx-2,j]
    for i in np.arange(0,Nx+1,1):
        v_hat[i,0] = 2*v_hat[i,1] - v_hat[i,2]
        v_hat[i,Ny] = 2*v_hat[i,Ny-1] - v_hat[i,Ny-2]
                            
    for i in np.arange(1,Nx+1,1):
        for j in np.arange(1,Ny+1,1):
    #       g[i,j] = 0.5 * rho * sig_x * sig_y * i*dx * i*dy * ( w_hat[i+1,j+1] - w_hat[i+1,j]-w_hat[i,j+1]+w_hat[i,j] )/((dx*dy)) + w_hat[i,j]/dt
            g[i,j] = 0.125 * rho * sig_x * sig_y * (i-0.5)*dx * (j-0.5)*dy * ( v_hat[i,j]- v_hat[i,j-1]-v_hat[i-1,j]+v_hat[i-1,j-1] )/(dx*dy) + v_hat[i-1,j-1]/dt
        v[k+1,i-1,:Ny] = np.linalg.solve(Ay,g[i,1:])
        
    # boundary condition for w[k+1]        
    for j in np.arange(0,Ny+1,1):
        v[:,0,j] = 2*v[:,1,j] - v[:,2,j]
        v[:,Nx,j] = 2*v[:,Nx-1,j] - v[:,Nx-2,j]
    for i in np.arange(0,Nx+1,1):
        v[:,i,0] = 2*v[:,i,1] - v[:,i,2]
        v[:,i,Ny] = 2*v[:,i,Ny-1] - v[:,i,Ny-2]             
   
V0 = v[300]        
#============================================================================================================================================================================================================
# Estimating Probability of hitting barrier by Monte Carlo simulation
'''
mc = 10000      #simulation number
count = 0

for i in np.arange(mc):
    ep = np.random.multivariate_normal([0,0],[[1,rho],[rho,1]],nt)
    ep1 = ep[:,0]
    ep2 = ep[:,1]
            
    #log-stock-price
    log_x = [np.log(x0)]
    log_y = [np.log(y0)]
    for i in np.arange(len(ep)):
        log_x.append( log_x[i] + (mu_x - 0.5 *(sig_x**2))*dt + sig_x*np.sqrt(dt)*ep1[i])
        log_y.append(log_y[i] + (mu_y - 0.5 *(sig_y**2))*dt + sig_y*np.sqrt(dt)*ep2[i])
        
    asset1 = np.array(np.exp(log_x))
    asset2 = np.array(np.exp(log_y))
    
    if min(asset1/x0) <= ki or min(asset2/y0)<=ki:
        count = count + 1
    else:
        continue

b = count/mc
'''
#============================================================================================================================================================================================================
# ELS price matrix and 3D-surface plot

B = 0.16091218253968254  # Probability of hitting barrier given in the investment information
els_px = W0*(1-B) + V0*B 
Els_price = w*(1-B) + v*B      
final_price = els_px[50,50]
x = np.zeros(Nx+1)
y = np.zeros(Ny+1)
for i in np.arange(Nx+1):
    x[i] = i*dx
for j in np.arange(Ny+1):
    y[j] = j*dy


xx , yy = np.meshgrid(x,y)
zz = els_px
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot_surface(xx,yy,zz,cmap=cm.jet)
plt.show()  

#============================================================================================================================================================================================================
# Delta Calculation and Graph
#%%
delta_x = np.zeros((Nt,Nx+1,Ny+1))
delta_y = np.zeros((Nt,Nx+1,Ny+1))
for i in range(Nt):
    a = pd.DataFrame(Els_price[i,:,:])
    b = pd.DataFrame(Els_price[i,:,:])
    delta_x[i,:,:] = ((a-a.shift(axis=0))/dx)/100
    delta_y[i,:,:] = ((b-b.shift(axis=1))/dy)/100
    
x = np.linspace(0,xmax,Nx+1);y = np.linspace(0,ymax,Ny+1);
#tau  = np.linspace(0,T,Nt)
#fig2 = plt.figure()
#plt.plot(x[1:],delta_x[0,1:,50]);
plt.plot(x[1:],delta_x[0,1:,50])
plt.plot(x[1:],delta_x[50,1:,51])
plt.plot(x[1:],delta_x[100,1:,51])
plt.plot(x[1:],delta_x[150,1:,51])
plt.plot(x[1:],delta_x[200,1:,51])
plt.plot(x[1:],delta_x[250,1:,51])
plt.plot(x[1:],delta_x[-1,1:,51])
#plt.xlabel('KOSPI')
#plt.ylabel('delta')
##plt.axvline(x=K0[0]*KI, color ='orange')
##K = [0.90, 0.90, 0.85, 0.85, 0.80,0.8];
plt.show()
#fig3 = plt.figure()
#plt.plot(y[1:],delta_y[0,1:,50])
#plt.plot(y[1:],delta_y[50,1:,50])
#plt.plot(y[1:],delta_y[100,1:,50])
#plt.plot(y[1:],delta_y[150,1:,50])
#plt.plot(y[1:],delta_y[200,1:,50])
#plt.plot(y[1:],delta_y[250,1:,50])
#plt.plot(y[1:],delta_y[-1,1:,50])

#plt.xlabel('EuroStoxx')
#plt.ylabel('delta')
#plt.axvline(x=K0[1]*KI, color ='orange')
#plt.show()
#%%
under_data = pd.read_excel("ELS_Hedge_data_price.xlsx", sheetname = 0, header=8, index_col=0)
under_data = under_data.reindex(index = under_data.index[5:])
under_data.index=pd.to_datetime(under_data.index, format='%Y-%m-%d')

under_p = under_data.loc["2018-02-13":].fillna(0)
xInd = np.around(list(under_p.iloc[:,0]/under_p.iloc[0,0]*50))-1
yInd = np.around(list(under_p.iloc[:,1]/under_p.iloc[0,1]*50))-1

delta_x = np.zeros((Nt,Nx+1,Ny+1))
delta_y = np.zeros((Nt,Nx+1,Ny+1))
for i in range(Nt):
    a = pd.DataFrame(Els_price[i,:,:])
    b = pd.DataFrame(Els_price[i,:,:])
    delta_x[i,:,:] = ((a-a.shift(axis=0))/dx)/100
    delta_y[i,:,:] = ((b-b.shift(axis=1))/dy)/100

#deltaforX = []
#for j in range(pp):
#    deltaforX.append(delta_x[-(j+1),int(xInd[j]),int(yInd[j])])
#
#deltaforY = []
#for j in range(pp):
#    deltaforY.append(delta_y[-(j+1),int(xInd[j]),int(yInd[j])])
    
#pd.DataFrame(deltaforX).plot()
#pd.DataFrame(deltaforY).plot()
    
plt.plot(x,delta_x)
plt.xlabel('EUROSTOXX50')
plt.ylabel('Delta')
plt.title('Delta With Respect to EUROSTOXX50')
plt.show()
plt.plot(y,delta_y)
plt.xlabel('HSCEI')
plt.ylabel('Delta')
plt.title('Delta With Respect to HSCEI')
plt.show()

# Gamma Calculation and Graph
gamma_x=np.zeros(Nx+1)
gamma_y=np.zeros(Ny+1)

for i in range(1,Nx,1):
    gamma_x[i]=(els_px[i+1,50]-2*els_px[i,50]+els_px[i-1,50])/dx**2
    gamma_y[i]=(els_px[50,i+1]-2*els_px[50,i]+els_px[50,i-1])/dy**2

plt.plot(x,gamma_x)
plt.xlabel('EUROSTOXX50')
plt.ylabel('Gamma')
plt.title('Gamma With Respect to EUROSTOXX50')
plt.show()
plt.plot(y,gamma_y)
plt.xlabel('HSCEI')
plt.ylabel('Gamma')
plt.title('Gamma With Respect to HSCEI')
plt.show()
#%%
#deltaforX = []
#deltaforY = []
#exactPrice = []
#for j in range(pp):
#    deltaforX.append(delta_x[-(j+1),int(xInd[j]),int(yInd[j])])
#    deltaforY.append(delta_y[-(j+1),int(xInd[j]),int(yInd[j])])
#    exactPrice.append(Els_price[-(j+1),int(xInd[j]),int(yInd[j])])
#pd.Series(deltaforX,index = under_p.index).to_csv("deltaX.csv")
#pd.Series(deltaforY,index = under_p.index).to_csv("deltaY.csv")
#pd.Series(exactPrice,index = under_p.index).to_csv("Price.csv")
pd.Series(deltaforX).plot()
pd.Series(deltaforY).plot()
#%%