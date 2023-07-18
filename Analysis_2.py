import time
import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


doPlot = True


## import the X and Tij befor and after injection
with open("X_T_BeforeInj.pickle","rb") as fi:
  Xbe, Cbe, Tbe = pickle.load(fi)
##
with open("X_T_AfterInj.pickle","rb") as fi:
  Xaf, Caf, Taf = pickle.load(fi)
##

Tbe_nonDiag = Tbe - np.diag(np.diag(Tbe))
Taf_nonDiag = Taf - np.diag(np.diag(Taf))

Tbe = Tbe_nonDiag
Taf = Taf_nonDiag

## number of neurons
Nf = Xbe.shape[1] 


if doPlot:
## plot T
  fig, ax = plt.subplots()
  im = ax.imshow(Tbe,cmap=cm.seismic,vmin=-0.8,vmax=0.8)
  ##ax.set_title("T before injection")
  ax.invert_yaxis()
  fig.colorbar(im)
  ax.set_xlabel("Neurons")
  ax.set_ylabel("Neurons")
  plt.tight_layout()
  plt.show()
  fig.savefig("T_before_injection.png",dpi=200)

  fig, ax = plt.subplots()
  im = ax.imshow(Taf,cmap=cm.seismic,vmin=-0.8,vmax=0.8)
  ##ax.set_title("T after injection")
  ax.invert_yaxis()
  fig.colorbar(im)
  ax.set_xlabel("Neurons")
  ax.set_ylabel("Neurons")
  plt.tight_layout()
  plt.show()
  fig.savefig("T_after_injection.png",dpi=200)

  fig, ax = plt.subplots()
  ax.plot(Cbe, 'r*')
  ax.set_xlabel("Neurons")
  ax.set_ylabel(r"$V_i^{(\mathrm{ext.})}$")
  ax.set(ylim=(-0.8,0.8))
  plt.tight_layout()
  plt.show()
  fig.savefig("ExternalV_before_injection.png",dpi=200)

  fig, ax = plt.subplots()
  ax.plot(Caf, 'r*')
  ax.set_xlabel("Neurons")
  ax.set_ylabel(r"$V_i^{(\mathrm{ext.})}$")
  ax.set(ylim=(-0.8,0.8))
  plt.tight_layout()
  plt.show()
  fig.savefig("ExternalV_after_injection.png",dpi=200)


##################
## Control neurons

def Vext(t):
  ######
  i_idx, j_idx = np.where(abs(Tbe_nonDiag)>0.1)
  ans  = np.zeros(Nf)
  indx = np.unique(i_idx) #[3,5]
  ans[indx] = 1000.

  return ans 

## initial voltage and final time
V0 = np.zeros(Nf)
n_final = 15

## reproduce
V = np.copy(V0)

def V_future(n, V0):
  dt = 1. ## in units of (0.1 s)
  add  = sum([np.dot(np.linalg.matrix_power(Tbe_nonDiag,k), Vext((n-k-1)*dt) ) for k in range(n)])
  V = np.dot( np.linalg.matrix_power(Tbe_nonDiag,n), V0) + add
  V[V<0] = 0.
  return V 


########################
def Vext(t):
  ######
  i_idx, j_idx = np.where(abs(Tbe_nonDiag)>0.1)
  ans  = np.zeros(Nf)
  indx = np.unique(i_idx)
  print(f"nodes that recieved external input: {indx}")
  ans[indx] = 1000.
  return ans 

## initial voltage and final time
V = np.zeros(Nf)
Vt_arr = []
t_arr = np.arange(0,8)
for k in t_arr:
  print(f"k: {k}")
  first = np.dot(Tbe_nonDiag, V)
  second = Vext(k)
  V = first + second
  V[V<0] = 0.
  Vt_arr.append(V)
Vt_arr = np.array(Vt_arr)

#plot strongest and weakest signals that did not recieve external input
idx = np.argsort(Vt_arr[-1,:])
if doPlot:
  fig, ax = plt.subplots()
  for n in list(np.flip(idx[-5:-2]))+list(np.flip(idx[0:3])): 
    ax.plot(Vt_arr[:,n], label=f"neuron: {n}")
  ax.set_xlabel('time')
  ax.set_ylabel('current')
  ax.set(xticks=[0],yticks=[0])
  ax.legend()
  plt.tight_layout()
  plt.show()
  fig.savefig("Exciting_Neurons_6_22.png")

## plot the external voltage
  fig, ax = plt.subplots()
  ax.plot([1000]*len(t_arr), label='neurons: 6, 22')
  ax.plot([0]*len(t_arr), label='other neurons')
  ax.set_xlabel('time')
  ax.set_ylabel(r'V$^{\mathrm{ext.}}$')
  ax.set(xticks=[0],yticks=[0])
  ax.legend()
  plt.tight_layout()
  plt.show()

####
####
####
def Vext(t):
  ######
  i_idx, j_idx = np.where(abs(Tbe_nonDiag)>0.1)
  ans  = np.zeros(Nf)
  indx = np.unique(i_idx)
  print(f"nodes that recieved external input: {indx}")
  ans[indx] = 1.
  if t > 1:
    ans = np.zeros(Nf)
  return ans 

## initial voltage and final time
V = np.zeros(Nf)
Vt_arr = []
Vext_arr = []
for k in np.arange(0,8):
  print(f"k: {k}")
  first = np.dot(Tbe_nonDiag, V)
  second = Vext(k)
  V = first + second
  V[V<0] = 0.
  Vt_arr.append(V)
  Vext_arr.append(Vext(k))
Vt_arr = np.array(Vt_arr)
Vext_arr = np.array(Vext_arr)

if doPlot:
#plot strongest and weakest signals that did not recieve external input
  fig, ax = plt.subplots()
  for n in list(np.flip(idx[-5:-2]))+list(np.flip(idx[0:3])): 
    ax.plot(np.arange(len(Vt_arr[:,n]))/10.,Vt_arr[:,n], label=f"neuron: {n}")
  #ax.set_xlabel('time (s)')
  ax.set_xlabel('time')
  ax.set_ylabel('current')
  #ax.set(xticks=[0],yticks=[0])
  ax.legend()
  plt.tight_layout()
  plt.show()
  fig.savefig("TemporaryExciting_Neurons_6_22.png")
## plot the external voltage
  fig, ax = plt.subplots()
  ax.plot(np.arange(len(Vext_arr[:,6]))/10.,Vext_arr[:,6], label='neurons: 6, 22')
  ax.plot(np.arange(len(Vext_arr[:,6]))/10.,[0]*len(t_arr), label='other neurons')
  #ax.set_xlabel('time (s)')
  ax.set_xlabel('time')
  ax.set_ylabel(r'V$^{\mathrm{ext.}}$')
  #ax.set(xticks=[0],yticks=[0])
  ax.legend()
  plt.tight_layout()
  plt.show()
  fig.savefig("ExternalVoltage_TemporaryExciting_Neurons_6_22.png")

####
####
####
def Vext(t):
  ans = np.ones(Nf)*abs(np.sin(3*t))
  if t > 10:
    ans = np.zeros(Nf)
  return ans 

## initial voltage and final time
V = np.zeros(Nf)
Vt_arr = []
Vext_arr = []
for k in np.arange(0,16):
  print(f"k: {k}")
  first = np.dot(Tbe_nonDiag, V)
  second = Vext(k)
  V = first + second
  V[V<0] = 0.
  Vt_arr.append(V)
  Vext_arr.append(Vext(k))
Vt_arr = np.array(Vt_arr)
Vext_arr = np.array(Vext_arr)

#plot strongest and weakest signals that did not recieve external input
idx = np.argsort(Vt_arr[-1,:])
if doPlot:
  fig, ax = plt.subplots()
  for n in idx:
    ax.plot(np.arange(len(Vt_arr[:,n]))/10.,Vt_arr[:,n], label=f"neuron: {n}")
  #ax.set_xlabel('time (s)')
  ax.set_xlabel('time')
  ax.set_ylabel('current')
  #ax.set(xticks=[0],yticks=[0])
  plt.tight_layout()
  plt.show()
  fig.savefig("Exciting_all_Neurons.png")

  ## plot the external voltage
  fig, ax = plt.subplots()
  ax.plot(np.arange(len(Vext_arr[:,0]))/10.,Vext_arr[:,0], label='all neurons')
  #ax.set_xlabel('time (s)')
  ax.set_xlabel('time')
  ax.set_ylabel(r'V$^{\mathrm{ext.}}$')
  #ax.set(xticks=[0],yticks=[0])
  ax.legend()
  plt.tight_layout()
  plt.show()
  fig.savefig("ExternalVoltage_Exciting_all_Neurons.png")






#######################################################
#######################################################
## construct the firing states in the observed activity
cut = 5
Vbe = np.zeros_like(Xbe) 
Vbe[Xbe>cut] = 1
Vaf = np.zeros_like(Xaf)
Vaf[Xaf>cut] = 1

## magnetization
def Magnet(V):
  m = np.mean(V,axis=1) 
  m = m[m>0]
  plt.hist(m);plt.yscale("log");plt.xlabel("m");plt.ylabel("Probability");plt.show() 
  return m


m_be = Magnet(Vbe)
m_af = Magnet(Vaf)


if doPlot:
## plot for publication
  fig, ax = plt.subplots()
  ax.hist(m_be,color='red',label='before',weights=[len(m_af)/len(m_be)]*len(m_be))
  ax.hist(m_af,color='green',label='after',alpha=0.5)
  ax.set_xlabel("m")
  ax.set_ylabel("number of data points")
  ax.legend()
  plt.tight_layout()
  plt.show()
  fig.savefig("m_distribution.png")


  t0_af = 2084.152000 ## start time of the after injection experiment.
  fig, ax = plt.subplots()
  ax.scatter(np.arange(len(np.mean(Vbe,axis=1)))/10.,       np.mean(Vbe,axis=1),color='red',label='before',marker='.')
  ax.scatter(np.arange(len(np.mean(Vaf,axis=1)))/10.+t0_af, np.mean(Vaf,axis=1),color='green',label='after',marker='.')
  ax.set_ylabel("m")
  #ax.set_xlabel("time (s)")
  ax.set_xlabel('time')
  ax.legend()
  plt.tight_layout()
  plt.show()
  fig.savefig("m_t.png")





## maximum likelihood estimation of gaussian parameters
mu_be = np.mean(m_be) 
mu_af = np.mean(m_af) 
sig_be = np.std(m_be)
sig_af = np.std(m_af)




## fit the parameters of the distribution function
def fit(m_):
  ## this range is quite sensitive
  ## range is good for gaussian distribtuin
  m_linspace = np.linspace(-2.7,2.7,100000)

  ## normalize such that the range of m_linspace is where the probability is high
  m_norm = (m_ - np.mean(m_))/np.std(m_)

  ## determine the best bandwidth
  ## use grid search cross-validation to optimize the bandwidth
  params = {'bandwidth': np.linspace(0.1, 0.5, 10)}
  grid = GridSearchCV(KernelDensity(), params, cv=2)
  grid.fit(m_norm.reshape(-1,1))
  print(grid.best_params_) 

  ## use a gaussian kernel density esstimator to find single-body phase-space density
  kde   = KernelDensity(kernel='gaussian', bandwidth=grid.best_params_['bandwidth']).fit(m_norm.reshape(-1,1))

  Log_f = kde.score_samples(m_linspace.reshape(-1,1)) 
  return np.polyfit(m_linspace,-Log_f,deg=4)  


coef_be = fit(m_be)
coef_af = fit(m_af)
with open("FitCoefficients.pickle","wb") as fi:
  pickle.dump([coef_be, coef_af],fi)
with open("FitCoefficients.pickle","rb") as fi:
  coef_be, coef_af = pickle.load(fi)


def convert_to_nonScale(mu,sig,coef):
  c4, c3, c2, c1, c0 = coef
  a = np.zeros_like(coef)
  a[0] = c0 + (c4*mu**4)/sig**4 - (c3*mu**3)/sig**3 + (c2*mu**2)/sig**2 - (c1*mu)/sig
  a[1] = (-4*c4*mu**3 + 3*c3*mu**2*sig - 2*c2*mu*sig**2 + c1*sig**3)/sig**4
  a[2] = (6*c4*mu**2 - 3*c3*mu*sig + c2*sig**2)/sig**4
  a[3] = (-4*c4*mu + c3*sig)/sig**4
  a[4] = c4/sig**4
  return a

a_be = convert_to_nonScale(mu_be,sig_be,coef_be)
a_af = convert_to_nonScale(mu_af,sig_af,coef_af)

@np.vectorize
def f_m(m):
  f_be = 0.
  f_af = 0.
  for i in range(len(a_be)):
    f_be += a_be[i]*m**i
    f_af += a_af[i]*m**i
  return f_be, f_af

m_linspace = np.linspace(-2.7,3.45,1000)
f_be, f_af = f_m(m_linspace)

#if doPlot:
## plot for publication
fig, ax = plt.subplots()
ax.plot(m_linspace,f_be,color='red',label='before')
ax.plot(m_linspace,f_af,color='green',label='after')
ax.set_xlabel("m")
ax.set_ylabel("F[m]")
ax.legend()
plt.tight_layout()
plt.show()
fig.savefig("F_m.png")





