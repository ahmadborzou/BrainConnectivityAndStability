import os
import time
import pickle
import numpy as np
import pandas as pd

from matplotlib import cm
import matplotlib.pyplot as plt
labelsize=16

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity




y_min = 5.
####################################################
## return X, y for a specific neuron 
def GetData(label,y_min):
  X = np.array(df[cols[1:]])[0:-1,:] ## states at t
  y =  np.array(df[label])[1:]       ## states at t+1
  ## drop the noise. Any T is the solution when X=y=0
  X[y<y_min] = 0
  y[y<y_min] = 0
  return X, y
####################################################
## remove noise
def RemNoise(x,cut=y_min):
  if x<cut:
    return 0 
  else:
    return x
####################################################
doPlot = True

if os.path.exists("Figures") == False:
  os.makedirs("Figures")

################
## read the data
################
mouse = "C4" # C4 or C7
df = pd.read_csv(f"data/{mouse}_Final_denoised_cell_traces.csv")
df = df.drop(index=0)
cols = list(df.columns)
cols[0]='t'
df.columns=cols
################
################

## change the type to float 
df = df.astype('float64')


## noise removal
for col in cols[1:]:df[col]=df[col].apply(RemNoise)

## Data
## 1st half houre is control 2nd half hour is test
## choose control.
t_half = 2000. ## check this for each experiment
exp = input("b: before injection. a: after injection. Insert a or b and hit eneter\n")
if exp == "a":
  df = df[df['t']>t_half]
elif exp == "b":
  df = df[df['t']<t_half]
else:
  raise(Exception("input should be a or b."))

## plot intensites
fig, ax = plt.subplots()
ax.hist(np.array(df[cols[1:]]).flatten(), bins=25,color='olivedrab')
ax.hist(np.array(df[cols[1:]]).flatten(), histtype='step', bins=25,color='k')
ax.set_xlabel(r"C$^{2+}$ intensity")
ax.set_ylabel("# of measurements")
ax.tick_params(labelsize=labelsize)
ax.vlines(5,0,1.6e6, color='r')
ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax.set_yscale("log")
plt.tight_layout()
plt.show()
if exp == "a":
  fileName = "Distribution_Observed_C2_intensity_After.png"
elif exp == "b":
  fileName = "Distribution_Observed_C2_intensity_Before.png"
fig.savefig(fileName,dpi=200)



## plot calcium intensity change over time
fig, ax = plt.subplots()
ax.scatter(df[cols[0]], df[cols[1]],marker='.',c='r')
ax.plot(df[cols[0]], df[cols[1]],color='darkblue')
#ax.set(xlim=(0,125),ylim=(0,20))
ax.set_xlabel(r"time (s)")
ax.set_xlabel(r"time")
ax.set_ylabel(r"$C^{2+}$ intensity")
ax.tick_params(labelsize=labelsize)
plt.tight_layout()
plt.show()
if exp == "a":
  fileName = "C2Intensity_time_After.png"
elif exp == "b":
  fileName = "C2Intensity_time_Before.png"
fig.savefig(fileName,dpi=200)



## number of features
Nf = len(cols[1:])


# print("\n\ny_min: %g. Ok?\n\n"%(y_min))
Tij       = np.zeros((Nf,Nf))
Ci        = np.zeros(Nf)
MSE_train = np.zeros(Nf)
MSE_test  = np.zeros(Nf)
Err_train = []
Err_test = []

## data
for ilabel, label in enumerate(cols[1:]):

  #if label != ' C03':continue

  X, y = GetData(label,y_min)

  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=np.random.RandomState(0))

## linear regression
  reg = LinearRegression().fit(X_train, y_train)

  ## set the interactions
  Tij[ilabel,:] = reg.coef_
  Ci[ilabel] = reg.intercept_

## plot the regression
  if ilabel==0:
    fig, ax = plt.subplots()
    ax.scatter(reg.predict(X_train),y_train,marker='.',color='b',alpha=1,label='train')
    ax.scatter(reg.predict(X_test),y_test,marker='.',color='r',alpha=1,label='test')
    ax.set_xlabel(r'$\hat{V}_i$')
    ax.set_ylabel(r'$V_i$')
    ax.set_xlim(0.9*y_min,1.1*y.max())
    ax.set_ylim(0.9*y_min,1.1*y.max())
    ax.tick_params(labelsize=16) 
    ax.legend()
    plt.tight_layout()
    #plt.show()
    if exp == "a":
      filename = "Figures/%s_ClassicHopfieldFit_After.png"%(label.replace(" ",""))
    elif exp == "b":
      filename = "Figures/%s_ClassicHopfieldFit_Before.png"%(label.replace(" ",""))
    fig.savefig(filename,dpi=200)
    

  
  MSE_train[ilabel] = mean_squared_error(y_train, reg.predict(X_train))
  MSE_test[ilabel]  = mean_squared_error(y_test, reg.predict(X_test))
  mask = y_train>0
  Err_train += list(abs(y_train - reg.predict(X_train))[mask]/y_train[mask])
  mask = y_test>0
  Err_test  += list(abs(y_test - reg.predict(X_test))[mask]/y_test[mask])



fig, ax = plt.subplots()
violin_parts = ax.violinplot(Err_train, vert=False, showmedians=True)
for vp in violin_parts['bodies']:
    vp.set_facecolor('r')
    vp.set_edgecolor('k')
    vp.set_linewidth(1)
    vp.set_alpha(0.5)
ax.set_ylabel("Distribution") 
ax.set_xlabel(r"$\frac{| V_i - \hat{V}_i |}{V_i}$")
ax.set_yticks([])
plt.tight_layout()
plt.show()
if exp == "a":
  fileName = "ErrorDistribution_After.png"
elif exp == "b":
  fileName = "ErrorDistribution_Before.png"
fig.savefig(fileName,dpi=200)


## save the results for next step
if exp == "a":
  fileName = "X_T_AfterInj.pickle"
elif exp == "b":
  fileName = "X_T_BeforeInj.pickle"
with open(fileName,"wb") as fi:
  pickle.dump([X, Ci, Tij], fi)




