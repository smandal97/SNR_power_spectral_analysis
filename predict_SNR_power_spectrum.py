#!/usr/bin/env python
# coding: utf-8

# run with the files 'l0_gam***.dat' in the same directory


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

#user parameters
M   = 1.0          #mass of progenitor in units of Chandrashekhar mass or 1.4M_sun
E   = 1.0          #explosion energy in units of 10^51 ergs
rho = 1.0          #ISM density in units of 1.67*10_24 g/cm^3 (or 1 H atom per cm^3)
T   = 100.         #remnant age in years
gam_mode = 1       #sets value of adiabatic index gamma, can be 1,2 or 3 for gamma = 5/3, 3/2, and 4/3 respectively
 
#dynamical age (tau=t/T in paper; see eqn 8)
T_dyn = (T/563.) * M**(-5./6.) * E**(1./2.) * rho**(1./3.)
print(T_dyn)

#l0,n1 and n2 data from hydro runs
if(gam_mode==1):
    data = np.loadtxt('l0_gam5o3.dat')
elif(gam_mode==2):
    data = np.loadtxt('l0_gam3o2.dat')
elif(gam_mode==3):
    data = np.loadtxt('l0_gam4o3.dat')
else:
    print("Error!")
    
#interpolate
logt_data = -4.+data[:,0]/10.
l0_data   = data[:,2]
n1_data   = data[:,3]
n2_data   = -data[:,4]

l0_interp = interp1d(logt_data, l0_data)
n1_interp = interp1d(logt_data, n1_data)
n2_interp = interp1d(logt_data, n2_data)

#calculate values of l0, n1 and n2
l0 = l0_interp(np.log10(T_dyn))
n1 = n1_interp(np.log10(T_dyn))
n2 = n2_interp(np.log10(T_dyn))


#produce power spectrum
l  = np.logspace(0.5,2.5,100)
Cl = ( (l/l0)**(-n1) + (l/l0)**(-n2) )**-1.







