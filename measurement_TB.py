#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 16:01:39 2018

@author: hugo

BEFORE USE : 
    
    Make sure you have executed the get_CO_surf.py script prior to this one and have .co_surf files in your directory
    
    Make sure you modified every variable that depend on the object you study. Quick way to find them : research TO BE MODIFIED in the script

"""
# =============================================================================
# Packages
# =============================================================================

from astropy import constants as const
from astropy.io import fits # Reading fits files
import pickle
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
import numpy as np
from copy import copy
from scipy.optimize import curve_fit
from get_CO_surf import storage
import scipy.stats as st
# =============================================================================
# Constants
# =============================================================================

G=const.G.value
M_sun=const.M_sun.value
c=const.c.value
au=const.au.value
hp=const.h.value
kB=const.k_B.value
arcsec=4.848136811095e-06
# =============================================================================
# Functions
# =============================================================================

def r(x,yf,yb):
    """ Radius of the orbit. y can either be the y coordinate of a front or of a back side.
    Units : x, yf, yb : pixel
            r : [au]
    """
    yc=(yf+yb)/2
    return ((x-xs)**2+((yf-yc)/np.cos(inc))**2)**0.5*px_size*D

def posmax(L):
    """Returns the index of the maximum of a list, which can contain None type"""
    idx=0
    for i in range(len(L)):
        try:
            if L[i] > L[idx]:
                idx=i
        except TypeError:
            pass
    return idx

def flux_to_Tbrigh(F, wl, BMAJ, BMIN):
    """
         Convert Flux density in Jy to brightness temperature [K]
     Flux [Jy]
     wl [m]
     BMAJ, BMIN in [deg], ie as in fits header

     T [K]
     """
    nu = c/wl 
    factor = 1e26 # 1 Jy = 10^-26 USI
    conversion_factor = (BMIN * BMAJ * (3600*arcsec)**2 * np.pi/4/np.log(2)) # beam 
    #F = factor *  conversion_factor * 2.*hp/c**2 * nu^3/ (np.exp(hp * nu / (kB * T)) -1.) 
    exp_m1 = factor *  conversion_factor * (2*hp*nu**3)/(F*c**2)
#    hnu_kT =  np.log(max(exp_m1,1e-10) + 1)
#    T = hp * nu / (hnu_kT * kB)
    if exp_m1 >0:
        hnu_kT =  np.log(exp_m1 + 1)
        T = hp * nu / (hnu_kT * kB)
    else:
        T=None
    
    return T 

# =============================================================================
# Data
# =============================================================================


path="/home/hugo/Documents/Stage/selection_objets/HD163296/Itziar/"     ######### /!\ TO BE MODIFIED FOR EACH OBJECT
fits_name = path+"HD163296_CO3-2.fits.gz"    ######### /!\ TO BE MODIFIED FOR EACH OBJECT
pc2au=648000/np.pi #parsec to au
deg2rad=np.pi/180 

D= 122 # (pc)                                                   ######### /!\ TO BE MODIFIED FOR EACH OBJECT
inc=38 *deg2rad # (rad)                                       ######### /!\ TO BE MODIFIED FOR EACH OBJECT


fh=fits.open(fits_name)
CDELT1=fh[0].header['CDELT1'] 
CO=fh[0].data
CO=CO[0]
nx=len(CO[0][0]) # number of columns
ny=len(CO[0]) # number of lines
nv=len(CO) #number of chanels
px_size=abs(CDELT1)*3600 #arcsec
restfreq=fh[0].header['RESTFRQ'] #freq of the transition studied
CRVAL3=fh[0].header['CRVAL3'] #frequency of the 1st channel
CDELT3=fh[0].header['CDELT3'] #freq step
BMIN=fh[0].header['BMIN'] # [deg] Beam major axis length
BMAJ=fh[0].header['BMAJ'] # [deg] Beam minor axis length
wl=c/CRVAL3
fh.close()

ext = ["_sup_back","_sup_front","_inf_back" ,"_inf_front"]

with open(fits_name+ext[0]+".co_surf", 'rb') as handle:
    data_sup_back=pickle.load(handle)
with open(fits_name+ext[1]+".co_surf", 'rb') as handle: 
    data_sup_front=pickle.load(handle)
with open(fits_name+ext[2]+".co_surf", 'rb') as handle:
    data_inf_back=pickle.load(handle)
with open(fits_name+ext[3]+".co_surf", 'rb') as handle:
    data_inf_front=pickle.load(handle)

(xs,ys)=data_sup_back.star_center # coordinates of the star's center /!\ IN THE CROPPED IMAGE /!\
PA=data_sup_back.PA # (deg)
ni=data_sup_back.ni # 1st interesting chan
nm=data_sup_back.nm # 0 velocity chan
nf=data_sup_back.nf # last interesting chan
xw0, yw0 = data_sup_back.window[0] # coordinates of the selected window 
xw1, yw1 = data_sup_back.window[1]

freq= CRVAL3 + CDELT3*np.arange(nv) # freq of the channels
freq=freq[ni:nf] # only the selected chan
v_syst= -(freq[nm-ni]-restfreq)*c/restfreq # global speed of the system
v_obs = -((freq-restfreq)/restfreq)*c-v_syst # radial velocity of the channels

pos_max_sup_back=np.array(data_sup_back.pos_maxima)
pos_max_sup_front=np.array(data_sup_front.pos_maxima)
pos_max_inf_back=np.array(data_inf_back.pos_maxima)
pos_max_inf_front=np.array(data_inf_front.pos_maxima)

n=len(pos_max_sup_back)

CO=data_sup_front.CO # Same image than in the repvious script 

# =============================================================================
# Temperature of brigtness
# =============================================================================
r_sup=[]
r_inf=[]
Tb_sup_front=[]
Tb_sup_back=[]
Tb_inf_front=[]
Tb_inf_back=[]
for i in range(n):
    if i <= 18 or i >= 36 : # removing the chans where the arms were too vertical               ######### /!\ TO BE MODIFIED FOR EACH OBJECT
        v0=v_obs[i]
        # --Sup surface:
        try:
            xmax=min(pos_max_sup_back[i][-1][0],pos_max_sup_front[i][-1][0]) # 
            xmin=max(pos_max_sup_back[i][0][0],pos_max_sup_front[i][0][0])
        except IndexError:
            r_sup.append(False)
            Tb_sup_front.append(False)
            Tb_sup_back.append(False)

        else:
            yb=np.array([])
            yf=np.array([])
            T_sup_f=np.array([])
            T_sup_b=np.array([])
            x=np.array([xmin+i for i in range(0,xmax-xmin)])
            for k in range(len(x)):
                yb=np.append(yb, pos_max_sup_back[i][k][1])
                yf=np.append(yf, pos_max_sup_front[i][k][1])
                T_sup_f=np.append(T_sup_f,flux_to_Tbrigh(CO[i][pos_max_sup_front[i][k][0],pos_max_sup_front[i][k][1]],wl,BMAJ,BMIN))
                T_sup_b=np.append(T_sup_b,flux_to_Tbrigh(CO[i][pos_max_sup_back[i][k][0],pos_max_sup_back[i][k][1]],wl,BMAJ,BMIN))
            r_sup.append(copy(r(x,yf,yb)))
            Tb_sup_front.append(T_sup_f)
            Tb_sup_back.append(T_sup_b)
        # --Inf surface:
        try:
            xmax=min(pos_max_inf_back[i][-1][0],pos_max_inf_front[i][-1][0]) # 
            xmin=max(pos_max_inf_back[i][0][0],pos_max_inf_front[i][0][0])
        except IndexError:
            r_inf.append(False)
            Tb_inf_front.append(False)
            Tb_inf_back.append(False)

        else:
            yb=np.array([])
            yf=np.array([])
            T_inf_f=np.array([])
            T_inf_b=np.array([])
            x=np.array([xmin+i for i in range(0,xmax-xmin)])
            for k in range(len(x)):
                yb=np.append(yb, pos_max_inf_back[i][k][1])
                yf=np.append(yf, pos_max_inf_front[i][k][1])
                T_inf_f=np.append(T_inf_f,flux_to_Tbrigh(CO[i][pos_max_sup_front[i][k][0],pos_max_sup_front[i][k][1]],wl,BMAJ,BMIN))
                T_inf_b=np.append(T_inf_b,flux_to_Tbrigh(CO[i][pos_max_sup_back[i][k][0],pos_max_sup_back[i][k][1]],wl,BMAJ,BMIN))
            r_inf.append(copy(r(x,yf,yb)))
            Tb_inf_front.append(T_inf_f)
            Tb_inf_back.append(T_inf_b)

    else:
        r_sup.append(False)        
        r_inf.append(False)
        Tb_sup_front.append(False)
        Tb_sup_back.append(False)
        Tb_inf_front.append(False)
        Tb_inf_back.append(False)



Tb_sup_back_flat=[]
Tb_sup_front_flat=[]
Tb_inf_back_flat=[]
Tb_inf_front_flat=[]

r_sup_flat=[]
r_inf_flat=[]


# flattening temp

for sublist in Tb_sup_back:
    try: 
        for item in sublist:
            Tb_sup_back_flat.append(item)
    except TypeError:
        pass

for sublist in Tb_sup_front:
    try: 
        for item in sublist:
            Tb_sup_front_flat.append(item)
    except TypeError:
        pass
    
for sublist in Tb_inf_back:
    try: 
        for item in sublist:
            Tb_inf_back_flat.append(item)
    except TypeError:
        pass

for sublist in Tb_inf_front:
    try: 
        for item in sublist:
            Tb_inf_front_flat.append(item)
    except TypeError:
        pass


# flattening radii
        
for sublist in r_inf:
    try: 
        for item in sublist:
            r_inf_flat.append(item)
    except TypeError:
        pass
    
for sublist in r_sup:
    try: 
        for item in sublist:
            r_sup_flat.append(item)
    except TypeError:
        pass

idx = np.argsort(r_sup_flat)
r_sup_flat=np.array(r_sup_flat)[idx]
Tb_sup_front_flat=np.array(Tb_sup_front_flat)[idx]
Tb_sup_back_flat=np.array(Tb_sup_back_flat)[idx]

idx = np.argsort(r_inf_flat)
r_inf_flat=np.array(r_inf_flat)[idx]
Tb_inf_front_flat=np.array(Tb_inf_front_flat)[idx]
Tb_inf_back_flat=np.array(Tb_inf_back_flat)[idx]

delta_sup=25 #                                               ######### /!\ TO BE MODIFIED FOR EACH OBJECT
delta_inf=10                                                 ######### /!\ TO BE MODIFIED FOR EACH OBJECT            
idx_sup_front_max=[]
idx_sup_back_max=[]
idx_inf_front_max=[]
idx_inf_back_max=[]

j=0
for i in range(int(len(Tb_sup_front_flat)/delta_sup)):
    try:
        idx_sup_front_max.append(i*delta_sup+posmax(Tb_sup_front_flat[i*delta_sup:(i+1)*delta_sup]))
        idx_sup_back_max.append(i*delta_sup+posmax(Tb_sup_back_flat[i*delta_sup:(i+1)*delta_sup]))
    except IndexError:
        break
for i in range(int(len(Tb_inf_front_flat)/delta_inf)):
    try:
        idx_inf_front_max.append(i*delta_inf+posmax(Tb_inf_front_flat[i*delta_inf:(i+1)*delta_inf]))
        idx_inf_back_max.append(i*delta_inf+posmax(Tb_inf_back_flat[i*delta_inf:(i+1)*delta_inf]))
    except IndexError:
        break

idx_sup_front_max=np.array(idx_sup_front_max)
idx_sup_back_max=np.array(idx_sup_back_max)
idx_inf_front_max=np.array(idx_inf_front_max)
idx_inf_back_max=np.array(idx_inf_back_max)

plt.plot(r_sup_flat[idx_sup_front_max], Tb_sup_front_flat[idx_sup_front_max],'+',color='blue')
plt.plot(r_sup_flat[idx_sup_back_max], Tb_sup_back_flat[idx_sup_back_max],'+',color='black')
plt.plot(r_inf_flat[idx_inf_front_max], Tb_inf_front_flat[idx_inf_front_max],'+',color='red')
plt.plot(r_inf_flat[idx_inf_back_max], Tb_inf_back_flat[idx_inf_back_max],'+',color='green')

#plt.ylim((0,50))
plt.xlabel('R [au]')
plt.ylabel('T_B [K]')
plt.show()


