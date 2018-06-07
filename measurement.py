#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 12:53:43 2018

@author: hugo
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

# =============================================================================
# Constants
# =============================================================================

G=const.G.value
M_sun=const.M_sun.value
c=const.c.value
au=const.au.value

# =============================================================================
# Functions
# =============================================================================

def r(x,yf,yb):
    """ Radius of the orbit. y can either be the y coordinate of a front or of a back side.
    Units : x, yf, yb : pixel
            r : au
    """
    yc=(yf+yb)/2
    return ((x-xs)**2+((yf-yc)/np.cos(inc))**2)**0.5*px_size*D

def h(yc):
    """ Height of the orbit. yc = (y_front+y_back)/2
    Units : x, yf, yb : pixel
            r : au
    
    """
    return abs((yc-ys)/np.sin(inc))*px_size*D

def v(x,yf,yb,v0):
    """ Velocity of the gas around the star
    Units: x, yf, yb : pixel
           v0, v : m/s             
    """
    return abs(v0*r(x,yf,yb)/((x-xs)*np.sin(inc)*px_size*D))

def v_kep(R,M):
    """ Returns the keplerian velocity at radius R (au) around a star of mass M (unit of M_sun)"""
    return np.sqrt(G*M*M_sun/(R*au))


# =============================================================================
# Data
# =============================================================================


path="/home/hugo/Documents/Stage/selection_objets/"
fits_name = path+"HD163296/Itziar/HD163296_CO3-2.fits.gz" 
pc2au=648000/np.pi #parsec to au
deg2rad=np.pi/180 

D= 122 # (pc)
inc=38 *deg2rad # (rad)


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


# =============================================================================
# Radius, height and velocity of the orbits:
# =============================================================================

r_front=[]
r_back=[]
h_front=[]
h_back=[]
v_front=[]
v_back=[]

for i in range(n):
    v0=v_obs[i]
    # --Sup surface:
    try:
        xmax=min(pos_max_sup_back[i][-1][0],pos_max_sup_front[i][-1][0]) # 
        xmin=max(pos_max_sup_back[i][0][0],pos_max_sup_front[i][0][0])
    except IndexError:
        r_front.append(False)
        h_front.append(False)
        v_front.append(False)
    else:
        yb=np.array([])
        yf=np.array([])
        yc=np.array([])
        x=np.array([xmin+i for i in range(0,xmax-xmin)])
        for k in range(len(x)):
            yb=np.append(yb, pos_max_sup_back[i][k][1])
            yf=np.append(yf, pos_max_sup_front[i][k][1])
            yc=(yb+yf)/2
        r_front.append(copy(r(x,yf,yb)))
        h_front.append(copy(h(yc)))
        v_front.append(copy(v(x,yf,yb,v0))) 
    
    # --Inf surface:
    try:
        xmax=min(pos_max_inf_back[i][-1][0],pos_max_inf_front[i][-1][0]) # 
        xmin=max(pos_max_inf_back[i][0][0],pos_max_inf_front[i][0][0])
    except IndexError:
        r_back.append(False)
        h_back.append(False)    
        v_back.append(False)
    else:
        yb=np.array([])
        yf=np.array([])
        yc=np.array([])
        x=np.array([xmin+i for i in range(0,xmax-xmin)])
        for k in range(len(x)):
            yb=np.append(yb, pos_max_inf_back[i][k][1])
            yf=np.append(yf, pos_max_inf_front[i][k][1])
            yc=(yb+yf)/2
        r_back.append(copy(r(x,yf,yb)))
        h_back.append(copy(h(yc)))
        v_back.append(copy(v(x,yf,yb,v0)))


# =============================================================================
# Test for a single chan   
# =============================================================================

n_test=17

# Visual checking of the pos of each maximum

img=rotate(CO[ni+n_test][xw0:xw1,yw0:yw1], PA, reshape=False) # image with the same rotation, window, etc ... as the get_CO_surf.py script
plt.imshow(img, cmap='afmhot')
plt.plot(xs, ys, "*", color="yellow")
x_max_inf_back=[pos_max_inf_back[n_test][k][0] for k in range(len(pos_max_inf_back[n_test]))]
y_max_inf_back=[pos_max_inf_back[n_test][k][1] for k in range(len(pos_max_inf_back[n_test]))]
x_max_inf_front=[pos_max_inf_front[n_test][k][0] for k in range(len(pos_max_inf_front[n_test]))]
y_max_inf_front=[pos_max_inf_front[n_test][k][1] for k in range(len(pos_max_inf_front[n_test]))]
x_max_sup_back=[pos_max_sup_back[n_test][k][0] for k in range(len(pos_max_sup_back[n_test]))]
y_max_sup_back=[pos_max_sup_back[n_test][k][1] for k in range(len(pos_max_sup_back[n_test]))]
x_max_sup_front=[pos_max_sup_front[n_test][k][0] for k in range(len(pos_max_sup_front[n_test]))]
y_max_sup_front=[pos_max_sup_front[n_test][k][1] for k in range(len(pos_max_sup_front[n_test]))]

plt.plot(x_max_inf_back, y_max_inf_back, "x", color="blue")
plt.plot(x_max_inf_front, y_max_inf_front, "x", color="green")
plt.plot(x_max_sup_back, y_max_sup_back,"x", color="black")
plt.plot(x_max_sup_front, y_max_sup_front, "x", color="red")
plt.show()

# Radius vs Height

plt.plot(r_front[n_test],h_front[n_test],'x')
plt.xlabel("R [au]")
plt.ylabel("h CO [au]")
plt.show()

# Raduis vs velocity

plt.plot(r_front[n_test],v_front[n_test]/1000)

# Fit of the velocity with a keplerian model

M, pcov = curve_fit(v_kep, r_front[n_test], v_front[n_test])
R=np.linspace(50,500,200)
plt.plot(R, v_kep(R,M)/1000)
plt.xlabel("R [au]")
plt.ylabel("v [km/s]")
plt.show()

print(M)

# =============================================================================
# Averaged on all the chan
# =============================================================================





























