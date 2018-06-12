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

def h(yc):
    """ Height of the orbit. yc = (y_front+y_back)/2
    Units : x, yf, yb : pixel
            r : [au]
    
    """
    return abs((yc-ys)/np.sin(inc))*px_size*D

def v(x,yf,yb,v0):
    """ Velocity of the gas around the star
    Units: x, yf, yb : [pixel]
           v0, v : [m/s]             
    """
    return abs(v0*r(x,yf,yb)/((x-xs)*np.sin(inc)*px_size*D))

def v_kep(R,M):
    """ Returns the keplerian velocity at radius R (au) around a star of mass M (unit of M_sun)"""
    return np.sqrt(G*M*M_sun/(R*au))

def confIntMean(a, conf=0.95):
    sem, m = st.sem(a), st.t.ppf((1+conf)/2., len(a)-1)
    return m*sem

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
# Radius, height and velocity of the orbits:
# =============================================================================

r_sup=[]
r_inf=[]
h_sup=[]
h_inf=[]
v_sup=[]
v_inf=[]

for i in range(n):
    if i <= 18 or i >= 36 : # removing the chans where the arms were too vertical
        v0=v_obs[i]
        # --Sup surface:
        try:
            xmax=min(pos_max_sup_back[i][-1][0],pos_max_sup_front[i][-1][0]) # 
            xmin=max(pos_max_sup_back[i][0][0],pos_max_sup_front[i][0][0])
        except IndexError:
            r_sup.append(False)
            h_sup.append(False)
            v_sup.append(False)
        else:
            yb=np.array([])
            yf=np.array([])
            yc=np.array([])
            x=np.array([xmin+i for i in range(0,xmax-xmin)])
            for k in range(len(x)):
                yb=np.append(yb, pos_max_sup_back[i][k][1])
                yf=np.append(yf, pos_max_sup_front[i][k][1])
                yc=(yb+yf)/2
                
            r_sup.append(copy(r(x,yf,yb)))
            h_sup.append(copy(h(yc)))
            v_sup.append(copy(v(x,yf,yb,v0))) 
        
        # --Inf surface:
        try:
            xmax=min(pos_max_inf_back[i][-1][0],pos_max_inf_front[i][-1][0]) # 
            xmin=max(pos_max_inf_back[i][0][0],pos_max_inf_front[i][0][0])
        except IndexError:
            r_inf.append(False)
            h_inf.append(False)    
            v_inf.append(False)
        else:
            yb=np.array([])
            yf=np.array([])
            yc=np.array([])
            x=np.array([xmin+i for i in range(0,xmax-xmin)])
            for k in range(len(x)):
                yb=np.append(yb, pos_max_inf_back[i][k][1])
                yf=np.append(yf, pos_max_inf_front[i][k][1])
                yc=(yb+yf)/2
            r_inf.append(copy(r(x,yf,yb)))
            h_inf.append(copy(h(yc)))
            v_inf.append(copy(v(x,yf,yb,v0)))
    else:
        r_sup.append(False)
        h_sup.append(False)
        v_sup.append(False)
        r_inf.append(False)
        h_inf.append(False)    
        v_inf.append(False)


# =============================================================================
# Test for a single chan   
# =============================================================================

n_test=17

# Visual checking of the pos of each maximum

img=CO[n_test] # image with the same rotation, window, etc ... as the get_CO_surf.py script
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

plt.plot(r_sup[n_test],h_sup[n_test],'x')
plt.xlabel("R [au]")
plt.ylabel("h CO [au]")
plt.title(" Radius vs heigh for channel " + str(n_test))
plt.show()

# Raduis vs velocity

plt.plot(r_sup[n_test],v_sup[n_test]/1000)

# Fit of the velocity with a keplerian model

M, pcov = curve_fit(v_kep, r_sup[n_test], v_sup[n_test])
R=np.linspace(50,500,200)
plt.plot(R, v_kep(R,M)/1000)
plt.xlabel("R [au]")
plt.ylabel("v [km/s]")
plt.title(" Radius vs speed for channel " + str(n_test))
plt.show()

print(M)

# =============================================================================
# All the chan
# =============================================================================

# Attempt 1 : group all the chan, fit

# Flatten the lists :
radius=[]
height=[]
speed=[]

for sublist in r_sup+r_inf:
    try: 
        for item in sublist:
            radius.append(item)
    except TypeError:
        pass

for sublist in h_sup+h_inf:
    try: 
        for item in sublist:
            height.append(item)
    except TypeError:
        pass
            
for sublist in v_sup+v_inf:
    try: 
        for item in sublist:
            speed.append(item)
    except TypeError:
        pass

radius=np.array(radius)
height=np.array(height)
speed=np.array(speed)

# Sort theese lists by increasing radii :

idx = np.argsort(radius)
radius=np.array(radius)[idx]
height=np.array(height)[idx]
speed=np.array(speed)[idx]

plt.plot(radius, height,"+")
plt.xlabel("R [au]")
plt.ylabel("h CO [au]")
plt.title(" Radius vs height on all the channels")
plt.show()

plt.plot(radius, speed/1000,"+")
M, pcov = curve_fit(v_kep, r_sup[n_test], v_sup[n_test])
R=np.linspace(50,500,200)
plt.plot(R,v_kep(R,M)/1000)
plt.xlabel("R [au]")
plt.ylabel("v [km/s]")
plt.ylim((1,8))
plt.title(" Radius vs speed all the channels + fit keplerian velocity")
plt.show()

print('mass of the star: ',float(M), ' Msun')


# Attempt 2 : fit each chan individualy, average on all the fits
mass=[]
for i in range(n):
    if type(r_sup[i])!= bool:
        M, pcov = curve_fit(v_kep, r_sup[i], v_sup[i])
        mass.append(M) 
    if type(r_inf[i])!=bool:
        M, pcov = curve_fit(v_kep, r_inf[i], v_inf[i])
        mass.append(M)
    

mass=np.mean(mass)

R=np.linspace(50,500,200)
plt.plot(radius, speed/1000,"+")
plt.plot(R, v_kep(R,mass)/1000)
plt.xlabel("R [au]")
plt.ylabel("v [km/s]")
plt.ylim((1,8))
plt.title(" Radius vs speed, fit chan 1 by 1 and averaged mass")
plt.show()
print('mass of the star: ',mass, ' Msun')



# =============================================================================
# Dispersion
# =============================================================================

deltaR=10 # (au)
Rmin=20
Rmax=550

radius_avg=[]
radius_err=[]
height_avg=[]
height_err=[]
speed_avg=[]
speed_err=[]

j=0
for i in range(int((Rmax-Rmin)/deltaR)):
    avg_r=[]
    avg_h=[]
    avg_s=[]
    while j<len(radius) and radius[j]>Rmin+i*deltaR and radius[j]<Rmin+(i+1)*deltaR :       
        avg_r.append(radius[j])
        avg_h.append(height[j])
        avg_s.append(speed[j]/1000)
        j+=1
    if len(avg_r)!=0:
        radius_avg.append(np.mean(avg_r))
        radius_err.append(confIntMean(avg_r)) # 95% confidence interval
        height_avg.append(np.mean(avg_h))
        height_err.append(confIntMean(avg_h)) # 95% confidence interval
        speed_avg.append(np.mean(avg_s))
        speed_err.append(confIntMean(avg_s)) # 95% confidence interval
        
        
plt.errorbar(radius_avg,height_avg,height_err,radius_err,fmt='+')
plt.xlabel("R [au]")
plt.ylabel("h CO [au]")
plt.show()
plt.errorbar(radius_avg,speed_avg,speed_err,radius_err,fmt='+')
plt.plot(R, v_kep(R,M)/1000)
plt.xlabel("R [au]")
plt.ylabel("v [km/s]")
plt.show()
        

# =============================================================================
# Temperature of brigtness
# =============================================================================

Tb_sup_front=[]
Tb_sup_back=[]
Tb_inf_front=[]
Tb_inf_back=[]
for i in range(n):
    if i <= 18 or i >= 36 : # removing the chans where the arms were too vertical
        v0=v_obs[i]
        # --Sup surface:
        try:
            xmax=min(pos_max_sup_back[i][-1][0],pos_max_sup_front[i][-1][0]) # 
            xmin=max(pos_max_sup_back[i][0][0],pos_max_sup_front[i][0][0])
        except IndexError:
            Tb_sup_front.append(False)
            Tb_sup_back.append(False)

        else:
            T_sup_f=np.array([])
            T_sup_b=np.array([])
            x=np.array([xmin+i for i in range(0,xmax-xmin)])
            for k in range(len(x)):
                T_sup_f=np.append(T_sup_f,flux_to_Tbrigh(CO[i][pos_max_sup_front[i][k][0],pos_max_sup_front[i][k][1]],wl,BMAJ,BMIN))
                T_sup_b=np.append(T_sup_b,flux_to_Tbrigh(CO[i][pos_max_sup_back[i][k][0],pos_max_sup_back[i][k][1]],wl,BMAJ,BMIN))

            Tb_sup_front.append(T_sup_f)
            Tb_sup_back.append(T_sup_b)
        # --Inf surface:
        try:
            xmax=min(pos_max_inf_back[i][-1][0],pos_max_inf_front[i][-1][0]) # 
            xmin=max(pos_max_inf_back[i][0][0],pos_max_inf_front[i][0][0])
        except IndexError:
            Tb_inf_front.append(False)
            Tb_inf_back.append(False)

        else:
            T_inf_f=np.array([])
            T_inf_b=np.array([])
            x=np.array([xmin+i for i in range(0,xmax-xmin)])
            for k in range(len(x)):
                T_inf_f=np.append(T_inf_f,flux_to_Tbrigh(CO[i][pos_max_sup_front[i][k][0],pos_max_sup_front[i][k][1]],wl,BMAJ,BMIN))
                T_inf_b=np.append(T_inf_b,flux_to_Tbrigh(CO[i][pos_max_sup_back[i][k][0],pos_max_sup_back[i][k][1]],wl,BMAJ,BMIN))

            Tb_inf_front.append(T_inf_f)
            Tb_inf_back.append(T_inf_b)

    else:
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

plt.plot(r_sup_flat, Tb_sup_front_flat,'+',color='blue')
plt.plot(r_inf_flat, Tb_inf_front_flat,'+',color='red')
plt.plot(r_sup_flat, Tb_sup_back_flat,'+',color='black')
plt.plot(r_inf_flat, Tb_inf_back_flat,'+',color='green')

#plt.ylim((0,50))
plt.xlabel('R [au]')
plt.ylabel('T_B [K]')
plt.show()





