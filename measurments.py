#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 09:30:47 2018

@author: hugo
"""

# =============================================================================
# Packages
# =============================================================================

from astropy.io import fits # Reading fits files
import pickle
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
from get_CO_surf_gauss_fit import storage
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
import database
import scipy.optimize as scpo
from scipy.ndimage.interpolation import rotate
from scipy.optimize import curve_fit
import pandas as pd
from scipy.misc import derivative
from skimage.transform import resize as imresize

# latex figure
#from matplotlib import rc
from matplotlib2tikz import save as tikz_save
# =============================================================================
# Classes
# =============================================================================

class FindPA():
    
    def __init__(self, obj):
        data=database.DATA(obj)
        self.const=database.CONSTANTS()
        fits_name = data.FITS  
        self.D=data.DIST
        self.inc=data.INC*self.const.deg2rad
        
        fh=fits.open(fits_name)
        CDELT1=fh[0].header['CDELT1'] 
        CO=fh[0].data
        CO=CO[0]
        self.nx=len(CO[0][0]) # number of columns
        self.ny=len(CO[0]) # number of lines
        self.nv=len(CO) #number of chanels
        self.px_size=abs(CDELT1)*3600 #arcsec
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
        self.xs,self.ys=data_sup_back.star_center # coordinates of the star's center /!\ IN THE CROPPED IMAGE /!\
        
        self.xw0, self.yw0 = data_sup_back.window[0] # coordinates of the selected window 
        self.xw1, self.yw1 = data_sup_back.window[1]
        self.xs,self.ys=(self.xs+self.xw0,self.ys+self.yw0) # coordinates of the star's center /!\ IN THE UNCROPPED IMAGE /!\
        self.X=(self.nx/2,self.ny/2) # coordinates of the center of rotation
        self.pos_max_sup_back=np.array(data_sup_back.pos_maxima)
        self.pos_max_sup_front=np.array(data_sup_front.pos_maxima)
        self.pos_max_inf_back=np.array(data_inf_back.pos_maxima)
        self.pos_max_inf_front=np.array(data_inf_front.pos_maxima)
        
        self.n=len(self.pos_max_sup_back)
        
        CO=data_sup_front.CO # Same image than in the previous script 
        
        self.opt_pa = self.PA_opt()
    
    def PA_opt(self):
        """ Finds the optimal value of PA by minimizing the disperion of the height of the gas disk. 
        Tested values are included in [PA-10,PA+10]
        """
        # Rotation of the image
        lowpa,highpa=-10,11
        PA_range=[a for a in range(lowpa,highpa)]
        dispersion=[]
        for pa in PA_range:
            r_sup=np.array([])
            h_sup=np.array([])
            (xs,ys)=self.rotate_vect((self.xs,self.ys),pa,self.X)
            for i in range(self.n):
                if len(self.pos_max_sup_back[i])!=0 and len(self.pos_max_sup_front[i])!=0:
                    temp_sb=[]
                    temp_sf=[]
                    for item in self.pos_max_sup_back[i]:
                        x,y=item
                        x,y=(x+self.xw0,y+self.yw0)
                        item=(x,y)
                        item_rot=self.rotate_vect(item, pa,self.X)
                        temp_sb.append(item_rot)
                    temp_sb=np.array(temp_sb)
                    interp_sup_back=interp1d(temp_sb[:,0],temp_sb[:,1], kind='cubic')
                    
                    for item in self.pos_max_sup_front[i]:
                        x,y=item
                        x,y=(x+self.xw0,y+self.yw0)
                        item=(x,y)
                        item_rot=self.rotate_vect(item, pa,self.X)
                        temp_sf.append(item_rot)
                    temp_sf=np.array(temp_sf)
                    interp_sup_front=interp1d(temp_sf[:,0],temp_sf[:,1], kind='cubic')
                    xmin=max(np.min(temp_sf[:,0]),np.min(temp_sb[:,0]))
                    xmax=min(np.max(temp_sf[:,0]),np.max(temp_sb[:,0]))
                    
                    if xmin>xmax: # if the two surface are misaligned : continue to the next iteration of the loop without doing anything
                        continue
                    x=np.linspace(xmin, xmax, 50)
                    yb=interp_sup_back(x)
                    yf=interp_sup_front(x)
                    yc=(yb+yf)/2
                    
                    r_sup=np.concatenate((r_sup,copy(self.r(x,yf,yb))))
                    h_sup=np.concatenate((h_sup,copy(self.h(yc))))
                    
            
            idx = np.argsort(r_sup)
            r_sup=np.array(r_sup)[idx]
            h_sup=np.array(h_sup)[idx]
            # Dispersion
            
            delta=100                                       
                            
            r_sup_avg=[]
            r_sup_std=[]
            h_sup_avg=[]
            h_sup_std=[]
    
            for i in range(int(len(r_sup)/delta)):
                # Sup surface
                r_sup_avg.append(np.mean(r_sup[i*delta:(i+1)*delta]))
                r_sup_std.append(np.std(r_sup[i*delta:(i+1)*delta]))
                h_sup_avg.append(np.mean(h_sup[i*delta:(i+1)*delta]))
                h_sup_std.append(np.std(h_sup[i*delta:(i+1)*delta]))
                
            r_sup_avg=np.array(r_sup_avg)
            r_sup_std=np.array(r_sup_std)
            h_sup_avg=np.array(h_sup_avg)
            h_sup_std=np.array(h_sup_std)

            
            dispersion_h=np.linalg.norm(h_sup_std)
            dispersion.append(dispersion_h)

        # Finding the value of PA that minimize the dispersion of height   
        
        dispersion_interp=interp1d(PA_range, dispersion,kind='cubic') 
        pa=np.linspace(-10,+10,50)
        plt.plot(pa,dispersion_interp(pa))
        plt.xlabel('Correction [°]')
        plt.ylabel('Dispersion [au]')
        tikz_save('dispersion.tex',figureheight='\\figureheight',figurewidth='\\figurewidth')
        plt.show()
        opt=scpo.minimize(dispersion_interp,0, bounds=[(lowpa,highpa-2)])
        
        return opt
    
    def r(self,x,yf,yb):
        """ Radius of the orbit. y can either be the y coordinate of a front or of a back side.
        Units : x, yf, yb : pixel
                r : [au]
        """
        yc=(yf+yb)/2
        return ((x-self.xs)**2+((yf-yc)/np.cos(self.inc))**2)**0.5*self.px_size*self.D

    def h(self,yc):
        """ Height of the orbit. yc = (y_front+y_back)/2
        Units : 
            x, yf, yb [pixel]
            r [au]
        
        """
        return abs((yc-self.ys)/np.sin(self.inc))*self.px_size*self.D
    
    def rotate_vect(self,vect,angle,X):
        """ Rotates the vector vect by the given angle around the point X
        Units: 
            angle [deg]"""
        vect=np.array(vect)
        X=np.array(X)
        vect=vect-X
        angle=angle*self.const.deg2rad
        RotMatrix=np.array([[np.cos(angle) , -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        return np.dot(RotMatrix,vect)+X
    

class MakeMeasurements():
    def __init__(self, obj, PA_corr=None):
        if PA_corr==None:
            param=FindPA(obj)
            self.PA_corr=float(param.opt_pa.x)
        else:
            self.PA_corr=float(PA_corr)
            
        data=database.DATA(obj)
        self.obj=obj
        self.const=database.CONSTANTS()
        fits_name = data.FITS  
        self.D=data.DIST
        self.inc=data.INC*self.const.deg2rad
        self.path=data.PATH
        ct=fits.open(data.CONT)
        self.cont_img=ct[0].data
        self.cont_img=self.cont_img[0,0,:,:]
        test=ct[0].header['CDELT1'] 

        ct.close()
        
        fh=fits.open(fits_name)
        CDELT1=fh[0].header['CDELT1'] 
        CO=fh[0].data
        self.CO=CO[0]
        self.nx=len(self.CO[0][0]) # number of columns
        self.ny=len(self.CO[0]) # number of lines
        self.nv=len(self.CO) #number of chanels
        self.px_size=abs(CDELT1)*3600 #arcsec
        restfreq=fh[0].header['RESTFRQ'] #freq of the transition studied
        CRVAL3=fh[0].header['CRVAL3'] #frequency of the 1st channel
        CDELT3=fh[0].header['CDELT3'] #freq step
        self.casa_version=4
        try: # depending on the version of casa used 
            self.BMIN=fh[0].header['BMIN'] # [deg] Beam major axis length
            self.BMAJ=fh[0].header['BMAJ'] # [deg] Beam minor axis length
        except KeyError: 
            self.BMAJ=0
            self.BMIN=0
            for item in fh[1].data:
                self.BMAJ+=item[0]
                self.BMIN+=item[1]
            self.BMAJ=self.BMAJ/len(fh[1].data) #arcsec
            self.BMIN=self.BMIN/len(fh[1].data) #arcsec
            self.casa_version=5
            
        self.wl=self.const.c/CRVAL3
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
        
        self.xs,self.ys=data_sup_back.star_center # coordinates of the star's center /!\ IN THE CROPPED IMAGE /!\
        self.PA=data_sup_back.PA
        self.ni=data_sup_back.ni # 1st interesting chan
        self.nm=data_sup_back.nm # 0 velocity chan
        self.nf=data_sup_back.nf # last interesting chan
        self.CO=self.CO[self.ni:self.nf]
        self.xw0, self.yw0 = data_sup_back.window[0] # coordinates of the selected window 
        self.xw1, self.yw1 = data_sup_back.window[1]
        freq= CRVAL3 + CDELT3*np.arange(self.nv) # freq of the channels
        freq=freq[self.ni:self.nf] # only the selected chan
        self.v_syst= -(freq[self.nm-self.ni]-restfreq)*self.const.c/restfreq # global speed of the system
        self.v_obs = -((freq-restfreq)/restfreq)*self.const.c-self.v_syst # radial velocity of the channels
        
        self.xs,self.ys=(self.xs+self.xw0,self.ys+self.yw0)
        self.X=(self.nx/2,self.ny/2)
        self.xs,self.ys=self.rotate_vect((self.xs,self.ys),self.PA_corr,self.X)
        self.pos_max_sup_back=np.array(data_sup_back.pos_maxima)
        self.pos_max_sup_front=np.array(data_sup_front.pos_maxima)
        self.pos_max_inf_back=np.array(data_inf_back.pos_maxima)
        self.pos_max_inf_front=np.array(data_inf_front.pos_maxima)
        self.n=len(self.pos_max_sup_back)  
        
        self.max_sup_back=np.array(data_sup_back.carac_gauss)
        self.max_sup_front=np.array(data_sup_front.carac_gauss)
        self.max_inf_back=np.array(data_inf_back.carac_gauss)
        self.max_inf_front=np.array(data_inf_front.carac_gauss)
        
        np.nan_to_num(self.CO,copy=False) # replaces the nan values by 0
        np.nan_to_num(self.cont_img,copy=False) # replaces the nan values by 0
        self.cont_img=imresize(self.cont_img,(self.nx,self.ny),mode='constant') #resize the cont image so match the shape of the CO images
        self.cont_img=rotate(self.cont_img,180-(self.PA+self.PA_corr), reshape=False)
            
        
        self.RotateData(self.PA_corr) # to use the value of PA we previously found
        idx_rings=self.Rings()
        self.pos_rings=idx_rings[0]*self.px_size*self.D
        self.pos_gaps=idx_rings[1]*self.px_size*self.D


    def RotateData(self,angle):
        """ Rotates the data by the angle PA_corr [deg]. Automatically called in __init__."""
        angle=self.PA_corr
        r_sup_m=np.array([])
        h_sup_m=np.array([])
        v_sup_m=np.array([])
        r_inf_m=np.array([])
        h_inf_m=np.array([])
        v_inf_m=np.array([])
        r_sup_g=np.array([])
        h_sup_g=np.array([])
        v_sup_g=np.array([])
        r_inf_g=np.array([])
        h_inf_g=np.array([])
        v_inf_g=np.array([])
        Tb_sup_front=np.array([])
        Tb_sup_back=np.array([])
        Tb_inf_front=np.array([])
        Tb_inf_back=np.array([])
        sigma_sb=np.array([])
        sigma_sf=np.array([])
        sigma_if=np.array([])
        sigma_ib=np.array([])
        
        for i in range(self.n): #[k for k in range(0,18)]+[k for k in range(33,self.n)]:   ##### TO BE MODIFIED to remove some chan
            v0=self.v_obs[i]
            # Sup surface:
            if len(self.pos_max_sup_back[i])!=0 and len(self.pos_max_sup_front[i])!=0:
                temp_sb_gauss=[]
                temp_sf_gauss=[]
                temp_sb_max=[]
                temp_sf_max=[]
                temp_list=np.array(self.max_sup_back[i])
                for item in temp_list:
                    x,a,ygauss,sigma,ymax=item
                    x,ygauss=(x+self.xw0,ygauss+self.yw0)
                    ymax=ymax+self.yw0
                    item_gauss=self.rotate_vect((x,ygauss), angle,self.X)
                    item_max=self.rotate_vect((x,ymax), angle,self.X)
                    temp_sb_gauss.append(item_gauss)
                    temp_sb_max.append(item_max)
                temp_sb_gauss=np.array(temp_sb_gauss)
                temp_sb_max=np.array(temp_sb_max)
                interp_sb_gauss=interp1d(temp_sb_gauss[:,0],temp_sb_gauss[:,1], kind='cubic')
                interp_sb_max=interp1d(temp_sb_max[:,0],temp_sb_max[:,1], kind='cubic')
                interp_T_sb=interp1d(temp_sb_gauss[:,0],temp_list[:,1],kind='cubic')
                interp_sigma_sb=interp1d(temp_sb_gauss[:,0],temp_list[:,3],kind='cubic')

                temp_list=np.array(self.max_sup_front[i])
                for item in temp_list:
                    x,a,ygauss,sigma,ymax=item
                    x,ygauss=(x+self.xw0,ygauss+self.yw0)
                    ymax=ymax+self.yw0
                    item_gauss=self.rotate_vect((x,ygauss), angle,self.X)
                    item_max=self.rotate_vect((x,ymax), angle,self.X)
                    temp_sf_gauss.append(item_gauss)
                    temp_sf_max.append(item_max)
                temp_sf_gauss=np.array(temp_sf_gauss)
                temp_sf_max=np.array(temp_sf_max)
                interp_sf_gauss=interp1d(temp_sf_gauss[:,0],temp_sf_gauss[:,1], kind='cubic')
                interp_sf_max=interp1d(temp_sf_max[:,0],temp_sf_max[:,1], kind='cubic')
                interp_T_sf=interp1d(temp_sf_gauss[:,0],temp_list[:,1],kind='cubic')
                interp_sigma_sf=interp1d(temp_sf_gauss[:,0],temp_list[:,3],kind='cubic')
                
                
                xmin=max(np.min(temp_sb_max[:,0]),np.min(temp_sf_max[:,0]))
                xmax=min(np.max(temp_sb_max[:,0]),np.max(temp_sf_max[:,0]))
                x=np.linspace(xmin, xmax, 100)
                yb_max=interp_sb_max(x)
                yf_max=interp_sf_max(x)
                yc_max=(yb_max+yf_max)/2
                
                r_sup_m=np.concatenate((r_sup_m,copy(self.r(x,yf_max,yb_max))))
                h_sup_m=np.concatenate((h_sup_m,copy(self.h(yc_max))))
                v_sup_m=np.concatenate((v_sup_m,copy(self.v(x,yf_max,yb_max,v0))))
                
                xmin=max(np.min(temp_sb_gauss[:,0]),np.min(temp_sf_gauss[:,0]))
                xmax=min(np.max(temp_sb_gauss[:,0]),np.max(temp_sf_gauss[:,0]))
                x=np.linspace(xmin, xmax, 100)
                yb_gauss=interp_sb_gauss(x)
                yf_gauss=interp_sf_gauss(x)
                yc_gauss=(yb_gauss+yf_gauss)/2
                
                r_sup_g=np.concatenate((r_sup_g,copy(self.r(x,yf_gauss,yb_gauss))))
                h_sup_g=np.concatenate((h_sup_g,copy(self.h(yc_gauss))))
                v_sup_g=np.concatenate((v_sup_g,copy(self.v(x,yf_gauss,yb_gauss,v0))))
                flux=interp_T_sb(x)
                Tb_sup_back=np.concatenate((Tb_sup_back,copy(self.flux_to_Tbrigh(flux,self.wl,self.BMAJ,self.BMIN))))
                flux=interp_T_sf(x)
                Tb_sup_front=np.concatenate((Tb_sup_front,copy(self.flux_to_Tbrigh(flux,self.wl,self.BMAJ,self.BMIN))))
                sigma_sb=np.concatenate((sigma_sb,copy(interp_sigma_sb(x))))
                sigma_sf=np.concatenate((sigma_sf,copy(interp_sigma_sf(x))))

            # Inf surface:
            if len(self.pos_max_inf_back[i])!=0 and len(self.pos_max_inf_front[i])!=0:
                temp_ib_gauss=[]
                temp_if_gauss=[]
                temp_ib_max=[]
                temp_if_max=[]
                temp_list=np.array(self.max_inf_back[i])
                for item in temp_list:
                    x,a,ygauss,sigma,ymax=item
                    x,ygauss=(x+self.xw0,ygauss+self.yw0)
                    ymax=ymax+self.yw0
                    item_gauss=self.rotate_vect((x,ygauss), angle,self.X)
                    item_max=self.rotate_vect((x,ymax), angle,self.X)
                    temp_ib_gauss.append(item_gauss)
                    temp_ib_max.append(item_max)
                temp_ib_gauss=np.array(temp_ib_gauss)
                temp_ib_max=np.array(temp_ib_max)
                interp_ib_gauss=interp1d(temp_ib_gauss[:,0],temp_ib_gauss[:,1], kind='cubic')
                interp_ib_max=interp1d(temp_ib_max[:,0],temp_ib_max[:,1], kind='cubic')
                interp_T_ib=interp1d(temp_ib_gauss[:,0],temp_list[:,1],kind='cubic')
                interp_sigma_ib=interp1d(temp_ib_gauss[:,0],temp_list[:,3],kind='cubic')

                temp_list=np.array(self.max_inf_front[i])
                for item in temp_list:
                    x,a,ygauss,sigma,ymax=item
                    x,ygauss=(x+self.xw0,ygauss+self.yw0)
                    ymax=ymax+self.yw0
                    item_gauss=self.rotate_vect((x,ygauss), angle,self.X)
                    item_max=self.rotate_vect((x,ymax), angle,self.X)
                    temp_if_gauss.append(item_gauss)
                    temp_if_max.append(item_max)
                temp_if_gauss=np.array(temp_if_gauss)
                temp_if_max=np.array(temp_if_max)
                interp_if_gauss=interp1d(temp_if_gauss[:,0],temp_if_gauss[:,1], kind='cubic')
                interp_if_max=interp1d(temp_if_max[:,0],temp_if_max[:,1], kind='cubic')
                interp_T_if=interp1d(temp_if_gauss[:,0],temp_list[:,1],kind='cubic')
                interp_sigma_if=interp1d(temp_if_gauss[:,0],temp_list[:,3],kind='cubic')
                
                xmin=max(np.min(temp_ib_max[:,0]),np.min(temp_if_max[:,0]))
                xmax=min(np.max(temp_ib_max[:,0]),np.max(temp_if_max[:,0]))
                x=np.linspace(xmin, xmax, 100)
                yb_max=interp_ib_max(x)
                yf_max=interp_if_max(x)
                yc_max=(yb_max+yf_max)/2
                
                r_inf_m=np.concatenate((r_inf_m,copy(self.r(x,yf_max,yb_max))))
                h_inf_m=np.concatenate((h_inf_m,copy(self.h(yc_max))))
                v_inf_m=np.concatenate((v_inf_m,copy(self.v(x,yf_max,yb_max,v0))))
                
                xmin=max(np.min(temp_ib_gauss[:,0]),np.min(temp_if_gauss[:,0]))
                xmax=min(np.max(temp_ib_gauss[:,0]),np.max(temp_if_gauss[:,0]))
                x=np.linspace(xmin, xmax, 100)
                yb_gauss=interp_ib_gauss(x)
                yf_gauss=interp_if_gauss(x)
                yc_gauss=(yb_gauss+yf_gauss)/2
                
                r_inf_g=np.concatenate((r_inf_g,copy(self.r(x,yf_gauss,yb_gauss))))
                h_inf_g=np.concatenate((h_inf_g,copy(self.h(yc_gauss))))
                v_inf_g=np.concatenate((v_inf_g,copy(self.v(x,yf_gauss,yb_gauss,v0))))
                flux=interp_T_ib(x)
                Tb_inf_back=np.concatenate((Tb_inf_back,copy(self.flux_to_Tbrigh(flux,self.wl,self.BMAJ,self.BMIN))))
                flux=interp_T_if(x)
                Tb_inf_front=np.concatenate((Tb_inf_front,copy(self.flux_to_Tbrigh(flux,self.wl,self.BMAJ,self.BMIN))))
                sigma_ib=np.concatenate((sigma_ib,copy(interp_sigma_ib(x))))
                sigma_if=np.concatenate((sigma_if,copy(interp_sigma_if(x))))

    
        idx = np.argsort(r_sup_g)
        self.r_sup_g=np.array(r_sup_g)[idx]
        self.h_sup_g=np.array(h_sup_g)[idx]
        self.v_sup_g=np.array(v_sup_g)[idx] 
        self.Tb_sup_back=np.array(Tb_sup_back)[idx]
        self.Tb_sup_front=np.array(Tb_sup_front)[idx]
        self.sigma_sb=np.array(sigma_sb)[idx]
        self.sigma_sf=np.array(sigma_sf)[idx]
        idx = np.argsort(r_sup_m)
        self.r_sup_m=np.array(r_sup_m)[idx]
        self.h_sup_m=np.array(h_sup_m)[idx]
        self.v_sup_m=np.array(v_sup_m)[idx] 
    
        idx = np.argsort(r_inf_g)
        self.r_inf_g=np.array(r_inf_g)[idx]
        self.h_inf_g=np.array(h_inf_g)[idx]
        self.v_inf_g=np.array(v_inf_g)[idx] 
        self.Tb_inf_back=np.array(Tb_inf_back)[idx]
        self.Tb_inf_front=np.array(Tb_inf_front)[idx]
        self.sigma_ib=np.array(sigma_ib)[idx]
        self.sigma_if=np.array(sigma_if)[idx]
        idx = np.argsort(r_inf_m)
        self.r_inf_m=np.array(r_inf_m)[idx]
        self.h_inf_m=np.array(h_inf_m)[idx]
        self.v_inf_m=np.array(v_inf_m)[idx] 
        
        if len(self.r_inf_m)==0 or len(self.r_inf_g)==0: 
            self.isinf=False #if no data for inf surface
        else: 
            self.isinf=True


    def SingleChan(self,n_test):
        """ Plots the height, the speed and the position of the maxima for a single chanel"""
        angle=self.PA_corr
        img=self.CO[n_test] # image with the same rotation, window, etc ... as the get_CO_surf.py script
        img=rotate(img,180-(self.PA+self.PA_corr),reshape=False)
        img=img[self.yw0:self.yw1,self.xw0:self.xw1]
        v0=self.v_obs[n_test]
        fig=plt.figure()
        plt.imshow(img, cmap='afmhot', origin='lower')
        plt.plot(self.xs-self.xw0, self.ys-self.yw0, "*", color="black")

        if len(self.max_sup_back[n_test])!=0 and len(self.max_sup_front[n_test])!=0:
            temp_sb_gauss=[]
            temp_sb_max=[]
            temp_sf_gauss=[]
            temp_sf_max=[]

            for item in self.max_sup_back[n_test]:
                x,a,ygauss,sigma,ymax=item
                x,ygauss=(x+self.xw0,ygauss+self.yw0)
                ymax=ymax+self.yw0
                item_gauss=(x,ygauss)
                item_max=(x,ymax)
                item_rot_gauss=self.rotate_vect(item_gauss, angle,self.X)
                item_rot_max=self.rotate_vect(item_max, angle,self.X)
                temp_sb_gauss.append(item_rot_gauss)
                temp_sb_max.append(item_rot_max)
            temp_sb_gauss=np.array(temp_sb_gauss)
            temp_sb_max=np.array(temp_sb_max)
            interp_sup_back_gauss=interp1d(temp_sb_gauss[:,0],temp_sb_gauss[:,1], kind='cubic')
            interp_sup_back_max=interp1d(temp_sb_max[:,0],temp_sb_max[:,1], kind='cubic')

            
            for item in self.max_sup_front[n_test]:
                x,a,ygauss,sigma,ymax=item
                x,ygauss=(x+self.xw0,ygauss+self.yw0)
                ymax=ymax+self.yw0
                item_gauss=(x,ygauss)
                item_max=(x,ymax)
                item_rot_gauss=self.rotate_vect(item_gauss, angle,self.X)
                item_rot_max=self.rotate_vect(item_max, angle,self.X)
                temp_sf_gauss.append(item_rot_gauss)
                temp_sf_max.append(item_rot_max)
            temp_sf_gauss=np.array(temp_sf_gauss)
            temp_sf_max=np.array(temp_sf_max)
            interp_sup_front_gauss=interp1d(temp_sf_gauss[:,0],temp_sf_gauss[:,1], kind='cubic')
            interp_sup_front_max=interp1d(temp_sf_max[:,0],temp_sf_max[:,1], kind='cubic')

            xmin=max(np.min(temp_sf_max[:,0]),np.min(temp_sb_max[:,0]))
            xmax=min(np.max(temp_sf_max[:,0]),np.max(temp_sb_max[:,0]))
            x=np.linspace(xmin, xmax, 20)
            yb=interp_sup_back_max(x)
            yf=interp_sup_front_max(x)
            plt.plot(x-self.xw0, yb-self.yw0, "+", color="blue",markersize=4,alpha=0.5)
            plt.plot(x-self.xw0, yf-self.yw0, "+", color="blue",markersize=4,alpha=0.5)
            yc=(yb+yf)/2
            r_sup_m=self.r(x,yf,yb)
            h_sup_m=self.h(yc)
            v_sup_m=self.v(x,yf,yb,v0)
            
            xmin=max(np.min(temp_sf_gauss[:,0]),np.min(temp_sb_gauss[:,0]))
            xmax=min(np.max(temp_sf_gauss[:,0]),np.max(temp_sb_gauss[:,0]))
            x=np.linspace(xmin, xmax, 20)
            yb=interp_sup_back_gauss(x)
            yf=interp_sup_front_gauss(x)

            temp_list=np.array(self.max_sup_back[n_test])
            interp_sigma_sb=interp1d(temp_sb_gauss[:,0],temp_list[:,3],kind='cubic')
            sigma_sb=interp_sigma_sb(x)
            temp_list=np.array(self.max_sup_front[n_test])
            interp_sigma_sf=interp1d(temp_sf_gauss[:,0],temp_list[:,3],kind='cubic')
            sigma_sf=interp_sigma_sf(x)
            plt.plot(x-self.xw0, yb-self.yw0, "x", color="red",markersize=5,alpha=0.5)
            plt.plot(x-self.xw0, yf-self.yw0, "x", color="red",markersize=5,alpha=0.5)
            yc=(yb+yf)/2
            r_sup_g=self.r(x,yf,yb)
            h_sup_g=self.h(yc)
            v_sup_g=self.v(x,yf,yb,v0)

            # Inf surface:
        if len(self.max_inf_back[n_test])!=0 and len(self.max_inf_front[n_test])!=0:
            temp_ib_gauss=[]
            temp_ib_max=[]
            temp_if_gauss=[]
            temp_if_max=[]

            for item in self.max_inf_back[n_test]:
                x,a,ygauss,sigma,ymax=item
                x,ygauss=(x+self.xw0,ygauss+self.yw0)
                ymax=ymax+self.yw0
                item_gauss=(x,ygauss)
                item_max=(x,ymax)
                item_rot_gauss=self.rotate_vect(item_gauss, angle,self.X)
                item_rot_max=self.rotate_vect(item_max, angle,self.X)
                temp_ib_gauss.append(item_rot_gauss)
                temp_ib_max.append(item_rot_max)
            temp_ib_gauss=np.array(temp_ib_gauss)
            temp_ib_max=np.array(temp_ib_max)
            interp_inf_back_gauss=interp1d(temp_ib_gauss[:,0],temp_ib_gauss[:,1], kind='cubic')
            interp_inf_back_max=interp1d(temp_ib_max[:,0],temp_ib_max[:,1], kind='cubic')

            
            for item in self.max_inf_front[n_test]:
                x,a,ygauss,sigma,ymax=item
                x,ygauss=(x+self.xw0,ygauss+self.yw0)
                ymax=ymax+self.yw0
                item_gauss=(x,ygauss)
                item_max=(x,ymax)
                item_rot_gauss=self.rotate_vect(item_gauss, angle,self.X)
                item_rot_max=self.rotate_vect(item_max, angle,self.X)
                temp_if_gauss.append(item_rot_gauss)
                temp_if_max.append(item_rot_max)
            temp_if_gauss=np.array(temp_if_gauss)
            temp_if_max=np.array(temp_if_max)
            interp_inf_front_gauss=interp1d(temp_if_gauss[:,0],temp_if_gauss[:,1], kind='cubic')
            interp_inf_front_max=interp1d(temp_if_max[:,0],temp_if_max[:,1], kind='cubic')



            xmin=max(np.min(temp_if_max[:,0]),np.min(temp_ib_max[:,0]))
            xmax=min(np.max(temp_if_max[:,0]),np.max(temp_ib_max[:,0]))
            x=np.linspace(xmin, xmax, 20)
            yb=interp_inf_back_max(x)
            yf=interp_inf_front_max(x)
#            plt.plot(x-self.xw0, yb-self.yw0, "+", color="blue",markersize=4,alpha=0.5)
#            plt.plot(x-self.xw0, yf-self.yw0, "+", color="blue",markersize=4,alpha=0.5)
            
            xmin=max(np.min(temp_if_gauss[:,0]),np.min(temp_ib_gauss[:,0]))
            xmax=min(np.max(temp_if_gauss[:,0]),np.max(temp_ib_gauss[:,0]))
            x=np.linspace(xmin, xmax, 20)
            yb=interp_inf_back_gauss(x)
            yf=interp_inf_front_gauss(x)
            
#            plt.plot(x-self.xw0, yb-self.yw0, "x", color="green",markersize=5,alpha=1)
#            plt.plot(x-self.xw0, yf-self.yw0, "x", color="green",markersize=5,alpha=1)

        plt.show()
        plt.close(fig)
        
        # Radius vs Height
        
        plt.plot(r_sup_m,h_sup_m,'bx')
        plt.plot(r_sup_g,h_sup_g,'rx')
        plt.xlabel("R [au]")
        plt.ylabel("h CO [au]")
        plt.title(" Radius vs heigh for channel " + str(n_test))
        plt.show()
#      
#        # Raduis vs velocity
#      
        plt.plot(r_sup_g,v_sup_g/1000)
        plt.xlabel("R [au]")
        plt.ylabel("v [km/s]")
        plt.title(" Radius vs speed for channel " + str(n_test))
        plt.show()

        
        

#        # Radius vs sigma
#        
#        plt.plot(r_sup,sigma_sb,'x')
#        plt.xlabel("R [au]")
#        plt.ylabel("Sigma")
#        plt.title(" Radius vs sigma for channel " + str(n_test))
#        plt.show()
#
#        plt.plot(r_sup,sigma_sf,'x')
#        plt.xlabel("R [au]")
#        plt.ylabel("Sigma")
#        plt.title(" Radius vs sigma for channel " + str(n_test))
#        plt.show()

    def HeightCO_gauss(self):
        """ Computes the height of CO in the disk. Uses the gaussian fit"""
        
        delta=50                                     
                        
        r_sup_avg=[]
        r_sup_std=[]
        h_sup_avg=[]
        h_sup_std=[]
        r_inf_avg=[]
        r_inf_std=[]
        h_inf_avg=[]
        h_inf_std=[]

        for i in range(int(len(self.r_sup_g)/delta)):
            # Sup surface
            r_sup_avg.append(np.mean(self.r_sup_g[i*delta:(i+1)*delta]))
            r_sup_std.append(np.std(self.r_sup_g[i*delta:(i+1)*delta]))
            h_sup_avg.append(np.mean(self.h_sup_g[i*delta:(i+1)*delta]))
            h_sup_std.append(np.std(self.h_sup_g[i*delta:(i+1)*delta]))
            
            
            # Inf surface
        for i in range(int(len(self.r_inf_g)/delta)):
            r_inf_avg.append(np.mean(self.r_inf_g[i*delta:(i+1)*delta]))
            r_inf_std.append(np.std(self.r_inf_g[i*delta:(i+1)*delta]))
            h_inf_avg.append(np.mean(self.h_inf_g[i*delta:(i+1)*delta]))
            h_inf_std.append(np.std(self.h_inf_g[i*delta:(i+1)*delta]))
                                                    
        r_sup_avg=np.array(r_sup_avg)
        r_sup_std=np.array(r_sup_std)
        h_sup_avg=np.array(h_sup_avg)
        h_sup_std=np.array(h_sup_std)

        r_inf_avg=np.array(r_inf_avg)
        r_inf_std=np.array(r_inf_std)
        h_inf_avg=np.array(h_inf_avg)
        h_inf_std=np.array(h_inf_std)
        
        fig_height=plt.figure()


        plt.errorbar(r_sup_avg,h_sup_avg,h_sup_std,r_sup_std,fmt='b+')   
        if self.isinf:                                   
            plt.errorbar(r_inf_avg,h_inf_avg,h_inf_std,r_inf_std,fmt='r+')
        for ring in self.pos_rings:
            plt.axvline(x=ring,color='black',linestyle='--')
        plt.xlabel("R [au]")
        plt.ylabel("h CO [au]")
        plt.grid()
        plt.show()
        plt.close(fig_height)
                
        save=pd.DataFrame({'r_sup_avg':r_sup_avg,'r_sup_std':r_sup_std,'h_sup_avg':h_sup_avg,'h_sup_std':h_sup_std})
        save.to_csv(self.path+self.obj+'.r_h_gauss_sup.txt',index=False)
        save=pd.DataFrame({'r_inf_avg':r_inf_avg,'r_inf_std':r_inf_std,'h_inf_avg':h_inf_avg,'h_inf_std':h_inf_std})
        save.to_csv(self.path+self.obj+'.r_h_gauss_inf.txt',index=False)
        return (r_sup_avg,r_sup_std,h_sup_avg,h_sup_std,r_inf_avg,r_inf_std,h_inf_avg,h_inf_std)
    
    def HeightCO_max(self):
        """ Computes the height of CO. Uses the max""" 
        delta=50                                      
                        
        r_sup_avg=[]
        r_sup_std=[]
        h_sup_avg=[]
        h_sup_std=[]
        r_inf_avg=[]
        r_inf_std=[]
        h_inf_avg=[]
        h_inf_std=[]

        for i in range(int(len(self.r_sup_m)/delta)):
            # Sup surface
            r_sup_avg.append(np.mean(self.r_sup_m[i*delta:(i+1)*delta]))
            r_sup_std.append(np.std(self.r_sup_m[i*delta:(i+1)*delta]))
            h_sup_avg.append(np.mean(self.h_sup_m[i*delta:(i+1)*delta]))
            h_sup_std.append(np.std(self.h_sup_m[i*delta:(i+1)*delta]))
            
            
            # Inf surface
        for i in range(int(len(self.r_inf_m)/delta)):
            r_inf_avg.append(np.mean(self.r_inf_m[i*delta:(i+1)*delta]))
            r_inf_std.append(np.std(self.r_inf_m[i*delta:(i+1)*delta]))
            h_inf_avg.append(np.mean(self.h_inf_m[i*delta:(i+1)*delta]))
            h_inf_std.append(np.std(self.h_inf_m[i*delta:(i+1)*delta]))
            
                

                           
        r_sup_avg=np.array(r_sup_avg)
        r_sup_std=np.array(r_sup_std)
        h_sup_avg=np.array(h_sup_avg)
        h_sup_std=np.array(h_sup_std)

        r_inf_avg=np.array(r_inf_avg)
        r_inf_std=np.array(r_inf_std)
        h_inf_avg=np.array(h_inf_avg)
        h_inf_std=np.array(h_inf_std)
        
        fig_height=plt.figure()


        plt.errorbar(r_sup_avg,h_sup_avg,h_sup_std,r_sup_std,fmt='b+')   
        if self.isinf:                                   
            plt.errorbar(r_inf_avg,h_inf_avg,h_inf_std,r_inf_std,fmt='r+')
        plt.xlabel("R [au]")
        plt.ylabel("h CO [au]")
        ax=plt.gca()
        for ring in self.pos_rings:
            plt.axvline(x=ring,color='black',linestyle='--')
        plt.grid()
        plt.show()
        plt.close(fig_height)
        
        save=pd.DataFrame({'r_sup_avg':r_sup_avg,'r_sup_std':r_sup_std,'h_sup_avg':h_sup_avg,'h_sup_std':h_sup_std})
        save.to_csv(self.path+self.obj+'.r_h_max_sup.txt',index=False)
        save=pd.DataFrame({'r_inf_avg':r_inf_avg,'r_inf_std':r_inf_std,'h_inf_avg':h_inf_avg,'h_inf_std':h_inf_std})
        save.to_csv(self.path+self.obj+'.r_h_max_inf.txt',index=False)
        return (r_sup_avg,r_sup_std,h_sup_avg,h_sup_std,r_inf_avg,r_inf_std,h_inf_avg,h_inf_std)
    
    def Speed_max(self):
        """ Computes the speed in the disk. Uses the max"""
        
        delta=50                                         
                        
        r_sup_avg=[]
        r_sup_std=[]
        v_sup_avg=[]
        v_sup_std=[]
        r_inf_avg=[]
        r_inf_std=[]
        v_inf_avg=[]
        v_inf_std=[]

        for i in range(int(len(self.r_sup_g)/delta)):
            # Sup surface
            r_sup_avg.append(np.mean(self.r_sup_m[i*delta:(i+1)*delta]))
            r_sup_std.append(np.std(self.r_sup_m[i*delta:(i+1)*delta]))
            v_sup_avg.append(np.mean(self.v_sup_m[i*delta:(i+1)*delta]))
            v_sup_std.append(np.std(self.v_sup_m[i*delta:(i+1)*delta]))         
            
            # Inf surface
        for i in range(int(len(self.r_inf_m)/delta)):
            r_inf_avg.append(np.mean(self.r_inf_m[i*delta:(i+1)*delta]))
            r_inf_std.append(np.std(self.r_inf_m[i*delta:(i+1)*delta]))
            v_inf_avg.append(np.mean(self.v_inf_m[i*delta:(i+1)*delta]))
            v_inf_std.append(np.std(self.v_inf_m[i*delta:(i+1)*delta]))
                           
        r_sup_avg=np.array(r_sup_avg)
        r_sup_std=np.array(r_sup_std)
        v_sup_avg=np.array(v_sup_avg)
        v_sup_std=np.array(v_sup_std)

        r_inf_avg=np.array(r_inf_avg)
        r_inf_std=np.array(r_inf_std)
        v_inf_avg=np.array(v_inf_avg)
        v_inf_std=np.array(v_inf_std)
        
        fig_height=plt.figure()
        plt.errorbar(r_sup_avg,v_sup_avg/1000,v_sup_std/1000,r_sup_std,fmt='b+')   
        M, dM=self.KeplerianFit(r_sup_avg,v_sup_avg,v_sup_std)
        rmin=min(r_sup_avg)
        rmax=max(r_sup_avg)
        R=np.linspace(rmin,rmax,200)
        plt.plot(R,self.v_kep(R,M)/1000,'c-')  
        print('mass of the star: ',M, '+/-', dM , ' Msun')
                               
        if self.isinf:                                   
            plt.errorbar(r_inf_avg,v_inf_avg/1000,v_inf_std/1000,r_inf_std,fmt='r+')
            Mi, dMi=self.KeplerianFit(r_inf_avg,v_inf_avg,v_inf_std)
            R=np.linspace(rmin,rmax,200)
            plt.plot(R,self.v_kep(R,Mi)/1000,'y-')   
            print('mass of the star: ',Mi, '+/-', dMi , ' Msun')
        for ring in self.pos_rings:
            plt.axvline(x=ring,color='black',linestyle='--')
        plt.xlabel("R [au]")
        plt.ylabel("v [km/s]")
        plt.grid()
        plt.show()
        plt.close(fig_height)
                
        save=pd.DataFrame({'r_sup_avg':r_sup_avg,'.r_sup_std':r_sup_std,'v_sup_avg':v_sup_avg,'v_sup_std':v_sup_std})
        save.to_csv(self.path+self.obj+'.r_v_max_sup.txt',index=False)
        save=pd.DataFrame({'r_inf_avg':r_inf_avg,'r_inf_std':r_inf_std,'v_inf_avg':v_inf_avg,'v_inf_std':v_inf_std})
        save.to_csv(self.path+self.obj+'.r_v_max_inf.txt',index=False)

        self.mass_max=M
        self.cov_mass_max=dM
    def Speed_gauss(self):
        """ Computes the speed in the disk. Uses the gaussian fit"""
        
        delta=50                                          
                        
        r_sup_avg=[]
        r_sup_std=[]
        v_sup_avg=[]
        v_sup_std=[]
        r_inf_avg=[]
        r_inf_std=[]
        v_inf_avg=[]
        v_inf_std=[]

        for i in range(int(len(self.r_sup_g)/delta)):
            # Sup surface
            r_sup_avg.append(np.mean(self.r_sup_g[i*delta:(i+1)*delta]))
            r_sup_std.append(np.std(self.r_sup_g[i*delta:(i+1)*delta]))
            v_sup_avg.append(np.mean(self.v_sup_g[i*delta:(i+1)*delta]))
            v_sup_std.append(np.std(self.v_sup_g[i*delta:(i+1)*delta]))         
            
            # Inf surface
        for i in range(int(len(self.r_inf_g)/delta)):
            r_inf_avg.append(np.mean(self.r_inf_g[i*delta:(i+1)*delta]))
            r_inf_std.append(np.std(self.r_inf_g[i*delta:(i+1)*delta]))
            v_inf_avg.append(np.mean(self.v_inf_g[i*delta:(i+1)*delta]))
            v_inf_std.append(np.std(self.v_inf_g[i*delta:(i+1)*delta]))
                           
        r_sup_avg=np.array(r_sup_avg)
        r_sup_std=np.array(r_sup_std)
        v_sup_avg=np.array(v_sup_avg)
        v_sup_std=np.array(v_sup_std)

        r_inf_avg=np.array(r_inf_avg)
        r_inf_std=np.array(r_inf_std)
        v_inf_avg=np.array(v_inf_avg)
        v_inf_std=np.array(v_inf_std)
        
        fig_height=plt.figure()
        plt.errorbar(r_sup_avg,v_sup_avg/1000,v_sup_std/1000,r_sup_std,fmt='b+')   
        M, dM=self.KeplerianFit(r_sup_avg,v_sup_avg,v_sup_std)
        rmin=min(r_sup_avg)
        rmax=max(r_sup_avg)
        R=np.linspace(rmin,rmax,200)
        plt.plot(R,self.v_kep(R,M)/1000,'c-')  
        print('mass of the star: ',M, '+/-', dM , ' Msun')
                               
        if self.isinf:                                   
            plt.errorbar(r_inf_avg,v_inf_avg/1000,v_inf_std/1000,r_inf_std,fmt='r+')
            Mi, dMi=self.KeplerianFit(r_inf_avg,v_inf_avg,v_inf_std)
            R=np.linspace(rmin,rmax,200)
            plt.plot(R,self.v_kep(R,Mi)/1000,'y-')   
            print('mass of the star: ',Mi, '+/-', dMi , ' Msun')
        for ring in self.pos_rings:
            plt.axvline(x=ring,color='black',linestyle=':')
        for gap in self.pos_gaps:
            plt.axvline(x=gap,color='black',linestyle='--')
        plt.xlabel("R [au]")
        plt.ylabel("v [km/s]")
        plt.grid()
        
        plt.show()
        plt.close(fig_height)
                
        save=pd.DataFrame({'r_sup_avg':r_sup_avg,'r_sup_std':r_sup_std,'v_sup_avg':v_sup_avg,'v_sup_std':v_sup_std})
        save.to_csv(self.path+self.obj+'.r_v_gauss_sup.txt',index=False)
        save=pd.DataFrame({'r_inf_avg':r_inf_avg,'r_inf_std':r_inf_std,'v_inf_avg':v_inf_avg,'v_inf_std':v_inf_std})
        save.to_csv(self.path+self.obj+'.r_v_gauss_inf.txt',index=False)

        self.mass_gauss=M
        self.cov_mass_gauss=dM
    
    def KeplerianFit(self,r,v,sigma):
        r=r[15:-5]
        v=v[15:-5]
        sigma=sigma[15:-5]
        M, pcov = curve_fit(self.v_kep, r, v,sigma=sigma)#, sigma=speed_std)
        dM=np.sqrt(np.diag(pcov))
        return M, dM
    
    def Speed_max_corr(self):
        """ Computes the speed in the disk. Uses the max. Keplerian velocity with z correction"""
        
        delta=50                                          
                        
        r_avg=[]
        r_std=[]
        v_avg=[]
        v_std=[]
        h_avg=[]
        h_std=[]

        for i in range(int(len(self.r_sup_m)/delta)):
            # Sup surface
            r_avg.append(np.mean(self.r_sup_m[i*delta:(i+1)*delta]))
            r_std.append(np.std(self.r_sup_m[i*delta:(i+1)*delta]))
            v_avg.append(np.mean(self.v_sup_m[i*delta:(i+1)*delta]))
            v_std.append(np.std(self.v_sup_m[i*delta:(i+1)*delta]))    
            h_avg.append(np.mean(self.h_sup_m[i*delta:(i+1)*delta]))
            h_std.append(np.std(self.h_sup_m[i*delta:(i+1)*delta]))  
            

        r_avg=np.array(r_avg)
        r_std=np.array(r_std)
        v_avg=np.array(v_avg)
        v_std=np.array(v_std)
        h_avg=np.array(h_avg)
        h_std=np.array(h_std)


        fig_speed=plt.figure()
        plt.errorbar(r_avg,v_avg/1000,v_std/1000,r_std,fmt='b+')   
        rmin=min(r_avg)
        rmax=max(r_avg)
        M, dM=self.KeplerianFitCorr(r_avg,h_avg,v_avg,v_std)
        R=np.linspace(rmin,rmax,200)
        self.interp_h_fct_r=interp1d(r_avg,h_avg,kind='cubic')
        plt.plot(R,self.v_kep_corr_2(R,self.interp_h_fct_r(R),M)/1000,'c-')  
        print('mass of the star: ',M, '+/-', dM , ' Msun')
        for ring in self.pos_rings:
            plt.axvline(x=ring,color='black',linestyle=':')
        for gap in self.pos_gaps:
            plt.axvline(x=gap,color='black',linestyle='--')
                               
        plt.xlabel("R [au]")
        plt.ylabel("v [km/s]")
        plt.grid()
        plt.show()
        plt.close(fig_speed)

        self.mass_max_zcorr=M
        self.cov_mass_max_zcorr=dM

    def Speed_gauss_corr(self):
        """ Computes the speed in the disk. Uses the gaussian fit. Keplerian velocity with z correction"""
        
        delta=50                                          
                        
        r_avg=[]
        r_std=[]
        v_avg=[]
        v_std=[]
        h_avg=[]
        h_std=[]

        for i in range(int(len(self.r_sup_g)/delta)):
            # Sup surface
            r_avg.append(np.mean(self.r_sup_g[i*delta:(i+1)*delta]))
            r_std.append(np.std(self.r_sup_g[i*delta:(i+1)*delta]))
            v_avg.append(np.mean(self.v_sup_g[i*delta:(i+1)*delta]))
            v_std.append(np.std(self.v_sup_g[i*delta:(i+1)*delta]))    
            h_avg.append(np.mean(self.h_sup_g[i*delta:(i+1)*delta]))
            h_std.append(np.std(self.h_sup_g[i*delta:(i+1)*delta]))  
            

        r_avg=np.array(r_avg)
        r_std=np.array(r_std)
        v_avg=np.array(v_avg)
        v_std=np.array(v_std)
        h_avg=np.array(h_avg)
        h_std=np.array(h_std)


        fig_speed=plt.figure()
        plt.errorbar(r_avg,v_avg/1000,v_std/1000,r_std,fmt='b+')   
        rmin=min(r_avg)
        rmax=max(r_avg)
        M, dM=self.KeplerianFitCorr(r_avg,h_avg,v_avg,v_std)
        R=np.linspace(rmin,rmax,200)
        self.interp_h_fct_r=interp1d(r_avg,h_avg,kind='cubic')
        plt.plot(R,self.v_kep_corr_2(R,self.interp_h_fct_r(R),M)/1000,'c-')  
        print('mass of the star: ',M, '+/-', dM , ' Msun')
        for ring in self.pos_rings:
            plt.axvline(x=ring,color='black',linestyle=':')
        for gap in self.pos_gaps:
            plt.axvline(x=gap,color='black',linestyle='--')
                               
        plt.xlabel("R [au]")
        plt.ylabel("v [km/s]")
        plt.grid()

        plt.show()

        plt.close(fig_speed)

        self.mass_gauss_zcorr=M
        self.cov_mass_gauss_zcorr=dM
                        
    def KeplerianFitCorr(self,r,z,v,sigma):
        r=r[15:-5]
        z=z[15:-5]
        v=v[15:-5]
        sigma=sigma[15:-5]
        self.interp_h_fct_r=interp1d(r,z,kind='cubic')
        M, pcov=curve_fit(self.v_kep_corr_1,r, v,sigma=sigma)
        dM=np.sqrt(np.diag(pcov))
        return M, dM
        
    def Temperature_gauss(self):
        """ Uses the gaussian fit"""

        delta=50                                                
                        
        r_sup_b_avg=[]
        r_sup_b_std=[]
        r_sup_f_avg=[]
        r_sup_f_std=[]
        T_sup_b_max=[]
        T_sup_b_std=[]
        T_sup_f_max=[]
        T_sup_f_std=[]
        r_inf_b_avg=[]
        r_inf_b_std=[]
        r_inf_f_avg=[]
        r_inf_f_std=[]
        T_inf_b_max=[]
        T_inf_b_std=[]
        T_inf_f_max=[]
        T_inf_f_std=[]
        
        for i in range(int(len(self.r_sup_g)/delta)):
            # Sup surface
            r_sup_b_avg.append(np.mean(self.r_sup_g[i*delta:(i+1)*delta]))
            r_sup_b_std.append(np.std(self.r_sup_g[i*delta:(i+1)*delta]))
            r_sup_f_avg.append(np.mean(self.r_sup_g[i*delta:(i+1)*delta]))
            r_sup_f_std.append(np.std(self.r_sup_g[i*delta:(i+1)*delta]))
            
            T_sup_b_max.append(np.max(self.Tb_sup_back[i*delta:(i+1)*delta]))
            T_sup_b_std.append(np.std(self.Tb_sup_back[i*delta:(i+1)*delta]))  
            T_sup_f_max.append(np.max(self.Tb_sup_front[i*delta:(i+1)*delta]))
            T_sup_f_std.append(np.std(self.Tb_sup_front[i*delta:(i+1)*delta])) 
            
            # Inf surface
        for i in range(int(len(self.r_inf_g)/delta)):
            r_inf_b_avg.append(np.mean(self.r_inf_g[i*delta:(i+1)*delta]))
            r_inf_b_std.append(np.std(self.r_inf_g[i*delta:(i+1)*delta]))
            r_inf_f_avg.append(np.max(self.r_inf_g[i*delta:(i+1)*delta]))
            r_inf_f_std.append(np.std(self.r_inf_g[i*delta:(i+1)*delta]))
            
            T_inf_b_max.append(np.max(self.Tb_inf_back[i*delta:(i+1)*delta]))
            T_inf_b_std.append(np.std(self.Tb_inf_back[i*delta:(i+1)*delta]))  
            T_inf_f_max.append(np.max(self.Tb_inf_front[i*delta:(i+1)*delta]))
            T_inf_f_std.append(np.std(self.Tb_inf_front[i*delta:(i+1)*delta])) 
                           
        r_sup_b_avg=np.array(r_sup_b_avg)
        r_sup_b_std=np.array(r_sup_b_std)
        r_sup_f_avg=np.array(r_sup_f_avg)
        r_sup_f_std=np.array(r_sup_f_std)
        T_sup_b_max=np.array(T_sup_b_max)
        T_sup_b_std=np.array(T_sup_b_std)
        T_sup_f_max=np.array(T_sup_f_max)
        T_sup_f_std=np.array(T_sup_f_std)
        
        r_inf_b_avg=np.array(r_inf_b_avg)
        r_inf_b_std=np.array(r_inf_b_std)
        r_inf_f_avg=np.array(r_inf_f_avg)
        r_inf_f_std=np.array(r_inf_f_std)
        T_inf_b_max=np.array(T_inf_b_max)
        T_inf_b_std=np.array(T_inf_b_std)
        T_inf_f_max=np.array(T_inf_f_max)
        T_inf_f_std=np.array(T_inf_f_std)       
        
        fig_temp=plt.figure()
        plt.errorbar(r_sup_b_avg,T_sup_b_max,0*T_sup_b_std,r_sup_b_std,fmt='k+')                                      
        plt.errorbar(r_sup_f_avg,T_sup_f_max,0*T_sup_f_std,r_sup_f_std,fmt='b+')  
        if self.isinf:
            plt.errorbar(r_inf_b_avg,T_inf_b_max,0*T_inf_b_std,r_inf_b_std,fmt='g+')                                      
            plt.errorbar(r_inf_f_avg,T_inf_f_max,0*T_inf_f_std,r_inf_f_std,fmt='r+')                                      
                                    
        plt.xlabel("R [au]")
        plt.ylabel("T [K]")
        for ring in self.pos_rings:
            plt.axvline(x=ring,color='black',linestyle='--')
        plt.grid()
        plt.show()

        plt.close(fig_temp)
        
        save=pd.DataFrame({'r_sup_b_avg':r_sup_b_avg,'r_sup_b_std':r_sup_b_std,'T_sup_b_max':T_sup_b_max,'r_sup_f_avg':r_sup_f_avg,'r_sup_f_std':r_sup_f_std,'T_sup_f_max':T_sup_f_max})
        save.to_csv(self.path+self.obj+'.r_T_sup.txt',index=False)
        save=pd.DataFrame({'r_inf_b_avg':r_inf_b_avg,'r_inf_b_std':r_inf_b_std,'T_inf_b_max':T_inf_b_max,'r_inf_f_avg':r_inf_f_avg,'r_inf_f_std':r_inf_f_std,'T_inf_f_max':T_inf_f_max})
        save.to_csv(self.path+self.obj+'.r_T_inf.txt',index=False)

    def Sigma(self):
        delta=50                                     
                        
        r_sup_avg=[]
        r_sup_std=[]
        sigma_sb_avg=[]
        sigma_sb_std=[]
        sigma_sf_avg=[]
        sigma_sf_std=[]


        for i in range(int(len(self.r_sup_m)/delta)):
            # Sup surface
            r_sup_avg.append(np.mean(self.r_sup_m[i*delta:(i+1)*delta]))
            r_sup_std.append(np.std(self.r_sup_m[i*delta:(i+1)*delta]))
            sigma_sb_avg.append(np.mean(self.sigma_sb[i*delta:(i+1)*delta]))
            sigma_sb_std.append(np.std(self.sigma_sb[i*delta:(i+1)*delta]))
            sigma_sf_avg.append(np.mean(self.sigma_sf[i*delta:(i+1)*delta]))
            sigma_sf_std.append(np.std(self.sigma_sf[i*delta:(i+1)*delta]))
            
        
        fig_height=plt.figure()
        plt.errorbar(r_sup_avg,sigma_sb_avg,sigma_sb_std,r_sup_std,fmt='b+')  
        ax=plt.gca()
        ax.set_ylim((0,30))
        
        plt.errorbar(r_sup_avg,sigma_sf_avg,sigma_sf_std,r_sup_std,fmt='k+')   

        plt.xlabel("R [au]")
        plt.ylabel("Sigma")
        plt.grid()
        tikz_save('sigma.tex')

        plt.show()
        fig_height.savefig(self.path+'sigma.png')
        plt.close(fig_height)
        
        fig=plt.figure()
        plt.plot(r_sup_avg,sigma_sb_std,'b')
        plt.plot(r_sup_avg,sigma_sf_std,'k')
        fig.savefig(self.path+'sigma_std.png')
        plt.grid()
        tikz_save('sigma_std.tex')

        plt.show()
        plt.close(fig)

     


    def Rings(self):
        unproj_img=imresize(self.cont_img,(int(self.ny/np.sin(self.inc)),self.nx),mode='constant') #resize the cont image so match the shape of the CO images
#        plt.imshow(unproj_img)
#        plt.show()
        nx,ny=np.shape(unproj_img)
        Theta=np.linspace(0,2*np.pi,100)
        rr=np.linspace(0,50,100)
        
        self.interp_img=interp2d([j for j in range(ny)],[i for i in range(nx)],unproj_img)
        rad_profile=np.zeros((100,1))
        for theta in Theta:
            tmp=[]
            for r in rr:
                tmp.append(self.interp_img(r*np.cos(theta)+self.xs,r*np.sin(theta)+self.ys/np.sin(self.inc)))
            tmp=np.array(tmp)

            rad_profile=rad_profile+tmp
        mat=np.diff(np.transpose(rad_profile))
        mat[abs(mat)<3e-3 ]=0
        
        pos_rings = (np.diff(np.sign(mat))>0).nonzero()[1] +1 # local max
        pos_gaps = (np.diff(np.sign(mat))<0).nonzero()[1] +1 # local max
        fig=plt.figure()
        plt.plot(rr*self.px_size*self.D,rad_profile*2*np.pi/100)
        plt.xlabel('R [au]')
        for xc in pos_rings:
            plt.axvline(x=rr[xc]*self.px_size*self.D)
        for xc in pos_gaps:
            plt.axvline(x=rr[xc]*self.px_size*self.D)
        plt.show()
        plt.close(fig)
        return rr[pos_rings],rr[pos_gaps]
        
        
        
    def r(self,x,yf,yb):
        """ Radius of the orbit. 
        Args:
            x: x coordinate 
            yf: y coordinate of the front maximum 
            yb: y coordinate of the back maximum
        Units:
            x, yf, yb [pixel]
            r [au]
        """
        yc=(yf+yb)/2
        return ((x-self.xs)**2+((yf-yc)/np.cos(self.inc))**2)**0.5*self.px_size*self.D

    def h(self,yc):
        """ Height of the orbit.
        Args:
            yc = (y_front+y_back)/2
        Units : 
            x, yf, yb [pixel]
            h [au]
        
        """
        return abs((yc-self.ys)/np.sin(self.inc))*self.px_size*self.D
    
    def rotate_vect(self,vect,angle,X):
        """ Rotates the vector vect by the given angle around the point X
        Args:
            vect: vector to be rotated
            angle: angle of rotation
            X: fixed point of the rotation
            
        Units: 
            angle [deg]
        """
        vect=np.array(vect)
        X=np.array(X)
        vect=vect-X
        angle=angle*self.const.deg2rad
        RotMatrix=np.array([[np.cos(angle) , -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        return np.dot(RotMatrix,vect)+X
    
    def v(self,x,yf,yb,v0):
        """ Velocity of the gas around the star.
        Args:
            x: x coordinate 
            yf: y coordinate of the front maximum 
            yb: y coordinate of the back maximum
            v0: velocity of the observer
            
        Units: 
            x, yf, yb [pixel]
            v0, v [m/s]             
        """
        return abs(v0*self.r(x,yf,yb)/((x-self.xs)*np.sin(self.inc)*self.px_size*self.D))

    def v_kep(self,R,M):
        """ Returns the keplerian velocity.
        Args:
            R: Radius of the orbit.
            M: Mass of the star.
        Units:
            R [au]
            M [M_sun]
        """
        return np.sqrt(self.const.G*M*self.const.M_sun/(R*self.const.au))
    
    def v_kep_corr_1(self,R,M):
        """ Returns the rotation velocity with a height correction"""
        Z=self.interp_h_fct_r(R)
        r=R*self.const.au
        z=Z*self.const.au
        return np.sqrt((self.const.G*M*self.const.M_sun*r**2)/((r**2+z**2)**(3/2)))
    
    def v_kep_corr_2(self,R,Z,M):
        """ Returns the rotation velocity with a height correction"""
        r=R*self.const.au
        z=Z*self.const.au
        return np.sqrt((self.const.G*M*self.const.M_sun*r**2)/((r**2+z**2)**(3/2)))

    def posmax(self,L):
        """Returns the index of the maximum of a list, which can contain None type"""
        idx=0
        for i in range(len(L)):
            try:
                if L[i] > L[idx]:
                    idx=i
            except TypeError:
                pass
        return idx
    
    def flux_to_Tbrigh(self,F, wl, BMAJ, BMIN):
        """ Convert Flux density in Jy to brightness temperature [K]
        Args:
            F: array containing the values of fulx
            wl: wavelength (cf fits header)
            BMAJ, BMIN: cf fits header
        Units:
            F [Jy]
            wl [m]
            BMAJ, BMIN [deg] or [arcsec] depending on the version of casa used
            T [K] 
        """
        nu = self.const.c/wl 
        factor = 1e26 # 1 Jy = 10^-26 USI
        if self.casa_version==5:
            conversion_factor = (BMIN * BMAJ *self.const.arcsec**2 *np.pi/4/np.log(2)) # beam 
        elif self.casa_version==4:
            conversion_factor = (BMIN * BMAJ * (3600*self.const.arcsec)**2 * np.pi/4/np.log(2)) # beam 
        T=[]
        for item in F:
            exp_m1 = factor *  conversion_factor * (2*self.const.hp*nu**3)/(item*self.const.c**2)
            if exp_m1 >0:
                hnu_kT =  np.log(exp_m1 + 1)
                T.append(self.const.hp * nu / (hnu_kT * self.const.kB))
            else:
                T.append(None)
        
        return np.array(T) 

    
if __name__=='__main__':
    obj=obj='HD163296_HR_CO'  ######### /!\ TO BE MODIFIED FOR EACH OBJECT  
    measure=MakeMeasurements(obj)
    measure.HeightCO_gauss()
    measure.HeightCO_max()
    measure.Speed_gauss()
    measure.Speed_max()
    measure.Speed_gauss_corr()
    measure.Speed_max_corr()

    measure.Temperature_gauss()
    save_param=pd.DataFrame({'PA':measure.PA+measure.PA_corr,'M_max':measure.mass_max,'cov_M_max':measure.cov_mass_max,'M_gauss':measure.mass_gauss,'cov_M_gauss':measure.cov_mass_gauss,'M_gauss_zcorr':measure.mass_gauss_zcorr,'cov_M_gauss_zcorr':measure.cov_mass_gauss_zcorr,'M_max_zcorr':measure.mass_max_zcorr,"cov_M_max_zcorr":measure.cov_mass_max_zcorr})
    save_param.to_csv(measure.path+measure.obj+'.saved_param.txt',index=False)
    
    
#    CO=measure.CO
#    fig,axs=plt.subplots(5,4,sharex=True, sharey=True)
#    axs=axs.flatten()
#    fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0,wspace=0)
#    for i in range(4,44,2):
#        ax=axs[int(i/2)-2]
#    
#        ax.imshow(CO[i],cmap='afmhot', origin='lower')
#        ax.set_xticks([55,255,455])
#        ax.set_yticks([56,256,456])
#        ax.set_xticklabels([int(i) for i in np.array([-200,0,200])*measure.px_size])
#        ax.set_yticklabels([int(i) for i in np.array([-200,0,200])*measure.px_size])
#        ax.axis('tight')
#        if i ==36:
#            ax.set_xlabel('$\Delta\alpha [\arcsec]$')
#            ax.set_ylabel('$\Delta\delta [\arcsec]$')
#    tikz_save('all_chan.tex')

        
        
