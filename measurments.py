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
import database
import scipy.optimize as scpo
from scipy.ndimage.interpolation import rotate
from scipy.optimize import curve_fit

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
        
        with open(fits_name+ext[0]+".gaussian.co_surf", 'rb') as handle:
            data_sup_back=pickle.load(handle)
        with open(fits_name+ext[1]+".gaussian.co_surf", 'rb') as handle: 
            data_sup_front=pickle.load(handle)
        with open(fits_name+ext[2]+".gaussian.co_surf", 'rb') as handle:
            data_inf_back=pickle.load(handle)
        with open(fits_name+ext[3]+".gaussian.co_surf", 'rb') as handle:
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
        Tests values included in [PA-10,PA+10]
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
                    x=np.linspace(xmin, xmax, 10)
                    yb=interp_sup_back(x)
                    yf=interp_sup_front(x)
                    yc=(yb+yf)/2
                    
                    r_sup=np.concatenate((r_sup,copy(self.r(x,yf,yb))))
                    h_sup=np.concatenate((h_sup,copy(self.h(yc))))
                    
            idx = np.argsort(r_sup)
            r_sup=np.array(r_sup)[idx]
            h_sup=np.array(h_sup)[idx]
            # Dispersion
            
            deltaR=10 # (au)                                                    ######### /!\ TO BE MODIFIED FOR EACH OBJECT
            Rmin=20   # (au)                                                    ######### /!\ TO BE MODIFIED FOR EACH OBJECT
            Rmax=550  # (au)                                                    ######### /!\ TO BE MODIFIED FOR EACH OBJECT
                            
            r_sup_avg=[]
            r_sup_std=[]
            h_sup_avg=[]
            h_sup_std=[]
            
            j=0
            k=0
            
            for i in range(int((Rmax-Rmin)/deltaR)):
                avg_h=[]
                while j<len(r_sup) and r_sup[j]>Rmin+i*deltaR and r_sup[j]<Rmin+(i+1)*deltaR :       
                    avg_h.append(h_sup[j])
                    j+=1
                if len(avg_h)!=0:
                    h_sup_avg.append(np.mean(avg_h))
                    h_sup_std.append(np.std(avg_h)) 
                else:
                    h_sup_avg.append(None)
                    h_sup_std.append(None) 
                # Removing the value with error sup than 3 sigma 
                avg_r=[]
                avg_h=[]
                while k<len(r_sup) and r_sup[k]>Rmin+i*deltaR and r_sup[k]<Rmin+(i+1)*deltaR :  
                    if h_sup_avg[i]!=None and abs(h_sup[k]-h_sup_avg[i])<3*h_sup_std[i]:
                        avg_r.append(r_sup[k])
                        avg_h.append(h_sup[k])
                    k+=1
                if len(avg_h)!=0:
                    r_sup_avg.append(np.mean(avg_r))
                    r_sup_std.append(np.std(avg_r))
                    h_sup_avg.pop()
                    h_sup_std.pop()
                    h_sup_avg.append(np.mean(avg_h))
                    h_sup_std.append(np.std(avg_h))
                else:
                    h_sup_avg.pop()
                    h_sup_std.pop()
                    r_sup_avg.append(None)
                    r_sup_std.append(None)
                    h_sup_avg.append(None)
                    h_sup_std.append(None) 
                avg_r=[]
            
                               
            r_sup_avg=np.array([item for item in r_sup_avg if item!= None])
            r_sup_std=np.array([item for item in r_sup_std if item!= None])
            h_sup_avg=np.array([item for item in h_sup_avg if item!= None])
            h_sup_std=np.array([item for item in h_sup_std if item!= None])
            
            dispersion_h=np.linalg.norm(h_sup_std)
            dispersion.append(dispersion_h)

        # Finding the value of PA that minimize the dispersion of height   
        
        dispersion_interp=interp1d(PA_range, dispersion,kind='cubic') 
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
        self.const=database.CONSTANTS()
        fits_name = data.FITS  
        self.D=data.DIST
        self.inc=data.INC*self.const.deg2rad
        self.path=data.PATH
        
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
        self.BMIN=fh[0].header['BMIN'] # [deg] Beam major axis length
        self.BMAJ=fh[0].header['BMAJ'] # [deg] Beam minor axis length
        self.wl=self.const.c/CRVAL3
        fh.close()
        
        ext = ["_sup_back","_sup_front","_inf_back" ,"_inf_front"]
        
        with open(fits_name+ext[0]+".gaussian.co_surf", 'rb') as handle:
            data_sup_back=pickle.load(handle)
        with open(fits_name+ext[1]+".gaussian.co_surf", 'rb') as handle: 
            data_sup_front=pickle.load(handle)
        with open(fits_name+ext[2]+".gaussian.co_surf", 'rb') as handle:
            data_inf_back=pickle.load(handle)
        with open(fits_name+ext[3]+".gaussian.co_surf", 'rb') as handle:
            data_inf_front=pickle.load(handle)
        
        self.xs,self.ys=data_sup_back.star_center # coordinates of the star's center /!\ IN THE CROPPED IMAGE /!\
        self.PA=data_sup_back.PA
        self.ni=data_sup_back.ni # 1st interesting chan
        nm=data_sup_back.nm # 0 velocity chan
        nf=data_sup_back.nf # last interesting chan
        self.CO=self.CO[self.ni:nf]
        self.xw0, self.yw0 = data_sup_back.window[0] # coordinates of the selected window 
        self.xw1, self.yw1 = data_sup_back.window[1]
        freq= CRVAL3 + CDELT3*np.arange(self.nv) # freq of the channels
        freq=freq[self.ni:nf] # only the selected chan
        v_syst= -(freq[nm-self.ni]-restfreq)*self.const.c/restfreq # global speed of the system
        self.v_obs = -((freq-restfreq)/restfreq)*self.const.c-v_syst # radial velocity of the channels
        
        self.xs,self.ys=(self.xs+self.xw0,self.ys+self.yw0)
        self.X=(self.nx/2,self.ny/2)
        self.xs,self.ys=self.rotate_vect((self.xs,self.ys),self.PA_corr,self.X)
        self.pos_max_sup_back=np.array(data_sup_back.pos_maxima)
        self.pos_max_sup_front=np.array(data_sup_front.pos_maxima)
        self.pos_max_inf_back=np.array(data_inf_back.pos_maxima)
        self.pos_max_inf_front=np.array(data_inf_front.pos_maxima)
        self.n=len(self.pos_max_sup_back)  
        
        self.max_sup_back=np.array(data_sup_back.max_value)
        self.max_sup_front=np.array(data_sup_front.max_value)
        self.max_inf_back=np.array(data_inf_back.max_value)
        self.max_inf_front=np.array(data_inf_front.max_value)
        
        self.RotateData(self.PA_corr)
        
    def RotateData(self,angle):
        """ Rotates the data by the angle PA_corr [deg] """
        angle=self.PA_corr
        r_sup=np.array([])
        h_sup=np.array([])
        v_sup=np.array([])
        r_inf=np.array([])
        h_inf=np.array([])
        v_inf=np.array([])
        Tb_sup_front=np.array([])
        Tb_sup_back=np.array([])
        Tb_inf_front=np.array([])
        Tb_inf_back=np.array([])
        for i in range(self.n):
            v0=self.v_obs[i]
            # Sup surface:
            if len(self.pos_max_sup_back[i])!=0 and len(self.pos_max_sup_front[i])!=0:
                temp_sb=[]
                temp_sf=[]
                T_temp=np.array(self.max_sup_back[i])
                for item in self.pos_max_sup_back[i]:
                    x,y=item
                    x,y=(x+self.xw0,y+self.yw0)
                    item=(x,y)
                    item_rot=self.rotate_vect(item, angle,self.X)
                    temp_sb.append(item_rot)
                temp_sb=np.array(temp_sb)
                interp_sup_back=interp1d(temp_sb[:,0],temp_sb[:,1], kind='cubic')
                interp_T_sup_back=interp1d(temp_sb[:,0],T_temp[:,1],kind='cubic')
                
                T_temp=np.array(self.max_sup_front[i])
                for item in self.pos_max_sup_front[i]:
                    x,y=item
                    x,y=(x+self.xw0,y+self.yw0)
                    item=(x,y)
                    item_rot=self.rotate_vect(item, angle,self.X)
                    temp_sf.append(item_rot)
                temp_sf=np.array(temp_sf)
                interp_sup_front=interp1d(temp_sf[:,0],temp_sf[:,1], kind='cubic')
                interp_T_sup_front=interp1d(temp_sf[:,0],T_temp[:,1],kind='cubic')

                xmin=max(np.min(temp_sf[:,0]),np.min(temp_sb[:,0]))
                xmax=min(np.max(temp_sf[:,0]),np.max(temp_sb[:,0]))
                x=np.linspace(xmin, xmax, 100)
                yb=interp_sup_back(x)
                yf=interp_sup_front(x)
                yc=(yb+yf)/2
                
                r_sup=np.concatenate((r_sup,copy(self.r(x,yf,yb))))
                h_sup=np.concatenate((h_sup,copy(self.h(yc))))
                v_sup=np.concatenate((v_sup,copy(self.v(x,yf,yb,v0))))
                flux=interp_T_sup_back(x)
                Tb_sup_back=np.concatenate((Tb_sup_back,copy(self.flux_to_Tbrigh(flux,self.wl,self.BMAJ,self.BMIN))))
                flux=interp_T_sup_front(x)
                Tb_sup_front=np.concatenate((Tb_sup_front,copy(self.flux_to_Tbrigh(flux,self.wl,self.BMAJ,self.BMIN))))

            # Inf surface:
            if len(self.pos_max_inf_back[i])!=0 and len(self.pos_max_inf_front[i])!=0:
                temp_ib=[]
                temp_if=[]
                T_temp=np.array(self.max_inf_back[i])
                for item in self.pos_max_inf_back[i]:
                    x,y=item
                    x,y=(x+self.xw0,y+self.yw0)
                    item=(x,y)
                    item_rot=self.rotate_vect(item, angle,self.X)
                    temp_ib.append(item_rot)
                temp_ib=np.array(temp_ib)
                interp_inf_back=interp1d(temp_ib[:,0],temp_ib[:,1], kind='cubic')
                interp_T_inf_back=interp1d(temp_ib[:,0],T_temp[:,1],kind='cubic')

                T_temp=np.array(self.max_inf_front[i])              
                for item in self.pos_max_inf_front[i]:
                    x,y=item
                    x,y=(x+self.xw0,y+self.yw0)
                    item=(x,y)
                    item_rot=self.rotate_vect(item, angle,self.X)
                    temp_if.append(item_rot)
                temp_if=np.array(temp_if)
                interp_inf_front=interp1d(temp_if[:,0],temp_if[:,1], kind='cubic')
                interp_T_inf_front=interp1d(temp_if[:,0],T_temp[:,1],kind='cubic')
                
                xmin=max(np.min(temp_if[:,0]),np.min(temp_ib[:,0]))
                xmax=min(np.max(temp_if[:,0]),np.max(temp_ib[:,0]))
                x=np.linspace(xmin, xmax, 100)
                yb=interp_inf_back(x)
                yf=interp_inf_front(x)
                yc=(yb+yf)/2
                
                r_inf=np.concatenate((r_inf,copy(self.r(x,yf,yb))))
                h_inf=np.concatenate((h_inf,copy(self.h(yc))))
                v_inf=np.concatenate((v_inf,copy(self.v(x,yf,yb,v0))))
                flux=interp_T_inf_back(x)
                Tb_inf_back=np.concatenate((Tb_inf_back,copy(self.flux_to_Tbrigh(flux,self.wl,self.BMAJ,self.BMIN))))
                flux=interp_T_inf_front(x)
                Tb_inf_front=np.concatenate((Tb_inf_front,copy(self.flux_to_Tbrigh(flux,self.wl,self.BMAJ,self.BMIN))))

    
        idx = np.argsort(r_sup)
        self.r_sup=np.array(r_sup)[idx]
        self.h_sup=np.array(h_sup)[idx]
        self.v_sup=np.array(v_sup)[idx] 
        self.Tb_sup_back=np.array(Tb_sup_back)[idx]
        self.Tb_sup_front=np.array(Tb_sup_front)[idx]
    
        idx = np.argsort(r_inf)
        self.r_inf=np.array(r_inf)[idx]
        self.h_inf=np.array(h_inf)[idx]
        self.v_inf=np.array(v_inf)[idx]
        self.Tb_inf_back=np.array(Tb_inf_back)[idx]
        self.Tb_inf_front=np.array(Tb_inf_front)[idx]
        
        if len(self.r_inf)==0 :
            self.isinf=False
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
        plt.imshow(img, cmap='afmhot')
        plt.plot(self.xs-self.xw0, self.ys-self.yw0, "*", color="yellow")
        
        if len(self.pos_max_sup_back[n_test])!=0 and len(self.pos_max_sup_front[n_test])!=0:
            temp_sb=[]
            temp_sf=[]
            for item in self.pos_max_sup_back[n_test]:
                x,y=item
                x,y=(x+self.xw0,y+self.yw0)
                item=(x,y)
                item_rot=self.rotate_vect(item, angle,self.X)
                temp_sb.append(item_rot)
            temp_sb=np.array(temp_sb)
            interp_sup_back=interp1d(temp_sb[:,0],temp_sb[:,1], kind='cubic')
            
            for item in self.pos_max_sup_front[n_test]:
                x,y=item
                x,y=(x+self.xw0,y+self.yw0)
                item=(x,y)
                item_rot=self.rotate_vect(item, angle,self.X)
                temp_sf.append(item_rot)
            temp_sf=np.array(temp_sf)
            interp_sup_front=interp1d(temp_sf[:,0],temp_sf[:,1], kind='cubic')

            xmin=max(np.min(temp_sf[:,0]),np.min(temp_sb[:,0]))
            xmax=min(np.max(temp_sf[:,0]),np.max(temp_sb[:,0]))
            x=np.linspace(xmin, xmax, 100)
            yb=interp_sup_back(x)
            yf=interp_sup_front(x)
            plt.plot(x-self.xw0, yb-self.yw0, "x", color="blue")
            plt.plot(x-self.xw0, yf-self.yw0, "x", color="black")
            yc=(yb+yf)/2
            r_sup=self.r(x,yf,yb)
            h_sup=self.h(yc)
            v_sup=self.v(x,yf,yb,v0)
                
            # Inf surface:
        if len(self.pos_max_inf_back[n_test])!=0 and len(self.pos_max_inf_front[n_test])!=0:
            temp_ib=[]
            temp_if=[]
            for item in self.pos_max_inf_back[n_test]:
                x,y=item
                x,y=(x+self.xw0,y+self.yw0)
                item=(x,y)
                item_rot=self.rotate_vect(item, angle,self.X)
                temp_ib.append(item_rot)
            temp_ib=np.array(temp_ib)
            interp_inf_back=interp1d(temp_ib[:,0],temp_ib[:,1], kind='cubic')

            for item in self.pos_max_inf_front[n_test]:
                x,y=item
                x,y=(x+self.xw0,y+self.yw0)
                item=(x,y)
                item_rot=self.rotate_vect(item, angle,self.X)
                temp_if.append(item_rot)
            temp_if=np.array(temp_if)
            interp_inf_front=interp1d(temp_if[:,0],temp_if[:,1], kind='cubic')
            
            xmin=max(np.min(temp_if[:,0]),np.min(temp_ib[:,0]))
            xmax=min(np.max(temp_if[:,0]),np.max(temp_ib[:,0]))
            x=np.linspace(xmin, xmax, 100)
            plt.plot(x-self.xw0, interp_inf_back(x)-self.yw0,"x", color="green")
            plt.plot(x-self.xw0, interp_inf_front(x)-self.yw0, "x", color="red")

        plt.show()
        plt.close(fig)
        
        # Radius vs Height
        
        plt.plot(r_sup,h_sup,'x')
        plt.xlabel("R [au]")
        plt.ylabel("h CO [au]")
        plt.title(" Radius vs heigh for channel " + str(n_test))
        plt.show()
#        
#        # Raduis vs velocity
#        
        plt.plot(r_sup,v_sup/1000)
#        
#        # Fit of the velocity with a keplerian model
#        
#        M, pcov = curve_fit(v_kep, r_sup[n_test], v_sup[n_test])
#        R=np.linspace(50,500,200)
#        plt.plot(R, v_kep(R,M)/1000)
        plt.xlabel("R [au]")
        plt.ylabel("v [km/s]")
        plt.title(" Radius vs speed for channel " + str(n_test))
        plt.show()
#        
#        print(M)


    def HeightCO(self):
        """ Computes the height of CO in the disk. Dispersion measured with 10au bins"""
        
        deltaR=10 # (au)                                                    ######### /!\ TO BE MODIFIED FOR EACH OBJECT
        Rmin=20   # (au)                                                    ######### /!\ TO BE MODIFIED FOR EACH OBJECT
        Rmax=550  # (au)                                                    ######### /!\ TO BE MODIFIED FOR EACH OBJECT
                        
        r_sup_avg=[]
        r_sup_std=[]
        h_sup_avg=[]
        h_sup_std=[]
        r_inf_avg=[]
        r_inf_std=[]
        h_inf_avg=[]
        h_inf_std=[]
        
        j=0
        k=0
        l=0
        m=0
        
        for i in range(int((Rmax-Rmin)/deltaR)):
            avg_h=[]
            # Sup surface
            while j<len(self.r_sup) and self.r_sup[j]>Rmin+i*deltaR and self.r_sup[j]<Rmin+(i+1)*deltaR :       
                avg_h.append(self.h_sup[j])
                j+=1
            if len(avg_h)!=0:
                h_sup_avg.append(np.mean(avg_h))
                h_sup_std.append(np.std(avg_h)) 
            else:
                h_sup_avg.append(None)
                h_sup_std.append(None) 
                
            # Removing the value with error sup than 3 sigma 
            avg_r=[]
            avg_h=[]
            while k<len(self.r_sup) and self.r_sup[k]>Rmin+i*deltaR and self.r_sup[k]<Rmin+(i+1)*deltaR :  
                if h_sup_avg[i]!=None and abs(self.h_sup[k]-h_sup_avg[i])<3*h_sup_std[i]:
                    avg_r.append(self.r_sup[k])
                    avg_h.append(self.h_sup[k])
                k+=1
            if len(avg_h)!=0:
                r_sup_avg.append(np.mean(avg_r))
                r_sup_std.append(np.std(avg_r))
                h_sup_avg.pop()
                h_sup_std.pop()
                h_sup_avg.append(np.mean(avg_h))
                h_sup_std.append(np.std(avg_h))
            else:
                h_sup_avg.pop()
                h_sup_std.pop()
                r_sup_avg.append(None)
                r_sup_std.append(None)
                h_sup_avg.append(None)
                h_sup_std.append(None) 
            
            # Inf surface
            avg_h=[]
            while m<len(self.r_inf) and self.r_inf[m]>Rmin+i*deltaR and self.r_inf[m]<Rmin+(i+1)*deltaR :       
                avg_h.append(self.h_inf[m])
                m+=1
            if len(avg_h)!=0:
                h_inf_avg.append(np.mean(avg_h))
                h_inf_std.append(np.std(avg_h)) 
            else:
                h_inf_avg.append(None)
                h_inf_std.append(None) 
                
            # Removing the value with error inf than 3 sigma 
            avg_r=[]
            avg_h=[]
            while l<len(self.r_inf) and self.r_inf[l]>Rmin+i*deltaR and self.r_inf[l]<Rmin+(i+1)*deltaR :  
                if h_inf_avg[i]!=None and abs(self.h_inf[l]-h_inf_avg[i])<3*h_inf_std[i]:
                    avg_r.append(self.r_inf[l])
                    avg_h.append(self.h_inf[l])
                l+=1
            if len(avg_h)!=0:
                r_inf_avg.append(np.mean(avg_r))
                r_inf_std.append(np.std(avg_r))
                h_inf_avg.pop()
                h_inf_std.pop()
                h_inf_avg.append(np.mean(avg_h))
                h_inf_std.append(np.std(avg_h))
            else:
                h_inf_avg.pop()
                h_inf_std.pop()
                r_inf_avg.append(None)
                r_inf_std.append(None)
                h_inf_avg.append(None)
                h_inf_std.append(None) 
            avg_r=[]
        
                           
        r_sup_avg=np.array([item for item in r_sup_avg if item!= None])
        r_sup_std=np.array([item for item in r_sup_std if item!= None])
        h_sup_avg=np.array([item for item in h_sup_avg if item!= None])
        h_sup_std=np.array([item for item in h_sup_std if item!= None])

        r_inf_avg=np.array([item for item in r_inf_avg if item!= None])
        r_inf_std=np.array([item for item in r_inf_std if item!= None])
        h_inf_avg=np.array([item for item in h_inf_avg if item!= None])
        h_inf_std=np.array([item for item in h_inf_std if item!= None])
        
        fig_height=plt.figure()
        plt.errorbar(r_sup_avg,h_sup_avg,h_sup_std,r_sup_std,fmt='b+')   
        if self.isinf:                                   
            plt.errorbar(r_inf_avg,h_inf_avg,h_inf_std,r_inf_std,fmt='r+')
        plt.xlabel("R [au]")
        plt.ylabel("h CO [au]")
        plt.grid()
        plt.show()
        fig_height.savefig(self.path+'heightCO.png')
        plt.close(fig_height)
                
        return (r_sup_avg,r_sup_std,h_sup_avg,h_sup_std,r_inf_avg,r_inf_std,h_inf_avg,h_inf_std)
    
    def HeightCO_bis(self):
        """ Computes the height of CO in the disk. Dispersion measured with 100pts bins"""
        
        delta=50                                              
                        
        r_sup_avg=[]
        r_sup_std=[]
        h_sup_avg=[]
        h_sup_std=[]
        r_inf_avg=[]
        r_inf_std=[]
        h_inf_avg=[]
        h_inf_std=[]

        for i in range(int(len(self.r_sup)/delta)):
            # Sup surface
            r_sup_avg.append(np.mean(self.r_sup[i*delta:(i+1)*delta]))
            r_sup_std.append(np.std(self.r_sup[i*delta:(i+1)*delta]))
            h_sup_avg.append(np.mean(self.h_sup[i*delta:(i+1)*delta]))
            h_sup_std.append(np.std(self.h_sup[i*delta:(i+1)*delta]))
            
            
            # Inf surface
        for i in range(int(len(self.r_inf)/delta)):
            r_inf_avg.append(np.mean(self.r_inf[i*delta:(i+1)*delta]))
            r_inf_std.append(np.std(self.r_inf[i*delta:(i+1)*delta]))
            h_inf_avg.append(np.mean(self.h_inf[i*delta:(i+1)*delta]))
            h_inf_std.append(np.std(self.h_inf[i*delta:(i+1)*delta]))
            
                

                           
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
        plt.grid()
        plt.show()
        fig_height.savefig(self.path+'heightCO_bis.png')
        plt.close(fig_height)
                
        return (r_sup_avg,r_sup_std,h_sup_avg,h_sup_std,r_inf_avg,r_inf_std,h_inf_avg,h_inf_std)
    
    def Speed(self):
        
        deltaR=10 # (au)                                                    
        Rmin=20   # (au)                                                    
        Rmax=550  # (au)                                                    
                        
        r_sup_avg=[]
        r_sup_std=[]
        v_sup_avg=[]
        v_sup_std=[]
        r_inf_avg=[]
        r_inf_std=[]
        v_inf_avg=[]
        v_inf_std=[]
        
        j=0
        k=0
        l=0
        m=0
        
        for i in range(int((Rmax-Rmin)/deltaR)):
            avg_v=[]
            while j<len(self.r_sup) and self.r_sup[j]>Rmin+i*deltaR and self.r_sup[j]<Rmin+(i+1)*deltaR :       
                avg_v.append(self.v_sup[j])
                j+=1
            if len(avg_v)!=0:
                v_sup_avg.append(np.mean(avg_v))
                v_sup_std.append(np.std(avg_v)) 
            else:
                v_sup_avg.append(None)
                v_sup_std.append(None) 
                
            # Removing the value with error sup than 3 sigma 
            avg_r=[]
            avg_v=[]
            while k<len(self.r_sup) and self.r_sup[k]>Rmin+i*deltaR and self.r_sup[k]<Rmin+(i+1)*deltaR :  
                if v_sup_avg[i]!=None and abs(self.v_sup[k]-v_sup_avg[i])<3*v_sup_std[i]:
                    avg_r.append(self.r_sup[k])
                    avg_v.append(self.v_sup[k])
                k+=1
            if len(avg_v)!=0:
                r_sup_avg.append(np.mean(avg_r))
                r_sup_std.append(np.std(avg_r))
                v_sup_avg.pop()
                v_sup_std.pop()
                v_sup_avg.append(np.mean(avg_v))
                v_sup_std.append(np.std(avg_v))
            else:
                v_sup_avg.pop()
                v_sup_std.pop()
                r_sup_avg.append(None)
                r_sup_std.append(None)
                v_sup_avg.append(None)
                v_sup_std.append(None) 
            avg_r=[]
            
            avg_v=[]
            while l<len(self.r_inf) and self.r_inf[l]>Rmin+i*deltaR and self.r_inf[l]<Rmin+(i+1)*deltaR :       
                avg_v.append(self.v_inf[l])
                l+=1
            if len(avg_v)!=0:
                v_inf_avg.append(np.mean(avg_v))
                v_inf_std.append(np.std(avg_v)) 
            else:
                v_inf_avg.append(None)
                v_inf_std.append(None) 
                
            # Removing the value with error inf than 3 sigma 
            avg_r=[]
            avg_v=[]
            while m<len(self.r_inf) and self.r_inf[m]>Rmin+i*deltaR and self.r_inf[m]<Rmin+(i+1)*deltaR :  
                if v_inf_avg[i]!=None and abs(self.v_inf[m]-v_inf_avg[i])<3*v_inf_std[i]:
                    avg_r.append(self.r_inf[m])
                    avg_v.append(self.v_inf[m])
                m+=1
            if len(avg_v)!=0:
                r_inf_avg.append(np.mean(avg_r))
                r_inf_std.append(np.std(avg_r))
                v_inf_avg.pop()
                v_inf_std.pop()
                v_inf_avg.append(np.mean(avg_v))
                v_inf_std.append(np.std(avg_v))
            else:
                v_inf_avg.pop()
                v_inf_std.pop()
                r_inf_avg.append(None)
                r_inf_std.append(None)
                v_inf_avg.append(None)
                v_inf_std.append(None) 
            avg_r=[]
        
                           
        r_sup_avg=np.array([item for item in r_sup_avg if item!= None])
        r_sup_std=np.array([item for item in r_sup_std if item!= None])
        v_sup_avg=np.array([item for item in v_sup_avg if item!= None])
        v_sup_std=np.array([item for item in v_sup_std if item!= None])

        r_inf_avg=np.array([item for item in r_inf_avg if item!= None])
        r_inf_std=np.array([item for item in r_inf_std if item!= None])
        v_inf_avg=np.array([item for item in v_inf_avg if item!= None])
        v_inf_std=np.array([item for item in v_inf_std if item!= None])
        
        
        fig_speed=plt.figure()
        plt.errorbar(r_sup_avg,v_sup_avg/1000,v_sup_std/1000,r_sup_std,fmt='b+')  
        M, dM=self.KeplerianFit(r_sup_avg,v_sup_avg)
        R=np.linspace(50,500,200)
        plt.plot(R,self.v_kep(R,M)/1000,'c-')  
        print('mass of the star: ',M, '+/-', dM , ' Msun')
                               
        if self.isinf:
            plt.errorbar(r_inf_avg,v_inf_avg/1000,v_inf_std/1000,r_inf_std,fmt='r+')
            M, dM=self.KeplerianFit(r_inf_avg,v_inf_avg)
            R=np.linspace(50,500,200)
            plt.plot(R,self.v_kep(R,M)/1000,'y-')   
            print('mass of the star: ',M, '+/-', dM , ' Msun')

        plt.xlabel("R [au]")
        plt.ylabel("v [km/s]")
        plt.grid()
        plt.show()
        fig_speed.savefig(self.path+'speed.png')
        plt.close(fig_speed)
        return (r_sup_avg,r_sup_std,v_sup_avg,v_sup_std,r_inf_avg,r_inf_std,v_inf_avg,v_inf_std)

    def Speed_bis(self):
        """ Computes the speed in the disk. Dispersion measured with 100pts bins"""
        
        delta=300                                              
                        
        r_sup_avg=[]
        r_sup_std=[]
        v_sup_avg=[]
        v_sup_std=[]
        r_inf_avg=[]
        r_inf_std=[]
        v_inf_avg=[]
        v_inf_std=[]

        for i in range(int(len(self.r_sup)/delta)):
            # Sup surface
            r_sup_avg.append(np.mean(self.r_sup[i*delta:(i+1)*delta]))
            r_sup_std.append(np.std(self.r_sup[i*delta:(i+1)*delta]))
            v_sup_avg.append(np.mean(self.v_sup[i*delta:(i+1)*delta]))
            v_sup_std.append(np.std(self.v_sup[i*delta:(i+1)*delta]))         
            
            # Inf surface
        for i in range(int(len(self.r_inf)/delta)):
            r_inf_avg.append(np.mean(self.r_inf[i*delta:(i+1)*delta]))
            r_inf_std.append(np.std(self.r_inf[i*delta:(i+1)*delta]))
            v_inf_avg.append(np.mean(self.v_inf[i*delta:(i+1)*delta]))
            v_inf_std.append(np.std(self.v_inf[i*delta:(i+1)*delta]))
                           
        r_sup_avg=np.array(r_sup_avg)
        r_sup_std=np.array(r_sup_std)
        v_sup_avg=np.array(v_sup_avg)
        v_sup_std=np.array(v_sup_std)

        r_inf_avg=np.array(r_inf_avg)
        r_inf_std=np.array(r_inf_std)
        v_inf_avg=np.array(v_inf_avg)
        v_inf_std=np.array(v_inf_std)
        
        fig_height=plt.figure()
        plt.errorbar(r_sup_avg,v_sup_avg,v_sup_std,r_sup_std,fmt='b+')   
        if self.isinf:                                   
            plt.errorbar(r_inf_avg,v_inf_avg,v_inf_std,r_inf_std,fmt='r+')
        plt.xlabel("R [au]")
        plt.ylabel("h CO [au]")
        plt.grid()
        plt.show()
        fig_height.savefig(self.path+'heightCO_bis.png')
        plt.close(fig_height)
                
        return (r_sup_avg,r_sup_std,v_sup_avg,v_sup_std,r_inf_avg,r_inf_std,v_inf_avg,v_inf_std)
    
    def KeplerianFit(self,r,v):
        M, pcov = curve_fit(self.v_kep, r, v)#, sigma=speed_std)
        dM=np.sqrt(np.diag(pcov))
        return M, dM

    def KeplerianFitCorr(self,r,v,h):
        M, pcov=curve_fit(self.v_kep_corr,(r,h), v)
        dM=np.sqrt(np.diag(pcov))
        return M, dM
    
    def Temperature(self):
        
        deltaR=10 # (au)                                                    ######### /!\ TO BE MODIFIED FOR EACH OBJECT
        Rmin=20   # (au)                                                    ######### /!\ TO BE MODIFIED FOR EACH OBJECT
        Rmax=550  # (au)                                                    ######### /!\ TO BE MODIFIED FOR EACH OBJECT
                        
        r_sup_b_avg=[]
        r_sup_b_std=[]
        r_sup_f_avg=[]
        r_sup_f_std=[]
        T_sup_back_avg=[]
        T_sup_back_std=[]
        T_sup_front_avg=[]
        T_sup_front_std=[]
        r_inf_b_avg=[]
        r_inf_b_std=[]
        r_inf_f_avg=[]
        r_inf_f_std=[]
        T_inf_back_avg=[]
        T_inf_back_std=[]
        T_inf_front_avg=[]
        T_inf_front_std=[]
        j=0
        k=0
        l=0
        m=0
        n=0
        o=0
        
        for i in range(int((Rmax-Rmin)/deltaR)):
            avg_Tb=[]
            avg_Tf=[]
            while j<len(self.r_sup) and self.r_sup[j]>Rmin+i*deltaR and self.r_sup[j]<Rmin+(i+1)*deltaR :       
                avg_Tb.append(self.Tb_sup_back[j])
                avg_Tf.append(self.Tb_sup_front[j])
                j+=1
            if len(avg_Tb)!=0:
                T_sup_back_avg.append(np.mean(avg_Tb))
                T_sup_back_std.append(np.std(avg_Tb)) 
                T_sup_front_avg.append(np.mean(avg_Tb))
                T_sup_front_std.append(np.std(avg_Tb)) 
            else:
                T_sup_back_avg.append(None)
                T_sup_back_std.append(None)
                T_sup_front_avg.append(None)
                T_sup_front_std.append(None)
                
            # Removing the value with error sup than 3 sigma 
            avg_r=[]
            avg_Tb=[]
            avg_Tf=[]
            while k<len(self.r_sup) and self.r_sup[k]>Rmin+i*deltaR and self.r_sup[k]<Rmin+(i+1)*deltaR :  
                if T_sup_back_avg[i]!=None and abs(self.Tb_sup_back[k]-T_sup_back_avg[i])<3*T_sup_back_std[i]:
                    avg_r.append(self.r_sup[k])
                    avg_Tb.append(self.Tb_sup_back[k])
                k+=1
            if len(avg_Tb)!=0:
                r_sup_b_avg.append(np.mean(avg_r))
                r_sup_b_std.append(np.std(avg_r))
                T_sup_back_avg.pop()
                T_sup_back_std.pop()
                T_sup_back_avg.append(np.mean(avg_Tb))
                T_sup_back_std.append(np.std(avg_Tb))
            else:
                T_sup_back_avg.pop()
                T_sup_back_std.pop()
                r_sup_b_avg.append(None)
                r_sup_b_std.append(None)
                T_sup_back_avg.append(None)
                T_sup_back_std.append(None) 
            while l<len(self.r_sup) and self.r_sup[l]>Rmin+i*deltaR and self.r_sup[l]<Rmin+(i+1)*deltaR :  
                if T_sup_front_avg[i]!=None and abs(self.Tb_sup_front[l]-T_sup_front_avg[i])<3*T_sup_front_std[i]:
                    avg_r.append(self.r_sup[l])
                    avg_Tb.append(self.Tb_sup_front[l])
                l+=1
            if len(avg_Tb)!=0:
                r_sup_f_avg.append(np.mean(avg_r))
                r_sup_f_std.append(np.std(avg_r))
                T_sup_front_avg.pop()
                T_sup_front_std.pop()
                T_sup_front_avg.append(np.mean(avg_Tb))
                T_sup_front_std.append(np.std(avg_Tb))
            else:
                T_sup_front_avg.pop()
                T_sup_front_std.pop()
                r_sup_f_avg.append(None)
                r_sup_f_std.append(None)
                T_sup_front_avg.append(None)
                T_sup_front_std.append(None) 

            # Inf surface
            
            avg_Tb=[]
            avg_Tf=[]
            while m<len(self.r_inf) and self.r_inf[m]>Rmin+i*deltaR and self.r_inf[m]<Rmin+(i+1)*deltaR :       
                avg_Tb.append(self.Tb_inf_back[m])
                avg_Tf.append(self.Tb_inf_front[m])
                m+=1
            if len(avg_Tb)!=0:
                T_inf_back_avg.append(np.mean(avg_Tb))
                T_inf_back_std.append(np.std(avg_Tb)) 
                T_inf_front_avg.append(np.mean(avg_Tb))
                T_inf_front_std.append(np.std(avg_Tb)) 
            else:
                T_inf_back_avg.append(None)
                T_inf_back_std.append(None)
                T_inf_front_avg.append(None)
                T_inf_front_std.append(None)
                
            # Removing the value with error inf than 3 sigma 
            avg_r=[]
            avg_Tb=[]
            avg_Tf=[]
            while n<len(self.r_inf) and self.r_inf[n]>Rmin+i*deltaR and self.r_inf[n]<Rmin+(i+1)*deltaR :  
                if T_inf_back_avg[i]!=None and abs(self.Tb_inf_back[n]-T_inf_back_avg[i])<3*T_inf_back_std[i]:
                    avg_r.append(self.r_inf[n])
                    avg_Tb.append(self.Tb_inf_back[n])
                n+=1
            if len(avg_Tb)!=0:
                r_inf_b_avg.append(np.mean(avg_r))
                r_inf_b_std.append(np.std(avg_r))
                T_inf_back_avg.pop()
                T_inf_back_std.pop()
                T_inf_back_avg.append(np.mean(avg_Tb))
                T_inf_back_std.append(np.std(avg_Tb))
            else:
                T_inf_back_avg.pop()
                T_inf_back_std.pop()
                r_inf_b_avg.append(None)
                r_inf_b_std.append(None)
                T_inf_back_avg.append(None)
                T_inf_back_std.append(None) 
            while o<len(self.r_inf) and self.r_inf[o]>Rmin+i*deltaR and self.r_inf[o]<Rmin+(i+1)*deltaR :  
                if T_inf_front_avg[i]!=None and abs(self.Tb_inf_front[o]-T_inf_front_avg[i])<3*T_inf_front_std[i]:
                    avg_r.append(self.r_inf[o])
                    avg_Tb.append(self.Tb_inf_front[o])
                o+=1
            if len(avg_Tb)!=0:
                r_inf_f_avg.append(np.mean(avg_r))
                r_inf_f_std.append(np.std(avg_r))
                T_inf_front_avg.pop()
                T_inf_front_std.pop()
                T_inf_front_avg.append(np.mean(avg_Tb))
                T_inf_front_std.append(np.std(avg_Tb))
            else:
                T_inf_front_avg.pop()
                T_inf_front_std.pop()
                r_inf_f_avg.append(None)
                r_inf_f_std.append(None)
                T_inf_front_avg.append(None)
                T_inf_front_std.append(None) 
                           
        r_sup_b_avg=np.array([item for item in r_sup_b_avg if item!= None])
        r_sup_b_std=np.array([item for item in r_sup_b_std if item!= None])
        r_sup_f_avg=np.array([item for item in r_sup_f_avg if item!= None])
        r_sup_f_std=np.array([item for item in r_sup_f_std if item!= None])
        T_sup_back_avg=np.array([item for item in T_sup_back_avg if item!= None])
        T_sup_back_std=np.array([item for item in T_sup_back_std if item!= None])
        T_sup_front_avg=np.array([item for item in T_sup_front_avg if item!= None])
        T_sup_front_std=np.array([item for item in T_sup_front_std if item!= None])


        r_inf_b_avg=np.array([item for item in r_inf_b_avg if item!= None])
        r_inf_b_std=np.array([item for item in r_inf_b_std if item!= None])
        r_inf_f_avg=np.array([item for item in r_inf_f_avg if item!= None])
        r_inf_f_std=np.array([item for item in r_inf_f_std if item!= None])
        T_inf_back_avg=np.array([item for item in T_inf_back_avg if item!= None])
        T_inf_back_std=np.array([item for item in T_inf_back_std if item!= None])
        T_inf_front_avg=np.array([item for item in T_inf_front_avg if item!= None])
        T_inf_front_std=np.array([item for item in T_inf_front_std if item!= None])
        
        fig_temp=plt.figure()
        plt.errorbar(r_sup_b_avg,T_sup_back_avg,T_sup_back_std,r_sup_b_std,fmt='k+')                                      
        plt.errorbar(r_sup_f_avg,T_sup_front_avg,T_sup_front_std,r_sup_f_std,fmt='b+')  
        if self.isinf:
            plt.errorbar(r_inf_b_avg,T_inf_back_avg,T_inf_back_std,r_inf_b_std,fmt='g+')                                      
            plt.errorbar(r_inf_f_avg,T_inf_front_avg,T_inf_front_std,r_inf_f_std,fmt='r+')                                      
                                    
        plt.xlabel("R [au]")
        plt.ylabel("T [K]")
        plt.grid()
        plt.show()
        fig_temp.savefig(self.path+'temps.png')
        plt.close(fig_temp)
        return (r_sup_b_avg,r_sup_b_std,T_sup_back_avg,T_sup_back_std,r_sup_f_avg,r_sup_f_std,T_sup_front_avg,T_sup_front_std,r_inf_b_avg,r_inf_b_std,T_inf_back_avg,T_inf_back_std,r_inf_f_avg,r_inf_f_std,T_inf_front_avg,T_inf_front_std)

        

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
            r [au]
        
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
    
    def v_kep_corr(self,X,M):
        """ Returns the rotation velocity with a height correction"""
        R,z=X
        r=R*self.const.au
        return np.sqrt(self.const.G*M*r**2)/(r**2+z**2)**(3/2)

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
            BMAJ, BMIN [deg]
            T [K] 
        """
        nu = self.const.c/wl 
        factor = 1e26 # 1 Jy = 10^-26 USI
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
    obj=obj='HD163296'  ######### /!\ TO BE MODIFIED FOR EACH OBJECT  
    measure=MakeMeasurements(obj)
    measure.HeightCO()
    measure.HeightCO_bis()
    measure.Speed()
    measure.Temperature()
    