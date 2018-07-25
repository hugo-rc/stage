# -*- coding: utf-8 -*-
"""
If you encounter "TclError: image "pyimageXX" doesn't exist", reboot the python core (ctrl+. on Spyder)
"""

# =============================================================================
# Packages
# =============================================================================
from astropy.io import fits # Reading fits files
from tkinter import * # graphical interface
from matplotlib.pyplot import imsave
import matplotlib.path as mpp #create polygon
import numpy as np 
from scipy.ndimage.interpolation import rotate
import pickle # binary files
from copy import copy
from scipy.optimize import curve_fit
import database
from skimage.transform import resize as imresize
import matplotlib.pyplot as plt

import cv2

# =============================================================================
# Classes
# =============================================================================

class PA_finder(Tk):
    """ Tool to find the PA of the disk"""
    def __init__(self,parent,fits_name, continuum_fits):
        Tk.__init__(self,parent)    # don't know exactly what it does, 
        self.parent = parent        # but it was recommended to put this
        
        self.continuum_fits=continuum_fits
        ct=fits.open(self.continuum_fits)
        self.cont_img=ct[0].data
        ct.close()
        self.cont_img=self.cont_img[0,0,:,:]
        self.fits_name=fits_name
        fh=fits.open(self.fits_name)
        CO=fh[0].data
        self.CO=CO[0]
        fh.close()
        self.nx=len(self.CO[0][0]) # number of columns
        self.ny=len(self.CO[0]) # number of lines
        self.nv=len(self.CO) #number of chanels
        self.n=int(self.nv/2)
        np.nan_to_num(self.CO,copy=False) # replaces the nan values by 0
        np.nan_to_num(self.cont_img,copy=False) # replaces the nan values by 0
        self.cont_img=imresize(self.cont_img,(self.ny,self.nx),mode='constant') #resize the cont image so match the shape of the CO images

        # User interface:
        
        self.w=max(self.nx,550) # width of the window
        self.h=max(self.ny,550) # height of the window
        
        self.canvas=Canvas(self, width=2*self.w, height=self.h)
        self.canvas.pack()
        
        
        imsave('background.jpg', self.CO[self.n],cmap='afmhot' )
        self.img=PhotoImage(file='background.jpg')
        self.background=self.canvas.create_image(0,0,anchor=NW,image=self.img)
        
        # User information:
        
        self.display_info()
        self.counter = Label(self)
        self.counter.place(x=int(7/6*self.w), y=int(1/4*self.h)-50)
        self.counter.configure(text=str(self.n+1)+"/"+str(self.nv))
        
        self.log = Label(self) 
        self.log.place(x=int(7/6*self.w), y=int(1/4*self.h)+200)
        
        # Interactions:
        
        self.prev_chan_but=Button(self.canvas, text='Previous channel', command=self.prev_chan)
        self.prev_chan_but.place(x=int(7/6*self.w), y=int(1/4*self.h))
        self.next_chan_but=Button(self.canvas, text="Next channel", command=self.next_chan)
        self.next_chan_but.place(x=int(7/6*self.w+200), y=int(1/4*self.h))
        
        self.first_im_but=Button(self.canvas, text="Frist image", command=self.first_img)
        self.first_im_but.place(x=int(7/6*self.w), y=int(1/4*self.h)+50)
        self.mid_im_but=Button(self.canvas, text="Mid image", command=self.mid_img)
        self.mid_im_but.place(x=int(7/6*self.w)+150, y=int(1/4*self.h)+50)
        self.last_im_but=Button(self.canvas, text="Last image", command=self.last_img)
        self.last_im_but.place(x=int(7/6*self.w)+300, y=int(1/4*self.h)+50)
        
        self.PA_entry=Entry(self)
        self.PA_entry.place(x=int(7/6*self.w), y=int(1/4*self.h)+100)
        self.PA_entry.delete(0, END)
        self.PA_entry.insert(0, "PA (deg) ?")
        self.rotate_but=Button(self.canvas, text='Rotate', command=self.rotate_img)
        self.rotate_but.place(x=int(7/6*self.w), y=int(1/4*self.h)+150)
        self.validate_but=Button(self.canvas, text='Validate', command=self.validate)
        self.validate_but.place(x=int(7/6*self.w)+200, y=int(1/4*self.h)+150)
        
    def display_info(self):
        """ User information"""
        S = Scrollbar(self)
        T = Text(self, height=6, width=50)
        S.place(x=int(7/6*self.w)+355, y=int(1/4*self.h)+250)
        T.place(x=int(7/6*self.w), y=int(1/4*self.h)+250)
        S.config(command=T.yview)
        T.config(yscrollcommand=S.set)
        quote = """ INFO : 
        Select the first relevant (not noisy) channel with First image button.
        Find the "flat arms" image, make it rotate to vertical by typing an angle and clicking the Rotate button.
        Once the "arms" are vertical, clic the Validate button (it might take some time). Clic on the Mid Image button.
        Select the last relevant channel with Last image button.
        You can crop the image by clicking on the Crop button.
        Finally, click on the Finish button to close the window.
        """
        T.insert(END, quote)
        T.config(state=DISABLED)

                
    def next_chan(self):
        """Go to the next channel"""
        if self.n<self.nv-1:
            self.n+=1
            self.counter.configure(text=str(self.n+1)+"/"+str(self.nv))
            imsave('background.jpg', self.CO[self.n], cmap='afmhot' )
            self.img=PhotoImage(file='background.jpg')
            self.canvas.itemconfig(self.background,image=self.img)      
        else:
            self.log.configure(text="last channel")
            
    def prev_chan(self):
        """Go to the previous channel"""
        if self.n>0:
            self.n-=1
            self.counter.configure(text=str(self.n+1)+"/"+str(self.nv))
            imsave('background.jpg', self.CO[self.n], cmap='afmhot' )
            self.img=PhotoImage(file='background.jpg')
            self.canvas.itemconfig(self.background,image=self.img)      
        else:
            self.log.configure(text="first channel")
    
    def first_img(self):
        """ Returns the index of the first interesting image"""
        self.log.configure(text=" Index of the first relevant channel saved")
        self.ni=self.n
        
    def mid_img(self):
        """ Returns the index of the "flat arms" image"""
        self.log.configure(text=" Index of the zero velocity channel saved")
        self.nm=self.n

    def last_img(self):
        """ Returns the index of the last interesting image"""
        self.log.configure(text=" Index of the last relevant channel saved")
        self.nf=self.n+1
            
    def rotate_img(self):
        """ Rotates the image by angle PA"""
        try:
            self.PA=int(self.PA_entry.get())
            self.log.configure(text="Preview")   
            self.CO_rot=rotate(self.CO[self.n],180-self.PA, reshape=False)
            imsave('background.jpg', self.CO_rot, cmap='afmhot' )
            self.img=PhotoImage(file='background.jpg')
            self.canvas.itemconfig(self.background,image=self.img)
        except ValueError:
            self.log.configure(text="ERROR: Please enter an float or integer")
           
    
    def validate(self):
        """ Validates the choice of PA"""
        self.log.configure(text="PA saved. Clic on Crop to select the area to study")
        self.CO=rotate(self.CO,180-self.PA,axes=(2,1), reshape=False)
        self.cont_img=rotate(self.cont_img,180-self.PA, reshape=False)
        # Removing the previous buttons and creating a new one to crop the image
        self.validate_but.destroy()
        self.rotate_but.destroy()
        self.crop_but=Button(self.canvas, text='Crop', command=self.crop)
        self.crop_but.place(x=int(7/6*self.w), y=int(1/4*self.h)+150)
        self.switch=False
        
        
    def crop(self):
        """ Crops the image to keep the interesting part"""
        if self.switch: # Second clic: crop
            (self.x0,self.x1)=(min(self.x0,self.x1),max(self.x0,self.x1)) # depending on how the rectangle has been selected
            (self.y0,self.y1)=(min(self.y0,self.y1),max(self.y0,self.y1)) # after this, (x0,y0)=Top Left, (x1,y1)=Bottom Right
            self.CO=self.CO[:,self.y0:self.y1,self.x0:self.x1]
            self.cont_img=self.cont_img[self.y0:self.y1,self.x0:self.x1]
            self.crop_but.destroy()
            self.canvas.delete(self.id_rect)
            imsave('background.jpg', self.CO[self.n], cmap='afmhot' )
            self.img=PhotoImage(file='background.jpg')
            self.canvas.itemconfig(self.background,image=self.img)
            self.star_but=Button(self.canvas, text='Find star center', command=self.star)
            self.star_but.place(x=int(7/6*self.w), y=int(1/4*self.h)+150)
        else: # First clic: select area
            checki=False
            checkm=False
            checkf=False
            try:
                self.ni=self.ni*1
                checki=True
            except:
                self.log.configure(text="ERROR: You forgot to define the first channel. \n Define it and press Done again")
            try:
                self.nm=self.nm*1
                checkm=True
            except:
                self.log.configure(text="ERROR: You forgot to define the zero velocity channel. \n Define it and press Done again")
            try:
                self.nf=self.nf*1
                checkf=True
            except:
                self.log.configure(text="ERROR: You forgot to define the last channel. \n Define it and press Done again")
            if checki and checkm and checkf:
                
                imsave('background.jpg', np.sum(copy(self.CO[self.ni:self.nf]),axis=0), cmap='afmhot' )
                self.img=PhotoImage(file='background.jpg')
                self.canvas.itemconfig(self.background,image=self.img)
                self.log.configure(text="Clic and drag to select a rectangle")
                self.canvas.bind("<Button-1>", self.rect_start)
                self.canvas.bind("<ButtonRelease-1>", self.rect_end)
                self.switch=True

    def rect_start(self,event):
        self.x0=event.x
        self.y0=event.y
        r=2
        try:
            self.canvas.delete(self.id_rect)
        except AttributeError:
            pass
        
        if self.x0<self.nx and self.y0<self.ny:         
            self.log.configure(text="Drag to select a rectangle")
            self.id_start=self.canvas.create_oval(self.x0-r, self.y0-r, self.x0+r, self.y0+r, fill = 'white')
        
    def rect_end(self,event):
        self.x1=event.x
        self.y1=event.y
        if self.x1>=self.nx: 
            self.x1=self.nx-1
        if self.y1>=self.ny:
            self.y1=self.ny-1
        if self.x0<self.nx and self.y0<self.ny: 
            self.log.configure(text="Area selected: clic on Crop to validate the selection")
            self.id_rect=self.canvas.create_rectangle(self.x0,self.y0,self.x1,self.y1,outline='white')
            self.canvas.delete(self.id_start)
            
    def gauss2d(self,X, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        x,y=X
        xo = float(xo)
        yo = float(yo)    
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g= offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
        return g.ravel() 
               
    def star(self):
        r=2
        ys1, xs1=np.unravel_index(np.argmax(self.cont_img, axis=None), self.cont_img.shape) # position of the max intensity in the continuum image
        x=np.linspace(0,self.x1-self.x0-1,self.x1-self.x0)
        y=np.linspace(0,self.y1-self.y0-1,self.y1-self.y0)
        x,y=np.meshgrid(x,y)
        initial_guess = (self.cont_img[xs1,ys1],xs1,ys1,10,10,0,0)
        popt,pcov=curve_fit(self.gauss2d,(x,y),self.cont_img.ravel(),p0=initial_guess)
        
        a,xs,ys, sigmax,sigmay,theta,offset=popt
        self.pos_star=(xs,ys)
        
        imsave('background.jpg', self.cont_img, cmap='afmhot' )
        self.img=PhotoImage(file='background.jpg')
        self.canvas.itemconfig(self.background,image=self.img)
        self.id_center=self.canvas.create_oval(xs-r, ys-r, xs+r, ys+r, fill = 'yellow')
        self.quit_but=Button(self.canvas, text='Quit', command=self.finish)
        self.quit_but.place(x=int(7/6*self.w), y=int(1/4*self.h)+150)
        self.ell_but=Button(self.canvas,text='Ellipse fit',command=self.elliptical_center)
        self.ell_but.place(x=int(7/6*self.w)+150, y=int(1/4*self.h)+150)
        self.manual_but=Button(self.canvas,text='Manual selection',command=self.manual_center)
        self.manual_but.place(x=int(7/6*self.w)+300, y=int(1/4*self.h)+150)
        self.star_but.destroy()
        

    def elliptical_center(self):
        r=2
        try:
            self.canvas.delete(self.id_center)
        except :
            pass
        imsave('background.jpg', self.cont_img, cmap='afmhot' )
        img=cv2.imread('background.jpg',0)
        ret,thresh=cv2.threshold(img,10,255,0)
        im2, contours, hierarchy=cv2.findContours(thresh,1,2)
        cnt=contours[0]
        ellipse=cv2.fitEllipse(cnt)
        xs,ys=ellipse[0]
        img=cv2.ellipse(img,ellipse,(255,0,0),2)
        imsave('background.jpg', img, cmap='afmhot' )
        self.img=PhotoImage(file='background.jpg')
        self.canvas.itemconfig(self.background,image=self.img)
        self.id_center=self.canvas.create_oval(xs-r, ys-r, xs+r, ys+r, fill = 'yellow')
        self.pos_star=(xs,ys)
        
        
    def manual_center(self):
        """Place manually the center of the star if the automatic method doesn't work"""
        self.canvas.unbind('<ButtonRelease-1>')
        self.canvas.bind("<Button-1>", self.clic_center)
        
    def clic_center(self,event):
        r=2
        try:
            self.canvas.delete(self.id_center)
        except :
            pass
        xs=event.x
        ys=event.y
        self.id_center=self.canvas.create_oval(xs-r,ys-r,xs+r,ys+r,fill='yellow')
        self.pos_star=(xs,ys)

        
    def finish(self):
        """Quit the window"""
        self.destroy()
            
class maxima_finder(Tk):
    def __init__(self,parent,CO,ni,nm,nf):
        Tk.__init__(self,parent) # don't know exactly what it does, 
        self.parent = parent     # but it was recommended to put this      
        self.ni=ni
        self.nf=nf
        self.CO=CO[ni:nf]
        self.nx=len(self.CO[0][0]) # number of columns
        self.ny=len(self.CO[0]) # number of lines
        self.nv=len(self.CO) #number of chanels
        self.n=0 # initialazing image counter
        self.mouse=[]
        self.id_point=[]
        self.id_point_max=[]
        self.pos_maxima=[]
        self.carac_gauss=[]
        self.storage_carac_gauss=[]
        self.switch=False
        self.id_point_corr=[]
        self.idx_maxima=[]
        
        fh=fits.open(fits_name)
        CDELT1=fh[0].header['CDELT1'] 
        self.px_size=abs(CDELT1)*3600 #arcsec
        restfreq=fh[0].header['RESTFRQ'] #freq of the transition studied
        CRVAL3=fh[0].header['CRVAL3'] #frequency of the 1st channel
        CDELT3=fh[0].header['CDELT3'] #freq step
        try: # depending on the version of casa used 
            self.BMIN=fh[0].header['BMIN']*3600 # [arcsec] Beam major axis length
            self.BMAJ=fh[0].header['BMAJ']*3600 # [arcsec] Beam minor axis length
        except KeyError: 
            self.BMAJ=0
            self.BMIN=0
            for item in fh[1].data:
                self.BMAJ+=item[0]
                self.BMIN+=item[1]
            self.BMAJ=self.BMAJ/len(fh[1].data) #arcsec
            self.BMIN=self.BMIN/len(fh[1].data) #arcsec
        fh.close()

        self.const=database.CONSTANTS()
        freq= CRVAL3 + CDELT3*np.arange(self.nv+self.ni) # freq of the channels
        freq=freq[self.ni:self.nf] # only the selected chan
        v_syst= -(freq[nm-self.ni]-restfreq)*self.const.c/restfreq # global speed of the system
        self.v_obs = -((freq-restfreq)/restfreq)*self.const.c-v_syst # radial velocity of the channels


        
        self.storage_pos_max=[] # initialazing storage variable for pos_maxima
        
        self.surface_storage={}
        self.saved_eraser=np.ones((self.ny,self.nx))
        self.relevant_chan=False

        
        # User interface:

        self.w=max(self.nx,550) # width of the window
        self.h=max(self.ny,550) # height of the window
        
        self.canvas=Canvas(self, width=2*550, height=550)
        self.canvas.pack()
        
        imsave('background.jpg', self.CO[self.n], cmap='afmhot' )
        self.img=PhotoImage(file='background.jpg')
        self.background=self.canvas.create_image(0,0,anchor=NW,image=self.img)
        
        xs,ys=PA_f.pos_star
        self.canvas.create_line((xs,0),(xs,self.ny),dash=(3,5),fill='white')
        self.canvas.create_line((0,ys),(self.nx,ys),dash=(3,5),fill='white')

        # User info:
        
        self.display_info()
        self.counter = Label(self)
        self.counter.place(x=int(7/6*self.w), y=int(1/4*self.h)-50)
        self.counter.configure(text=str(self.n+1+self.ni)+"/"+str(self.nv+self.ni)+ "  ||  speed:"+str(self.v_obs[self.n])+'m/s')
        self.detection = Label(self) 
        self.detection.place(x=int(7/6*self.w), y=int(1/4*self.h)+250)
        
        #Interaction:
        
        self.canvas.bind("<Button-1>", self.add_point)
        self.canvas.bind("<Button-3>", self.remove_point)
        
        self.comp_max=Button(self.canvas, text="Compute maxima",command=self.compute_maxima)
        self.comp_max.place(x=int(7/6*self.w), y=int(1/4*self.h))
        self.corr_but=Button(self.canvas, text="Correction", command=self.correction)
        self.corr_but.place(x=int(7/6*self.w), y=int(1/4*self.h)+50)
        self.next_chan_but=Button(self.canvas, text="Next channel", command=self.next_chan)
        self.next_chan_but.place(x=int(7/6*self.w), y=int(1/4*self.h)+100)
        self.reset_but=Button(self.canvas, text='RESET', command=self.reset)
        self.reset_but.place(x=int(7/6*self.w), y=int(1/4*self.h)+200)
        
    def display_info(self):
        """ User information"""
        S = Scrollbar(self)
        T = Text(self, height=6, width=50)
        S.place(x=int(7/6*self.w)+355, y=int(3/4*self.h))
        T.place(x=int(7/6*self.w), y=int(3/4*self.h))
        S.config(command=T.yview)
        T.config(yscrollcommand=S.set)
        quote = """ INFO : 
        Left click to add a point 
        Right click to remove the last point 
        Click on Compute maxima button to see the position of the maxima
        Click on Correction button to select and remove the unwanted points
        Click on Next channel button to go to the next channel"""
        T.insert(END, quote)
        T.config(state=DISABLED)
        
        
    def add_point(self,event) :
        """Adds the coordinates of the click to a list"""
        x = event.x 
        y = event.y 
        if x<=self.nx and y<=self.ny:
            r = 2 # radius of the circle
            self.detection.configure(text="detected click on (" + str(event.x) +";"+ str(event.y)+")")
            self.id_point=self.id_point+[self.canvas.create_oval(x-r, y-r, x+r, y+r, fill = 'white')]
            self.mouse.append([x,y])
    
    def remove_point(self,event):
        """Removes the last point created"""
        if len(self.id_point)==0:
                self.detection.configure(text="there is no points yet : cannot remove one !")
        else:
            self.detection.configure(text="right click detected : last point removed !")
            self.canvas.delete(self.id_point.pop())
            self.mouse.pop()
        
    def add_point_corr(self,event) :
        """Add the coordinates of the click to a correction list"""
        x = event.x 
        y = event.y 
        if x<=self.nx and y<=self.ny:
            r = 2 # radius of the circle
            self.detection.configure(text="detected click on (" + str(event.x) +";"+ str(event.y)+")")
            self.id_point_corr=self.id_point_corr+[self.canvas.create_oval(x-r, y-r, x+r, y+r, fill = 'red')]
            self.mouse.append([x,y])
            
    def reset(self):
        """ Reset all the parameters of the current channel. Does NOT change previous channels"""
        self.mouse.clear()
        self.pos_maxima.clear()
        self.detection.configure(text="reset")
        for item in self.id_point:
            self.canvas.delete(item) 
        if len(self.id_point_max)>0:
            for item in self.id_point_max:
                self.canvas.delete(item) 
        if len(self.id_point_corr)>0:
            for item in self.id_point_corr:
                self.canvas.delete(item) 
        self.id_point.clear()
        self.id_point_max.clear()
        self.id_point_corr.clear()
        self.idx_maxima.clear()
        self.saved_eraser=np.ones((self.ny,self.nx))
        self.relevant_chan=False
        self.switch=False
        self.canvas.bind("<Button-1>", self.add_point)
        self.carac_gauss.clear()

        
    def create_mask(self):
        """Returns an array with 1 inside the polygon and 0 outside.
        Input : poly = list of tuples (coordinates of each angle of the polygon)
        Output : mask = array
        """
        x, y = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        x, y = x.flatten(), y.flatten()
    
        points = np.vstack((x,y)).T
        poly=mpp.Path(self.mouse)
        mask = poly.contains_points(points)
        mask=mask.reshape(self.ny,self.nx)
        return mask
    
    def gaussian(self,y,a,y0,sigma):
        return a*np.exp((-(y-y0)**2)/(2*sigma**2))    
    
    def compute_maxima(self):
        """Compute the maxima for each row of your selected area"""
        r=2
        self.relevant_chan=True
        self.detection.configure(text="detection of maxima")
        if len(self.mouse)>0:
            self.mask=self.create_mask()
            masked_im=self.mask*self.CO[self.n]        
            masked_im=np.transpose(masked_im)
            for x in range(len(masked_im)):
                column=masked_im[x] 
                column_sigma=[]
                y=np.argmax(column)
                ymin=y-(self.BMAJ/self.px_size)/2
                ymax=y+(self.BMAJ/self.px_size)/2
                for j in range(len(column)):
                    if j<ymin or j>ymax:
                        column_sigma.append(10**15)
                    else:
                        column_sigma.append(1)

#                ymin=np.argmin(column_sigma)
#                ymax=np.argmax(column_sigma[ymin:])+ymin
                if column[y]!=0:
                    popt,pcov=curve_fit(self.gaussian,[i for i in range(len(column))],  column,p0=[1,y,1],sigma=np.array(column_sigma),absolute_sigma=False, bounds=([-np.inf,ymin,0],[np.inf,ymax,np.inf]))
                    a,y0,sigma=popt
                    self.pos_maxima.append((x,y0))
                    self.carac_gauss.append((x,a,y0,sigma,y))
                    y0=int(y0)
                    self.idx_maxima.append((x,y0))
                    self.id_point_max=self.id_point_max+[self.canvas.create_oval(x-r, y0-r, x+r, y0+r, fill = 'green')]
            self.mouse.clear()
    
    def correction(self):
        """Select the points you want to remove"""
        if self.switch:
            self.detection.configure(text="unwanted points removed ! \n click correction again to remove other points")
            eraser=self.create_mask()
            self.saved_eraser=self.saved_eraser*np.logical_not(copy(eraser))
            for i in range(len(eraser[0])):
                for j in range(len(eraser)):
                    if eraser[j][i] and (i,j) in self.idx_maxima:
                        pos_item=self.idx_maxima.index((i,j))
                        self.canvas.delete(self.id_point_max.pop(pos_item))
                        self.idx_maxima.remove((i,j))
                        for item in self.pos_maxima:
                            if item[0]==i:
                                self.pos_maxima.remove(item)
                        for item in self.carac_gauss:
                            if item[0]==i:
                                self.carac_gauss.remove(item)
            for k in range(len(self.id_point_corr)):
                self.canvas.delete(self.id_point_corr.pop())
            self.mouse.clear()
        else:
            self.detection.configure(text="select the points you want to remove \n click correction again to remove them")
            self.mouse.clear()
            self.canvas.bind("<Button-1>", self.add_point_corr)
            self.switch=True
            
    def next_chan(self):
        """Go to the next channel"""
        if self.n<self.nv-1:
            self.storage_pos_max.append(copy(self.pos_maxima)) # /!\ copy otherwise storage_pos_max will be modified when pos_maxima is modified
            self.storage_carac_gauss.append(copy(self.carac_gauss))  
            
            if self.relevant_chan:
                self.storage_points(self.n,self.mask,self.saved_eraser)

            self.n+=1
            self.reset()
            self.detection.configure(text="")
            self.counter.configure(text=str(self.n+1+self.ni)+"/"+str(self.nv+self.ni)+ "  ||  speed:"+str(self.v_obs[self.n])+'m/s')
            imsave('background.jpg', self.CO[self.n], cmap='afmhot' )
            self.img=PhotoImage(file='background.jpg')
            self.canvas.itemconfig(self.background,image=self.img)
        else:
            self.storage_pos_max.append(copy(self.pos_maxima)) # /!\ copy otherwise storage_pos_max will be modified when pos_maxima is modified
            self.storage_carac_gauss.append(copy(self.carac_gauss))
            self.detection.configure(text="job done")
            self.destroy()
            
    def storage_points(self,n,mask,eraser):
        temp={}
        temp['mask']=mask
        temp['eraser']=eraser
        self.surface_storage[n+self.ni]=temp
        
        
class maxima_finder_noUI():
    def __init__(self,data_reused, CO): 
        self.data_reused=data_reused
        self.ni=self.data_reused.ni
        self.nf=self.data_reused.nf
        [(x0, y0),(x1,y1)]=self.data_reused.window
        self.CO=CO
        self.PA=data_reused.PA
        np.nan_to_num(self.CO,copy=False) # replaces the nan values by 0
        self.CO=rotate(self.CO,180-self.PA,axes=(2,1), reshape=False)
        self.CO=self.CO[self.ni:self.nf,y0:y1,x0:x1]
        self.nx=len(self.CO[0][0]) # number of columns
        self.ny=len(self.CO[0]) # number of lines
        self.nv=len(self.CO) #number of chanels


        self.storage_carac_gauss=[]
        self.storage_pos_max=[] 

        
        
        select=self.data_reused.selection

        for i in range(self.nv):
            self.pos_maxima=[]
            self.carac_gauss=[]
            if i+self.ni in select.keys():
                mask=select[i+self.ni]['mask']
                eraser=select[i+self.ni]['eraser']
#                if len(np.shape(eraser))!=2:
#                    eraser=np.prod(eraser,axis=0)
                self.compute_maxima(mask,eraser,i)
            self.storage_carac_gauss.append(self.carac_gauss)
            self.storage_pos_max.append(self.pos_maxima)
            

    def compute_maxima(self,mask,eraser,n):
        """Compute the maxima for each row of your selected area"""
        masked_im=mask*self.CO[n] 
        masked_im=np.transpose(masked_im)
        for x in range(len(masked_im)):
            column=masked_im[x] 
            column_sigma=[]
            for item in column:
                if item==0:
                    column_sigma.append(10**15)
                else:
                    column_sigma.append(1)
            y=np.argmax(column)
            ymin=np.argmin(column_sigma)
            ymax=np.argmax(column_sigma[ymin:])+ymin
            if column[y]!=0: 
                try:
                    popt,pcov=curve_fit(self.gaussian,[i for i in range(len(column))],  column,p0=[1,y,1],sigma=np.array(column_sigma),absolute_sigma=False, bounds=([-np.inf,ymin,0],[np.inf,ymax,np.inf]))
                    a,y0,sigma=popt
                    if eraser[int(y0),x]:
                        self.pos_maxima.append((x,y0))
                        self.carac_gauss.append((x,a,y0,sigma,y))
                    y0=int(y0)
                except RuntimeError:
                    pass


    def gaussian(self,y,a,y0,sigma):
        return a*np.exp((-(y-y0)**2)/(2*sigma**2))    

class storage():
    """Contains all the relevant data about the object"""
    def __init__(self, CO, PA, xs, ys, pos_maxima, carac_gauss, obj, ni, nm, nf, window):
        """Contains: CO, pos_maxima, star_center, obj, PA, ni, nm (0 velocity index), nf, window"""
        self.CO=CO
        self.pos_maxima=pos_maxima
        self.carac_gauss=carac_gauss
        self.star_center=(xs,ys)
        self.obj=obj
        self.PA=PA
        self.ni=ni
        self.nm=nm
        self.nf=nf
        self.window=window

class storage_plus():
    """Contains all the relevant data about the object"""
    def __init__(self, CO, PA, xs, ys, pos_maxima, carac_gauss, obj, ni, nm, nf, window, selection):
        """Contains: CO, pos_maxima, star_center, obj, PA, ni, nm (0 velocity index), nf, window"""
        self.CO=CO
        self.pos_maxima=pos_maxima
        self.carac_gauss=carac_gauss
        self.star_center=(xs,ys)
        self.obj=obj
        self.PA=PA
        self.ni=ni
        self.nm=nm
        self.nf=nf
        self.window=window
        self.selection=selection
# =============================================================================
# Main script
# =============================================================================


                
if __name__ == "__main__":
    obj="test_12CO" #studied object                                              ######### /!\ TO BE MODIFIED FOR EACH OBJECT
    UI=True # enables or disables UI                                                   ######### /!\ TO BE MODIFIED FOR EACH OBJECT
    data=database.DATA(obj)
    path=data.PATH
    continuum_fits= data.CONT
    fits_name = data.FITS
    
    ext = ["_sup_back", "_sup_front","_inf_back","_inf_front"]
    
    if UI:
        PA_f = PA_finder(None, fits_name, continuum_fits)   
        PA_f.title(" Selection of the parameters" )
        PA_f.mainloop()
        ni=PA_f.ni
        nm=PA_f.nm
        nf=PA_f.nf
        CO=PA_f.CO     
        window=[(PA_f.x0, PA_f.y0),(PA_f.x1,PA_f.y1)] # coordinates of the resized window (Top Left and Bottom Right)
        xs, ys=PA_f.pos_star
        for e in ext:
            Max_f = maxima_finder(None, CO, ni,nm, nf)
            Max_f.title("Maxima of emission for "+e[1:]+" surface")
            Max_f.mainloop()    
            data=storage(CO,PA_f.PA,xs,ys,Max_f.storage_pos_max, Max_f.storage_carac_gauss,obj, ni, nm, nf, window)
            data_plus=storage_plus(CO,PA_f.PA,xs,ys,Max_f.storage_pos_max, Max_f.storage_carac_gauss,obj, ni, nm, nf, window, Max_f.surface_storage)
            file_name=fits_name+e+'.co_surf'
            file_name_plus=fits_name+e+'.extended_info.co_surf'
            with open(file_name, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(file_name_plus, 'wb') as handle:
                pickle.dump(data_plus, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
    else:
        obj_sim= "HD97048_HR_13CO" # image used to get the surfaces with UI            ######### /!\ TO BE MODIFIED FOR EACH OBJECT
        data_sim=database.DATA(obj_sim)
        fits_name_sim=data_sim.FITS
        for e in ext:
            with open(fits_name_sim+e+".extended_info.co_surf", 'rb') as handle: 
                data_reused=pickle.load(handle)
                fh=fits.open(fits_name)
                CO=fh[0].data 
                CO=CO[0]
                fh.close()
                PA=data_reused.PA
                xs,ys=data_reused.star_center
                ni=data_reused.ni
                nm=data_reused.nm
                nf=data_reused.nf
                window=data_reused.window
                Max_f = maxima_finder_noUI(data_reused,CO)
                CO=Max_f.CO
                data=storage(CO,PA,xs,ys,Max_f.storage_pos_max, Max_f.storage_carac_gauss,obj, ni, nm, nf, window)
                file_name=fits_name+e+'.co_surf'
                with open(file_name, 'wb') as handle:
                    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

