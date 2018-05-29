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
        self.n=0
        
        # User interface:
        
        self.w=max(self.nx,550) # width of the window
        self.h=max(self.ny,550) # height of the window
        
        self.canvas=Canvas(self, width=2*self.w, height=self.h)
        self.canvas.pack()
        
        
        imsave('background.jpg', self.CO[self.n])
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
            imsave('background.jpg', self.CO[self.n])
            self.img=PhotoImage(file='background.jpg')
            self.canvas.itemconfig(self.background,image=self.img)      
        else:
            self.log.configure(text="last channel")
            
    def prev_chan(self):
        """Go to the previous channel"""
        if self.n>0:
            self.n-=1
            self.counter.configure(text=str(self.n+1)+"/"+str(self.nv))
            imsave('background.jpg', self.CO[self.n])
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
        self.nf=self.n
            
    def rotate_img(self):
        """ Rotates the image by angle PA"""
        try:
            self.PA=int(self.PA_entry.get())
            self.log.configure(text="Preview")   
            self.CO_rot=rotate(self.CO[self.n],self.PA, reshape=False)
            imsave('background.jpg', self.CO_rot)
            self.img=PhotoImage(file='background.jpg')
            self.canvas.itemconfig(self.background,image=self.img)
        except ValueError:
            self.log.configure(text="ERROR: Please enter an float or integer")
           
    
    def validate(self):
        """ Validates the choice of PA"""
        self.log.configure(text="PA saved. Clic on Crop to select the area to study")
        self.CO=rotate(self.CO,self.PA,axes=(2,1), reshape=False)
        self.cont_img=rotate(self.cont_img,self.PA, reshape=False)
        # Removing the previous buttons and creating a new one to crop the image
        self.validate_but.destroy()
        self.rotate_but.destroy()
        self.crop_but=Button(self.canvas, text='Crop', command=self.crop)
        self.crop_but.place(x=int(7/6*self.w), y=int(1/4*self.h)+150)
        self.switch=False
        
        
    def crop(self):
        """ Crops the image to keep the interesting part"""
        if self.switch: # Second clic: crop
            (self.x0,self.x1)=(min(self.x0,self.x1),max(self.x0,self.x1))
            (self.y0,self.y1)=(min(self.y0,self.y1),max(self.y0,self.y1))
            self.CO=self.CO[:,self.y0:self.y1,self.x0:self.x1]
            self.cont_img=self.cont_img[self.y0:self.y1,self.x0:self.x1]
            self.crop_but.destroy()
            self.cancel_but.destroy()
            self.canvas.delete(self.id_rect)
            imsave('background.jpg', self.CO[self.n])
            self.img=PhotoImage(file='background.jpg')
            self.canvas.itemconfig(self.background,image=self.img)
            self.finish_but=Button(self.canvas, text='Done !', command=self.finish)
            self.finish_but.place(x=int(7/6*self.w), y=int(1/4*self.h)+150)
        else: # First clic: select area
            self.log.configure(text="Clic and drag to select a rectangle")
            self.cancel_but=Button(self.canvas, text='Cancel', command=self.cancel)
            self.cancel_but.place(x=int(7/6*self.w+200), y=int(1/4*self.h)+150)
            self.canvas.bind("<Button-1>", self.rect_start)
            self.canvas.bind("<ButtonRelease-1>", self.rect_end)
            self.switch=True

    def rect_start(self,event):
        self.x0=event.x
        self.y0=event.y
        r=2
        self.log.configure(text="Drag to select a rectangle")
        self.id_start=self.canvas.create_oval(self.x0-r, self.y0-r, self.x0+r, self.y0+r, fill = 'white')
        
    def rect_end(self,event):
        self.x1=event.x
        self.y1=event.y
        self.log.configure(text="Area selected: clic on Crop to validate the selection")
        self.id_rect=self.canvas.create_rectangle(self.x0,self.y0,self.x1,self.y1,outline='white')
        self.canvas.delete(self.id_start)

        
    def cancel(self):
        self.log.configure(text="Clic and drag to select a rectangle")
        self.canvas.delete(self.id_rect)
        
    def finish(self):
        """Quit the window"""
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
            self.pos_star=np.unravel_index(np.argmax(self.cont_img, axis=None), self.cont_img.shape) # position of the max intensity in the continuum image == position of the star
            self.destroy()
            
class maxima_finder(Tk):
    def __init__(self,parent,CO):
        Tk.__init__(self,parent) # don't know exactly what it does, 
        self.parent = parent     # but it was recommended to put this        
        self.CO=CO
        
        self.nx=len(self.CO[0][0]) # number of columns
        self.ny=len(self.CO[0]) # number of lines
        self.nv=len(self.CO) #number of chanels
        self.n=0 # initialazing image counter
        self.mouse=[]
        self.id_point=[]
        self.id_point_max=[]
        self.pos_maxima=[]
        self.switch=False
        self.id_point_corr=[]
        
        self.storage_pos_max=[] # initialazing storage variable for pos_maxima
        
        # User interface:

        self.w=max(self.nx,550) # width of the window
        self.h=max(self.ny,550) # height of the window
        
        self.canvas=Canvas(self, width=2*550, height=550)
        self.canvas.pack()
        
        imsave('background.jpg', self.CO[self.n])
        self.img=PhotoImage(file='background.jpg')
        self.background=self.canvas.create_image(0,0,anchor=NW,image=self.img)
        
        # User info:
        
        self.display_info()
        self.counter = Label(self)
        self.counter.place(x=int(7/6*self.w), y=int(1/4*self.h)-50)
        self.counter.configure(text=str(self.n+1)+"/"+str(self.nv))
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
        self.switch=False
        self.canvas.bind("<Button-1>", self.add_point)

        
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
    
    def compute_maxima(self):
        """Compute the maxima for each row of your selected area"""
        r=2
        self.detection.configure(text="detection of maxima")
        if len(self.mouse)>0:
            mask=self.create_mask()
            masked_im=mask*self.CO[self.n]        
            masked_im=np.transpose(masked_im)
            for x in range(len(masked_im)):
                column=masked_im[x] 
                y=np.argmax(column)
                if column[y]>0:
                    self.pos_maxima.append((x,y))
                    self.id_point_max=self.id_point_max+[self.canvas.create_oval(x-r, y-r, x+r, y+r, fill = 'green')]
            self.mouse.clear()
    
    def correction(self):
        """Select the points you want to remove"""
        if self.switch:
            self.detection.configure(text="unwanted points removed ! \n click correction again to remove other points")
    
            eraser=self.create_mask()
            for i in range(len(eraser[0])):
                for j in range(len(eraser)):
                    if eraser[j][i] and (i,j) in self.pos_maxima:
                        pos_item=self.pos_maxima.index((i,j))
                        self.pos_maxima.remove((i,j))
                        self.canvas.delete(self.id_point_max.pop(pos_item))
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
            self.storage_pos_max.append(self.pos_maxima)
            self.n+=1
            self.reset()
            self.detection.configure(text="")
            self.counter.configure(text=str(self.n+1)+"/"+str(self.nv))
            imsave('background.jpg', self.CO[self.n])
            self.img=PhotoImage(file='background.jpg')
            self.canvas.itemconfig(self.background,image=self.img)
        else:
            self.detection.configure(text="job done")
            self.destroy()
            
class storage():
    """Contains all the relevant data about the object"""
    def __init__(self, CO, PA, xs, ys, pos_maxima, obj, nm):
        self.CO=CO
        #self.cont_img=cont_img
        self.pos_maxima=pos_maxima
        self.star_center=(xs,ys)
        self.obj=obj
        self.PA=PA
        self.help="Contains: CO, pos_maxima, star_center, obj, PA, nm (0 velocity index)"
        
# =============================================================================
# Main script
# =============================================================================


                
if __name__ == "__main__":
    path="/home/hugo/Documents/Stage/selection_objets/"
    obj="HD163296" #studied object
    continuum_fits=path+obj+"/Itziar/HD163296_continuum.fits"
    fits_name = path+"HD163296/Itziar/HD163296_CO3-2.fits.gz" 
    
    ext = ["_sup_back","_sup_front","_inf_back" ,"_inf_front"]

    
    PA_f = PA_finder(None, fits_name, continuum_fits)   
    PA_f.title(" Selection of the parameters" )
    PA_f.mainloop()
    ni=PA_f.ni
    nm=PA_f.nm
    nf=PA_f.nf
    CO=PA_f.CO     
    CO=CO[ni:nf] # Removing the noisy channels
    xs, ys=PA_f.pos_star
    for e in ext:
        Max_f = maxima_finder(None, CO)
        Max_f.title("Maxima of emission for "+e+" surface")
        Max_f.mainloop()    
        data=storage(CO,PA_f.PA,xs,ys,Max_f.storage_pos_max,obj, nm)
        file_name=fits_name+e+'.co_surf'
        with open(file_name, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    

