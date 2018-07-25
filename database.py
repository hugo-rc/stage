from astropy import constants as const
import numpy as np

class DATA():
    """ Charateristic of the object
    
    UNITS: 
        DIST [pc]
        INC [deg]
        PA [deg]
    
    """
    def __init__(self, name):
        self.name=name
        if name=='HD163296':
            self.HD163296()
        if name=='HD163296_HR_CO':
            self.HD163296_HR_CO()
        if name=='HD163296_HR_13CO':
            self.HD163296_HR_13CO()
        if name=='HD163296_HR_C18O':
            self.HD163296_HR_C18O()
        if name=='HD97048_13CO':
            self.HD97048_13CO()
        if name=='HD97048_HR_13CO':
            self.HD97048_HR_13CO()
        if name=='HD97048_HR_13CO_contsub':
            self.HD97048_HR_13CO_contsub()
        if name=='HD97048_12CO':
            self.HD97048_12CO()
        if name=='AS209':
            self.AS209()
        if name=='IMLupi_CO_contsub':
            self.IMLupi_CO_contsub()
        if name=='IMLupi_CO_nocontsub':
            self.IMLupi_CO_nocontsub()
        if name=='IMLupi_13CO_nocontsub':
            self.IMLupi_13CO_nocontsub()
        if name=="HD100453":
            self.HD100453()
        if name=='HD142527_13CO':
            self.HD142527_13CO()
        if name=='HD34282':
            self.HD34282()
        if name=='HD135344B':
            self.HD135344B()
    def HD163296(self):
        self.PATH="/home/hugo/Documents/Stage/selection_objets/HD163296/Itziar/"     
        self.FITS = self.PATH+"HD163296_CO3-2.fits.gz"  
        self.CONT = self.PATH+"HD163296_continuum.fits"
        self.DIST=101.5
        self.INC=47.6
    def HD163296_HR_CO(self):
        self.PATH="/home/hugo/Documents/Stage/selection_objets/HD163296/HR/"     
        self.FITS = self.PATH+"HD163296_CO_100m.s-1.image.fits.gz"  
        self.CONT = self.PATH+"HD163296_calibrated_final_cont_2spw_ap.image.fits.gz"
        self.DIST=101.5
        self.INC=47.6#47.6
        
    def HD163296_HR_13CO(self):
        self.PATH="/home/hugo/Documents/Stage/selection_objets/HD163296/HR/"     
        self.FITS = self.PATH+"HD163296_13CO_100m.s-1.image.fits.gz"  
        self.CONT = self.PATH+"HD163296_calibrated_final_cont_2spw_ap.image.fits.gz"
        self.DIST=101.5
        self.INC=47.6
        
    def HD163296_HR_C18O(self):
        self.PATH="/home/hugo/Documents/Stage/selection_objets/HD163296/product/"     
        self.FITS = self.PATH+"HD163296_C18O.pbcor.fits"  
        self.CONT = self.PATH+"calibrated_final_cont.pbcor.fits"
        self.DIST=101.5
        self.INC=47.6

        
    def HD97048_12CO(self):
        self.PATH="/home/hugo/Documents/Stage/selection_objets/HD97048/"     
        self.FITS = self.PATH+"HD_97048_12CO_21_uniform_image.image.fits"  
        #self.CONT=self.PATH+"HD97048_band7_continuum_selfcal_1024_0.07arcsec_0.13mJy_briggs.image.fits"
        self.CONT=self.PATH+"HD97048_b6_continuum_selfcal_uniform.image.fits"
        self.DIST=184.8
        self.INC=41
        
    def HD97048_13CO(self):
        self.PATH="/home/hugo/Documents/Stage/selection_objets/HD97048/"     
        self.FITS = self.PATH+"HD_97048_13CO_21_uniform_image.image.fits"  
        #self.CONT=self.PATH+"HD97048_band7_continuum_selfcal_1024_0.07arcsec_0.13mJy_briggs.image.fits"
        self.CONT=self.PATH+"HD97048_b6_continuum_selfcal_uniform.image.fits"
        self.DIST=184.8
        self.INC=41

    def HD97048_HR_13CO(self):
        self.PATH="/home/hugo/Documents/Stage/selection_objets/HD97048/HR/"     
        self.FITS = self.PATH+"HD_97048_13CO32_briggs_selfcal_nocontsub.fits"  
        #self.CONT=self.PATH+"HD97048_band7_continuum_selfcal_1024_0.07arcsec_0.13mJy_briggs.image.fits"
        self.CONT=self.PATH+"HD97048_b7_selfcal_automask_briggs.fits"
        self.DIST=184.8
        self.INC=41

    def HD97048_HR_13CO_contsub(self):
        self.PATH="/home/hugo/Documents/Stage/selection_objets/HD97048/HR/"     
        self.FITS = self.PATH+"HD_97048_13CO32_briggs_selfcal_contsub.fits"  
        #self.CONT=self.PATH+"HD97048_band7_continuum_selfcal_1024_0.07arcsec_0.13mJy_briggs.image.fits"
        self.CONT=self.PATH+"HD97048_b7_selfcal_automask_briggs.fits"
        self.DIST=184.8
        self.INC=41

    def AS209(self):
        self.PATH="/home/hugo/Documents/Stage/selection_objets/AS209/"     
        self.FITS = self.PATH+"AS209.CO.contsub.image.pbcor.fits"  
        self.CONT=self.PATH+"AS209.cont.image.pbcor.fits"
        self.DIST=120.9
        self.INC=35
    def IMLupi_CO_nocontsub(self):
        self.PATH="/home/hugo/Documents/Stage/selection_objets/IMLupi/"     
        self.FITS = self.PATH+"my_12CO_no_contsub.pbcor.fits"  
        self.CONT=self.PATH+"IMLupi_continuum.image.fits"
        self.DIST=158.4
        self.INC=48
    def IMLupi_CO_contsub(self):
        self.PATH="/home/hugo/Documents/Stage/selection_objets/IMLupi/"     
        self.FITS = self.PATH+"my_12CO_contsub.pbcor.fits"  
        self.CONT=self.PATH+"IMLupi_continuum.image.fits"
        self.DIST=158.4
        self.INC=48
    def IMLupi_13CO_nocontsub(self):
        self.PATH="/home/hugo/Documents/Stage/selection_objets/IMLupi/"     
        self.FITS = self.PATH+"my_13CO_no_contsub.pbcor.fits"  
        self.CONT=self.PATH+"IMLupi_continuum.image.fits"
        self.DIST=158.4
        self.INC=48
    def HD100453(self):
        self.PATH="/home/hugo/Documents/Stage/selection_objets/HD100453/"
        self.FITS=self.PATH+"HD_100453_12CO_21_briggs_image_nocontsub_.image.pbcor.fits.gz"
        self.CONT=self.PATH+"HD100453_cavity_ring_ring_data_superuniform.image.fits"
        self.DIST=104.2
        self.INC=34
        
    def HD142527_13CO(self):
        self.PATH="/home/hugo/Documents/Stage/selection_objets/HD142527/product_2012/"
        self.FITS=self.PATH+"HD142527_13CO32_image.pbcor.fits"
        self.CONT=self.PATH+"calibrated_final_cont_image.pbcor.fits"
        self.DIST=157.3
        self.INC=27
    def HD34282(self):
        self.PATH="/home/hugo/Documents/Stage/selection_objets/HD34282/"
        self.FITS=self.PATH+"HD_34282_12CO_21_briggs_image.image.fits"
        self.CONT=self.PATH+"HD34282_b6_continuum_selfcal_briggs.image.fits"
        self.DIST=311.6 #160 or 400
        self.INC=56
    def HD135344B(self):
        self.PATH="/home/hugo/Documents/Stage/selection_objets/HD135344B/product/"
        self.FITS=self.PATH+"calibrated.13co.ms.image.13co.source3.image.pbcor.fits"
        self.CONT=self.PATH+"calibrated.cont.ms.image.continuum.source3.image.pbcor.fits"
        self.DIST=135.7
        self.INC=62
        

class CONSTANTS():
    def __init__(self):
        self.G=const.G.value
        self.M_sun=const.M_sun.value
        self.c=const.c.value
        self.au=const.au.value
        self.hp=const.h.value
        self.kB=const.k_B.value
        self.arcsec=4.848136811095e-06
        self.pc2au=648000/np.pi #parsec to au
        self.deg2rad=np.pi/180 
        
