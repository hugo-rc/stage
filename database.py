class DATA():
    """ Charateristic of the object
    
    UNITS: 
        DIST [pc]
        INC [deg]
    
    """
    def __init__(self, name):
        self.name=name
        if name=='HD163296':
            self.HD163296()
        if name=='HD97048':
            self.HD97048()
        if name=='AS209':
            self.AS209()
    def HD163296(self):
        self.PATH="/home/hugo/Documents/Stage/selection_objets/HD163296/Itziar/"     
        self.FITS = self.PATH+"HD163296_CO3-2.fits.gz"  
        self.CONT = self.PATH+"HD163296_continuum.fits"
        self.DIST=101.5
        self.INC=47.7
    def HD97048(self):
        self.PATH="/home/hugo/Documents/Stage/selection_objets/HD97048/"     
        self.FITS = self.PATH+"HD_97048_13CO_21_uniform_image.image.fits"  
        #self.CONT=self.PATH+"HD97048_band7_continuum_selfcal_1024_0.07arcsec_0.13mJy_briggs.image.fits"
        self.CONT=self.PATH+"HD97048_b6_continuum_selfcal_uniform.image.fits"
        self.DIST=158
        self.INC=41
    def AS209(self):
        self.PATH="/home/hugo/Documents/Stage/selection_objets/AS209/"     
        self.FITS = self.PATH+"AS209.C18O.contsub.image.pbcor.fits"  
        self.CONT=self.PATH+"AS209.cont.image.pbcor.fits"


