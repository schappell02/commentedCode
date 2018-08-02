import scipy
import pyfits
from scipy import stats
from scipy import special
from scipy import integrate
from gcwork import starTables
import nmpfit_sy
import asciidata, os, sys, pickle
import nmpfit_sy#2 as nmpfit_sy
import numpy as np
import math
import pdb
import time
import scipy.optimize as opter
from scipy.optimize import fsolve
from scipy.optimize import minimize
from matplotlib.ticker import ScalarFormatter 
import datetime
import time
import threading
from astropy.stats import LombScargle
import pylab as py
import mysql.connector
import scYoung

homeRoot = '/u/schappell/'
plotRoot = homeRoot+'plots/'
tableRoot = homeRoot+'tables/'
cRoot = homeRoot+'code/c/'
mnoutRoot = homeRoot+'pmnOld/'

pi = math.pi

#constants and dimensional analysis
mass = 3.960e6 #mass of Sgr A*, in solar masses, according to my align
masse = 0.164e6
dist = 7828.0 #distance to Sgr A*, in pc, according to my align
G = 6.6726e-8
msun = 1.99e33
GM = G * mass * msun
GMe = G * masse * msun
mass_g = mass * msun
masse_g = masse * msun
sec_in_yr = 3.1557e7
cm_in_au = 1.496e13
cm_in_pc = 3.086e18
km_in_pc = 3.086e13
au_in_pc = 206265.0
asy_to_kms = dist * cm_in_au / (1e5 * sec_in_yr)
as_to_km = dist * cm_in_au / (1e5)
density0 = 3.2e4
density0_g = 3.2e4 * msun
density0e = 1.3e4
density0e_g = density0e * msun
GM_as_yr = GM * sec_in_yr**2 * 1e-15 / as_to_km**3



'''one_star - class of object for one star in databse
    
    Required input:
    -sname = name of star in str format
    -align = home directory of align
    -points = points directory within align home directory
    
    Optional input:
    -ak_correct = if set to True, star's magnitude will be Ak corrected, if False, this will not happen,
                  set to True unless otherwise changed
    
    Attributes:
    - name = sname, name of star
    - align = align home directory
    - points = points subdirectory
    - years = time of observations, in units of years, in array
    - x = observed x/RA positions of star, in array
    - y = observed y/Dec positions of star, in array
    - xe/ye = errors in positions of star, in arrays
    - epoch = number of observations for star, integer number
    - fit = current polynomial fit for star, every star stars with just 'Point', can be updated with
            set_vel, set_acc, and set_jerk class functions
    - mag = magnitude in K' filter, of ak_correct is set to True, ak correction is applied, if False
            no ak correction
    - pOld = probability of being late-type, from 0 to 1, confirmed early-type stars are set to 0.0,
             confirmed late-type stars are set to 1.0
    - pYng = probability of being early-type, from 0 to 1, confirmed late-type stars are set to 0.0,
             confirmed early-type stars are set to 1.0
    
    
    Functions:
    - set_vz() = get line-of-sight velocities for star. No input needed. Returns to database and if
                 any viable radial velocity measurements are recorded (ie both the radial velocity
                 and it's error are non-zero) will be attributed to star in array form
                 
                - Added Attributes:
                    + vz_date = array of radial velocity dates, in years
                    + vz = array of radial velocities, in km/s
                    + vz_err = array of error in radial velocity measurements, in km/s
    
    - set_gcows(gcows_status) = sets attribute in_gcows, or whether star is in GCOWS footprint
    
                - Required Input:
                    + gcows_status = 0/1, whether star is in GCOWS footprint
                                    
                - Added Attributes:
                    + in_gcows = either 0 for not or 1 for in GCOWS footprint
                                    
    - set_vel(xt0, yt0, x0, y0, x0e, y0e, xv, yv, xve, yve, xchi2, ychi2) = sets attributes for velocity
                fit for star in both x/RA and y/Dec directions, attributes are only set if star is measured
                in 3 or more epochs, attributes 'fit' set to 'Vel' to designate that star is currently
                best fit with velocity fit
                
                -Required Input: **Units are not specified, but need to be consistent between all inputs)
                    + x/yt0 = t0 or time which is set to 0 for velocity fit in x/RA and y/Dec direction
                    + x/y0 = positions in x/RA and y/Dec at t0 or time is 0 for velocity fit
                    + x/y0e = error in x/y0
                    + x/yv = velocity term for velocity fit in x/RA and y/Dec directions
                    + x/yve = error in x/yv
                    + x/ychi2 = chi squared for x/RA and y/Dec velocity fit
                    
                -Added Attributes:
                    + v_xt0, v_yt0, v_x0, v_y0, v_x0e, v_y0e, v_xv, v_yv, v_xve, v_yve, v_xchi2, v_ychi2
                        = set to corresponding terms from input, 'v_' denotes from the velocity fit
                    + xt0, yt0, x0, y0, x0e, y0e, xv, yv, xve, yve, xchi2, ychi2 = set to corresponding
                        terms from input
                    + v_R and R = projected position, calculated from x0 and y0
                    + v_chix/yr and chix/yr = reduced chi^2 in x and y directions for velocity fit
                    
    - set_acc(self, xt0, yt0, x0, y0, x0e, y0e, xv, yv, xve, yve, xa, ya, xae, yae, xchi2, ychi2) = 
                sets attributes for acceleration fit for star in both x/RA and y/Dec directions, attributes
                are only set if star is measured in 4 or more epochs, calculates projected and tangential
                accelerations (ar, at) and respective errors (radial and tangential in relation to Sgr A*),
                F Test run to compare chi^2 between velocity and acceleration fits (whole function does not 
                run if star does not have velocity attributes, so run set_vel before this function), if passes 
                F Test, star's values for its orbit are set to those for the accel fit (denoted by a_*) and 
                fit attribute set to 'Acc'
                
                -Required Input: **Units are not specified, but need to be consistent between all inputs)
                    + x/yt0 = t0 or time which is set to 0 for accel fit in x/RA and y/Dec direction
                    + x/y0 = positions in x/RA and y/Dec at t0 or time is 0 for accel fit
                    + x/y0e = error in x/y0
                    + x/yv = velocity term for accel fit in x/RA and y/Dec directions
                    + x/yve = error in x/yv
                    + x/ya = acceleration term for accel fit in x/RA and y/Dec directions
                    + x/yae = error in x/ya
                    + x/ychi2 = chi squared for x/RA and y/Dec accel fit
                
                -Optional Input:
                    + pval = P value used for F Test comparison between velocity and acceleration fit, unless
                        otherwise stated, is set to 4.0
                        
                -Added Attributes:
                    + a_xt0, a_yt0, a_x0, a_y0, a_x0e, a_y0e, a_xv, a_yv, a_xve, a_yve, a_xa, a_ya, a_xae, a_yae,
                        a_xchi2, a_ychi2 = set to corresponding terms from input, 'a_' denotes from the accel fit
                    + a_R = projected position, calculated from x0 and y0 for accel fit
                    + a_chix/yr = reduced chi^2 in x and y directions for accel fit
                    + va_x/yFval = F ratio for x/RA and y/Dec between velocity and acceleration fits
                    + va_x/yFprob = F distribution value corresponding to F ratio, to compare to corresponding P value

    - set_jerk(self, xt0, yt0, x0, y0, x0e, y0e, xv, yv, xve, yve, xa, ya, xae, yae, xj, yj, xje, yje, xchi2, ychi2) =
            sets attributes for jerk (derivative of acceleration) fit for star in both x/RA and y/Dec directions, 
            attributes are only set if star is measured in 5 or more epochs, calculates projected and tangential
            accelerations (ar, at) and respective errors (radial and tangential in relation to Sgr A*), F Test run to 
            compare chi^2 between acceleration and jerk fits, only when star passed F Test between vel and accel 
            (whole function does not run if star does not have velocity and accel attributes, so run set_vel and 
            set_acc before this function), if passes F Test, star's values for its orbit are set to those for the jerk 
            fit (denoted by j_*) and fit attribute set to 'Jerk'

            -Required Input: **Units are not specified, but need to be consistent between all inputs)
                + x/yt0 = t0 or time which is set to 0 for accel fit in x/RA and y/Dec direction
                + x/y0 = positions in x/RA and y/Dec at t0 or time is 0 for jerk fit
                + x/y0e = error in x/y0
                + x/yv = velocity term for jerk fit in x/RA and y/Dec directions
                + x/yve = error in x/yv
                + x/ya = acceleration term for jerk fit in x/RA and y/Dec directions
                + x/yae = error in x/ya
                + x/yj = jerk term for jerk fit in x/RA and y/Dec directions
                + x/yje = errors in x/yj
                + x/ychi2 = chi squared for x/RA and y/Dec accel fit

            -Optional Input:
                + pval = P value used for F Test comparison between acceleration and jerk fit, unless
                    otherwise stated, is set to 4.0

            -Added Attributes:
                + j_xt0, j_yt0, j_x0, j_y0, j_x0e, j_y0e, j_xv, j_yv, j_xve, j_yve, j_xa, j_ya, j_xae, j_yae,
                    j_xy, j_yj, j_xje, j_yje, j_xchi2, j_ychi2 = set to corresponding terms from input, j_' denotes 
                    from the jerk fit
                + j_R = projected position, calculated from x0 and y0 for jerk fit
                + j_chix/yr = reduced chi^2 in x and y directions for jerk fit
                + aj_x/yFval = F ratio for x/RA and y/Dec between acceleration and jerk fits
                + aj_x/yFprob = F distribution value corresponding to F ratio, to compare to corresponding P value

'''
class one_star():
    def __init__(self,sname,align,points,ak_correct=True):
        self.name = sname
        self.align = align
        self.points = points

        pointsFile = np.loadtxt(align+points+sname+'.points')
        self.years = pointsFile[:,0]
        self.x = pointsFile[:,1]
        self.y = pointsFile[:,2]
        #self.path_length = np.array([math.sqrt((self.x[i]-self.x[i+1])**2 +(self.y[i]-self.y[i+1])**2) for i in range(len(self.x)-1)])
        #self.totpl = np.sum(self.path_length)
        self.xe = pointsFile[:,3]
        self.ye = pointsFile[:,4]
        self.epoch = len(pointsFile[:,0])
        self.fit = 'Point'

#search database for this star, save info about stars designated as late/early type and magnitude
        database = mysql.connector.connect(host="galaxy1.astro.ucla.edu",user="dbread",passwd="t36fCEtw",db="gcg")
        cur = database.cursor()
        cur.execute("SELECT young, old, kp, Ak_sch FROM stars WHERE name='{0}'".format(sname))
        for row in cur:
            if (ak_correct==True):
                self.mag = row[2] + (2.7 - row[3])
            else:
                self.mag = row[2]
            if (row[0] == 'T'):
                self.pOld = 0.0
                self.pYng = 1.0
            elif (row[1] == 'T'):
                self.pOld = 1.0
                self.pYng = 0.0

#search different table for prob of being late/early type from Do et al 2013
        cur.execute("SELECT probYngSimPrior, probOldSimPrior FROM unknownSims WHERE name='{0}'".format(sname))
        for row in cur:
            self.pYng = row[0]
            self.pOld = row[1]

#hard code certain stars as certain spectral types
        if ((sname == 'S0-38') | (sname == 'S0-49') | (sname == 'S0-35') | (sname == 'S1-32')):
            self.pOld = 1.0
            self.pYng = 0.0
        if (sname == 'S0-61'):
            self.pOld = 0.0
            self.pYng = 1.0


#get radial velocity information from another table in database
    def get_vz(self):
        database = mysql.connector.connect(host="galaxy1.astro.ucla.edu",user="dbread",passwd="t36fCEtw",db="gcg")
        cur = database.cursor()
        cur.execute("SELECT ddate, vlsr, vz_err FROM spectra WHERE name='{0}'".format(self.name))
        tmp_date = np.array([])
        tmp_vz = np.array([])
        tmp_vzerr = np.array([])
        for row in cur:
            if ((row[1] !=0) & (row[2] != 0) & (row[2] != None)):
                tmp_date = np.append(tmp_date,row[0])
                tmp_vz = np.append(tmp_vz, row[1])
                tmp_vzerr = np.append(tmp_vzerr, row[2])

        self.vz_date = tmp_date
        self.vz = tmp_vz
        self.vz_err = tmp_vzerr



    def set_gcows(self,gcows_status):
        self.in_gcows = gcows_status



    def set_vel(self, xt0, yt0, x0, y0, x0e, y0e, xv, yv, xve, yve, xchi2, ychi2):
        if (self.epoch > 2):
#Set values for velocity fit
            self.fit = 'Vel'
            self.v_xt0 = xt0
            self.v_yt0 = yt0
            self.v_x0 = x0
            self.v_y0 = y0
            self.v_R = np.hypot(x0, y0)
            self.v_x0e = x0e
            self.v_y0e = y0e
            self.v_xv = xv
            self.v_yv = yv
            self.v_xve = xve
            self.v_yve = yve
            self.v_xchi2 = xchi2
            self.v_ychi2 = ychi2
            self.v_xchi2r = self.v_xchi2 / (self.epoch - 2.0)
            self.v_ychi2r = self.v_ychi2 / (self.epoch - 2.0)
            vmod_xdiff = self.v_xv*(self.years[0] - self.v_xt0) - self.v_xv*(self.years[-1] - self.v_xt0)
            vmod_ydiff = self.v_yv*(self.years[0] - self.v_yt0) - self.v_yv*(self.years[-1] - self.v_yt0)
            self.v_modelPL = math.sqrt(vmod_xdiff**2 + vmod_ydiff**2)
            self.v_xres = self.x - (self.v_x0 + self.v_xv*(self.years - self.v_xt0))
            self.v_yres = self.y - (self.v_y0 + self.v_yv*(self.years - self.v_yt0))
            self.v_xres_sig = self.v_xres / self.xe
            self.v_yres_sig = self.v_yres / self.ye

#For now, as no other kinematic fits tested, set the star's overall kinematic constants
#to that from velocity fit
            self.xt0 = xt0
            self.yt0 = yt0
            self.x0 = x0
            self.y0 = y0
            self.R = np.hypot(x0, y0)
            self.x0e = x0e
            self.y0e = y0e
            self.xv = xv
            self.yv = yv
            self.xve = xve
            self.yve = yve
            self.xchi2 = xchi2
            self.ychi2 = ychi2
            self.xchi2r = self.xchi2 / (self.epoch - 2.0)
            self.ychi2r = self.ychi2 / (self.epoch - 2.0)
            self.modelPL = math.sqrt(vmod_xdiff**2 + vmod_ydiff**2)
            self.xres = self.x - (self.v_x0 + self.v_xv*(self.years - self.v_xt0))
            self.yres = self.y - (self.v_y0 + self.v_yv*(self.years - self.v_yt0))
            self.xres_sig = self.v_xres / self.xe
            self.yres_sig = self.v_yres / self.ye

        else:
            print self.name+' has less than 3 epochs, no velocity fit'



    def set_acc(self, xt0, yt0, x0, y0, x0e, y0e, xv, yv, xve, yve, xa, ya, xae, yae, xchi2, ychi2,pval=4.0):
        
        if ((self.epoch > 3) & hasattr(self, 'v_xv')):
#set values from acceleration fit
            self.a_xt0 = xt0
            self.a_yt0 = yt0
            self.a_x0 = x0
            self.a_y0 = y0
            self.a_R = np.hypot(x0, y0)
            self.a_x0e = x0e
            self.a_y0e = y0e
            self.a_xv = xv
            self.a_yv = yv
            self.a_xve = xve
            self.a_yve = yve
            self.a_xa = xa
            self.a_ya = ya
            self.a_xae = xae
            self.a_yae = yae

#Calculate radial and tangential acceleration components in the plane of the sky
            ar = ((xa*x0) + (ya*y0)) / self.a_R
            at = ((xa*y0) - (ya*x0)) / self.a_R
            are =  (xae*x0/self.a_R)**2 + (yae*y0/self.a_R)**2
            are += (y0*x0e*at/self.a_R**2)**2 + (x0*y0e*at/self.a_R**2)**2
            are =  np.sqrt(are)
            ate =  (xae*y0/self.a_R)**2 + (yae*x0/self.a_R)**2
            ate += (y0*x0e*ar/self.a_R**2)**2 + (x0*y0e*ar/self.a_R**2)**2
            ate =  np.sqrt(ate)
            
            self.a_ar = ar
            self.a_at = at
            self.a_are = are
            self.a_ate = ate
            
            self.a_xchi2 = xchi2
            self.a_ychi2 = ychi2
            self.a_xchi2r = self.a_xchi2 / (self.epoch - 3.0)
            self.a_ychi2r = self.a_ychi2 / (self.epoch - 3.0)
            amod_xdiff = self.a_xv*(self.years[0] - self.a_xt0) - self.a_xv*(self.years[-1] - self.a_xt0) + 0.5*self.a_xa*(self.years[0] - self.a_xt0)**2 - 0.5*self.a_xa*(self.years[-1] - self.a_xt0)**2
            amod_ydiff = self.a_yv*(self.years[0] - self.a_yt0) - self.a_yv*(self.years[-1] - self.a_yt0) + 0.5*self.a_ya*(self.years[0] - self.a_yt0)**2 - 0.5*self.a_ya*(self.years[-1] - self.a_yt0)**2
            self.a_modelPL = math.sqrt(amod_xdiff**2 + amod_ydiff**2)
            self.a_xres = self.x - (self.a_x0 + self.a_xv*(self.years - self.a_xt0) + 0.5*self.a_xa*(self.years - self.a_xt0)**2)
            self.a_yres = self.y - (self.a_y0 + self.a_yv*(self.years - self.a_yt0) + 0.5*self.a_ya*(self.years - self.a_yt0)**2)
            self.a_xres_sig = self.a_xres / self.xe
            self.a_yres_sig = self.a_yres / self.ye

#Calculating value corresponding to input P Value
            signif = scipy.special.erfc(pval/math.sqrt(2.0))
#F ratio
            self.va_xFval = (self.v_xchi2 - self.a_xchi2) / (self.a_xchi2/(self.epoch - 3.0))
            self.va_yFval = (self.v_ychi2 - self.a_ychi2) / (self.a_ychi2/(self.epoch - 3.0))
#Corresponding values on F distribution
            self.va_xFprob = stats.f.sf(self.va_xFval, 1, (self.epoch - 3.0))
            self.va_yFprob = stats.f.sf(self.va_yFval, 1, (self.epoch - 3.0))

#If passes F Test, set the overall kinematic values of th star to that from the accel fit; force F test for S0-16 and S0-17
            if ((self.va_xFprob < signif) | (self.va_yFprob < signif) | (self.name == 'S0-16') | (self.name == 'S0-17')):
                self.fit ='Acc'
                self.xt0 = xt0
                self.yt0 = yt0
                self.x0 = x0
                self.y0 = y0
                self.R = np.hypot(x0, y0)
                self.x0e = x0e
                self.y0e = y0e
                self.xv = xv
                self.yv = yv
                self.xve = xve
                self.yve = yve
                self.xa = xa
                self.ya = ya
                self.xae = xae
                self.yae = yae
                
                self.ar = ar
                self.at = at
                self.are = are
                self.ate = ate
                
                self.xchi2 = xchi2
                self.ychi2 = ychi2
                self.xchi2r = self.xchi2 / (self.epoch - 3.0)
                self.ychi2r = self.ychi2 / (self.epoch - 3.0)
                self.modelPL = math.sqrt(amod_xdiff**2 + amod_ydiff**2)
                self.xres = self.x - (self.a_x0 + self.a_xv*(self.years - self.a_xt0) + 0.5*self.a_xa*(self.years - self.a_xt0)**2)
                self.yres = self.y - (self.a_y0 + self.a_yv*(self.years - self.a_yt0) + 0.5*self.a_ya*(self.years - self.a_yt0)**2)
                self.xres_sig = self.a_xres / self.xe
                self.yres_sig = self.a_yres / self.ye

        elif ( hasattr(self, 'v_xv')==False):
            print 'Need to run set_vel for '+self.name
        else:
            print self.name+' has less than 4 epochs, no accel fit'


    def set_jerk(self, xt0, yt0, x0, y0, x0e, y0e, xv, yv, xve, yve, xa, ya, xae, yae, xj, yj, xje, yje, xchi2, ychi2,pval=4.0):
            
        if ((self.epoch > 4) & hasattr(self, 'a_xa')):
#Set values from jerk fit
            self.j_xt0 = xt0
            self.j_yt0 = yt0
            self.j_x0 = x0
            self.j_y0 = y0
            self.j_R = np.hypot(x0, y0)
            self.j_x0e = x0e
            self.j_y0e = y0e
            self.j_xv = xv
            self.j_yv = yv
            self.j_xve = xve
            self.j_yve = yve
            self.j_xa = xa
            self.j_ya = ya
            self.j_xae = xae
            self.j_yae = yae
            
#Calculate radial and tangential accelerations in the plane of the sky and errors
            ar = ((xa*x0) + (ya*y0)) / self.a_R
            at = ((xa*y0) - (ya*x0)) / self.a_R
            are =  (xae*x0/self.a_R)**2 + (yae*y0/self.a_R)**2
            are += (y0*x0e*at/self.a_R**2)**2 + (x0*y0e*at/self.a_R**2)**2
            are =  np.sqrt(are)
            ate =  (xae*y0/self.a_R)**2 + (yae*x0/self.a_R)**2
            ate += (y0*x0e*ar/self.a_R**2)**2 + (x0*y0e*ar/self.a_R**2)**2
            ate =  np.sqrt(ate)
            
            self.j_ar = ar
            self.j_at = at
            self.j_are = are
            self.j_ate = ate
            
            self.j_xj = xj
            self.j_yj = yj
            self.j_xje = xje
            self.j_yje = yje
            self.j_xchi2 = xchi2
            self.j_ychi2 = ychi2
            self.j_xchi2r = self.j_xchi2 / (self.epoch - 4.0)
            self.j_ychi2r = self.j_ychi2 / (self.epoch - 4.0)
            jmod_xdiff = self.j_xv*(self.years[0] - self.j_xt0) - self.j_xv*(self.years[-1] - self.j_xt0) + 0.5*self.j_xa*(self.years[0] - self.j_xt0)**2 - 0.5*self.j_xa*(self.years[-1] - self.j_xt0)**2 + self.j_xj*(self.years[0] - self.j_xt0)**3/6.0 - self.j_xj*(self.years[-1] - self.j_xt0)**3/6.0
            jmod_ydiff = self.j_yv*(self.years[0] - self.j_yt0) - self.j_yv*(self.years[-1] - self.j_yt0) + 0.5*self.j_ya*(self.years[0] - self.j_yt0)**2 - 0.5*self.j_ya*(self.years[-1] - self.j_yt0)**2 + self.j_yj*(self.years[0] - self.j_yt0)**3/6.0 - self.j_yj*(self.years[-1] - self.j_yt0)**3/6.0
            self.j_modelPL = math.sqrt(jmod_xdiff**2 + jmod_ydiff**2)
            self.j_xres = self.x - (self.j_x0 + self.j_xv*(self.years - self.j_xt0) + 0.5*self.j_xa*(self.years - self.j_xt0)**2 + self.j_xj*(self.years - self.j_xt0)**3/6.0)
            self.j_yres = self.y - (self.j_y0 + self.j_yv*(self.years - self.j_yt0) + 0.5*self.j_ya*(self.years - self.j_yt0)**2 + self.j_yj*(self.years - self.j_yt0)**3/6.0)
            self.j_xres_sig = self.j_xres / self.xe
            self.j_yres_sig = self.j_yres / self.ye

#Run F test between accel and jerk, only if passes F Test between velocity and accel
            if (self.fit == 'Acc'):
                signif = scipy.special.erfc(pval/math.sqrt(2.0))
#Calculate corresponding value for input P value, F ratios in x and y directions, and the corresponding value on the F distribution
                self.aj_xFval = (self.a_xchi2 - self.j_xchi2) / (self.j_xchi2/(self.epoch - 4.0))
                self.aj_yFval = (self.a_ychi2 - self.j_ychi2) / (self.j_ychi2/(self.epoch - 4.0))
                self.aj_xFprob = stats.f.sf(self.aj_xFval, 1, (self.epoch - 4.0))
                self.aj_yFprob = stats.f.sf(self.aj_yFval, 1, (self.epoch - 4.0))

#If star passes F Test, set overall kinematic values for star to that from jerk fit; force F test for S0-16 and S0-17
                if ((self.aj_xFprob < signif) | (self.aj_yFprob < signif) | (self.name == 'S0-16') | (self.name == 'S0-17')):
                    self.fit ='Jerk'
                    self.xt0 = xt0
                    self.yt0 = yt0
                    self.x0 = x0
                    self.y0 = y0
                    self.R = np.hypot(x0, y0)
                    self.x0e = x0e
                    self.y0e = y0e
                    self.xv = xv
                    self.yv = yv
                    self.xve = xve
                    self.yve = yve
                    self.xa = xa
                    self.ya = ya
                    self.xae = xae
                    self.yae = yae

                    self.ar = ar
                    self.at = at
                    self.are = are
                    self.ate = ate
                    
                    self.xj = xj
                    self.yj = yj
                    self.xje = xje
                    self.yje = yje
                    self.xchi2 = xchi2
                    self.ychi2 = ychi2
                    self.xchi2r = self.xchi2 / (self.epoch - 4.0)
                    self.ychi2r = self.ychi2 / (self.epoch - 4.0)
                    self.modelPL = math.sqrt(jmod_xdiff**2 + jmod_ydiff**2)
                    self.xres = self.x - (self.j_x0 + self.j_xv*(self.years - self.j_xt0) + 0.5*self.j_xa*(self.years - self.j_xt0)**2 + self.j_xj*(self.years - self.j_xt0)**3/6.0)
                    self.yres = self.y - (self.j_y0 + self.j_yv*(self.years - self.j_yt0) + 0.5*self.j_ya*(self.years - self.j_yt0)**2 + self.j_yj*(self.years - self.j_yt0)**3/6.0)
                    self.xres_sig = self.j_xres / self.xe
                    self.yres_sig = self.j_yres / self.ye

        elif ( hasattr(self, 'a_xa')==False):
            print 'Need to run set_accel for '+self.name
        else:
            print self.name+' has less than 5 epochs, no jerk fit'


    
        
        
'''stars - class of object for set of stars in database, goes through velocity and acceleration fits
    for each star (and jerk if parameter polyj is set) and sets those values for each star by using
    one_star class
    
    Required input:
    - root = path of root directory
    - poly = directory of polyfit results in root directory
    
    
    Optional input:
    - pval = corresponding P Value, set to 4 unless otherwise set
    - polyj = directory of polyfit for jerk fit, if not stated, jerk values not set for star sample
    
    Attributes:
    - root = root directory set by root parameter
    - poly = polyfit subdirectory
    - stars = array of stars of class one_star, with attributes for velocity, acceleration, and potentially
            jerk fit applied
    
    
    Functions:
    - sample_Like(in_GCfield=True,chainsDir='efit/chains_lessrv/',addErr=True,file='/u/schappell/Downloads/NIRC2 radial mask/nirc2_gcows_2010_all_mask.fits',
                    savefile='/u/schappell/code/c/gcows_field.dat',magCut=15.5,Rcut=5.0,epochs=3,sigmaCut = 3.0,chiCut=10.0,label='') = function for determining
                    sample of stars used in primary sample for Bayesian analysis, cuts to determine sample made in K' magnitude, number of epochs detected, projected
                    distance, significance of unphysical acceleration (accelerations not in the direction towards Sgr A*), non-zero probability of being late-type, 
                    and reduced chi^2 from best kinematic fit, outputs table in .tex form with stars and their info listed (hard coded path as 
                    /u/schappell/tables/mn_sample_table.tex), and outputs .dat file with R, acceleration in the radial direction, acceleration error, and probability 
                    of being late type for this sample in cgs units
    
                -Optional Input:
                    + in_GCfield = select out stars which are not in designated field, defined by file parameter, set to True unless otherwise set
                    + chainsDir = directory for chains from likelihood fit to S0-2/S0-38's orbit and Sgr A* parameters such as position and velocity, will be used
                            only if addErr is set to True
                    + addErr = whether error in position is updated with uncertainty in position and velocity of Sgr A* from S0-2 and/or S0-38 orbital fit and
                            if error will be updated with empirically additive error to account for uncertainty from source confusion by stars below detection limit,
                            set to True unless otherwise stated
                    + file = file path for mask, fits file, of GCOWS footprint in form of pixels
                    + savefile = file path of resulting output file with list of x and y positions, in pixels, which are in GCOWS field
                    + magCut = limit in K' magnitude of stars considered for sample, set to 15.5 unless otherwise stated
                    + Rcut = cut in projected distance, in arcsec, 5.0 unless otherwise stated
                    + epochs = cut in number of epochs detected, 3 unless otherwise stated
                    + sigmaCut = cut in sigma used to cut out stars with significant unphysical accelerations, or accelerations not in direction towards Sgr A*, set to
                            3.0 unless otherwise stated
                    + chiCut = cut to reduced chi^2 in x and y direction, applied to best kinematic fit (velocity, acceleration, or jerk), set to 10.0 unless otherwise
                            stated
                    + label = string value of flag added to outputted .dat files with sample information
                    
    - schodel_samp(label='',innerCut=5.0,outerCut=15.0,magCut=17.75,lmagCut=0.0) = determines sample from Schodel et al 2010 work, stored in database and outputs
                    a .dat file with projected positions listed (in cm) and probability of being late-type all set to 1 (assumed to be late-type), cuts in outer
                    and inner projected distance used, as well as magnitude cuts
                    
                -Optional Input:
                    + label = label for .dat file, string value, blank unless otherwise stated
                    + innerCut = inner radius cut, in arcsec, set to 5 unless changed
                    + outerCut = outer radius cut, in arcsec, set to 15 unless otherwise stated
                    + magCut = higher magnitude cut in K', 17.75 unless changed
                    + lmagCut = lower magnitude cut in K', 0.0 unless changed
                    
    - run_MN(,label_in='',label_out='',innerCut=5.0,Rcut=5.0,outerCut=15.0,mnAlpha=6.0,mnDelta=4.0,mnBreak=0.5,max_r=5.0,situation=3,nonRadial=1) = compiles C++ code for
                    likelihood analysis using MultiNest and runs it with inputted star samples
                    
                -Optional Input:
                    + label_in = string value, label of GCOWS and Schodel samples (.dat files) to be used for likelihood analysis, should be same label for both
                    + label_out = label of resulting files from MultiNest, such as the posterior
                    + innerCut = inner cut, in projected radius, arcsec, for Schodel sample, should be equal to or greater than Rcut to avoid stars being double counted
                    + Rcut = outer radius cut for GCOWS sample
                    + outerCut = outer radius cut for Schodel sample, make sure value is consistent with values used for Schodel sample construction
                    + mnAlpha/Delta/Break = values for set Alpha, Delta, and Break radius for broken power law fit in likelihood analysis, if situation is set to 1 or 2
                            then these values are set and only gamma is left as free parameter, break radius is in pc
                    + max_r = maximum radius, in pc, integrated to in likelihood analysis, set to 5 pc unless otherwise stated
                    + situation = value between 1 and 4, even numbers only use GCOWS sample, no Schodel, odd values use both GCOWS and Schodel sample, situations 1 and 2
                            only leave gamma as free parameter, Alpha, Delta, and break radius are set to input values, while all 4 parameters are left free for situations 
                            3 and 4, situation set to 3 or use both GCOWS and Schodel samples and leave all 4 parameters free in broken power law
                    + nonRadial = whether it is required that GCOWS sample be in GCOWS footprint, if set to 1, this is required, as well as sample be within projected radius 
                            cut and footprint is used in likelihood analysis if set to 0 only requirement is being within projected radius
                            
    - in_gcows(file='/u/schappell/Downloads/NIRC2 radial mask/nirc2_gcows_2010_all_mask.fits',savefile='/u/schappell/code/c/gcows_field.dat') = cycles through stars in
                    sample and determines which are in GCOWS footprint by turning their arcsec positions into pixels in mask, sets set_gcows values for stars
                    
                -Optional Input:
                    + file = path for mask used
                    + savefile = path for file to be outputted with list of pixels that are within mask
                    
    - updateErr_all(chainsDir=None,addErr=False) = controls updating error for both the uncertainty in position and velocity of Sgr A* from star orbital fit and from
                    empirically found additive error to account for source confusion caused by stars below detection limit, each update in error can only be done once,
                    attributes are checked to make sure not run twice or more, plot showing additive error as a function of magnitude is created, as well as comparison
                    between distribution of tangential acceleration before and after additive error is applied to sample
                    
                -Optional Input:
                    + chainsDir = directory which contains chains with proper motion fit and errors to Sgr A*, if left as None, this error will not be added in quadrature
                            to stellar errors in position
                    + addErr = determines if empirical additive error is added in quadrature to acceleration uncertainties
                    
    - all_vz = cycles through stars and runs get_vz function in one_star class to load radial velocity information for all stars which have it
    
    - plot_V_vs_R(inner=1.0) = plots combined velocity vs projected radius for early and late-type stars, only consider stars outside of given inner radius (inner), stars with
                    radial velocity measurements and with reduced chi^2 less than 10 in both X and Y directions. For every star, total velocity is calculated from proper motion
                    in the plane of the sky and the RV measurement with the smallest error. Plots show stars binned in phase space and scatter plots separated by early and late type stars.
    
                -Optional Input:
                    + inner = inner projected radius cut (in arcsec) only stars outside of this radius are considered and plotted
                    
    -plot_chi2r = creates and sabes various plots for chi^2 (reduced) diagnostics, including histograms of reduced chi^2 in X and Y directions, chi^2_reduced vs projected R, 
                    vs magnitude (K'), vs number of epochs, histograms of chi^2_reduded with the limits of K' < 15.5, R < 5arcsec, and non-zero probability of being late type,
                    and plot of average reduced chi^2 vs epoch number
                    
    
    '''
class stars():
    def __init__(self,root,poly,points,pointsj=None,polyj=None,pval=4.0):
        self.root = root
        self.poly = poly

        #call in velocity info
        vel_fit = asciidata.open(self.root + self.poly + 'fit.linearFormal')
        vel_t0 = asciidata.open(self.root + self.poly + 'fit.lt0')
        v_xt0 = vel_t0[1].tonumpy()
        v_yt0 = vel_t0[2].tonumpy()
        v_names = vel_fit[0].tonumpy()
        v_x0 = vel_fit[1].tonumpy()
        v_xv = vel_fit[2].tonumpy()
        v_x0e = vel_fit[3].tonumpy()
        v_xve = vel_fit[4].tonumpy()
        v_xchi2 = vel_fit[5].tonumpy()
        v_y0 = vel_fit[7].tonumpy()
        v_yv = vel_fit[8].tonumpy()
        v_y0e = vel_fit[9].tonumpy()
        v_yve = vel_fit[10].tonumpy()
        v_ychi2 = vel_fit[11].tonumpy()

        #call in acceleration info
        acc_fit = asciidata.open(self.root + self.poly + 'fit.accelFormal')
        acc_t0 = asciidata.open(self.root + self.poly + 'fit.t0')
        a_xt0 = acc_t0[1].tonumpy()
        a_yt0 = acc_t0[2].tonumpy()
        a_names = acc_fit[0].tonumpy()
        a_x0 = acc_fit[1].tonumpy()
        a_xv = acc_fit[2].tonumpy()
        a_xa = acc_fit[3].tonumpy()
        a_x0e = acc_fit[4].tonumpy()
        a_xve = acc_fit[5].tonumpy()
        a_xae = acc_fit[6].tonumpy()
        a_xchi2 = acc_fit[7].tonumpy()
        a_y0 = acc_fit[9].tonumpy()
        a_yv = acc_fit[10].tonumpy()
        a_ya = acc_fit[11].tonumpy()
        a_y0e = acc_fit[12].tonumpy()
        a_yve = acc_fit[13].tonumpy()
        a_yae = acc_fit[14].tonumpy()
        a_ychi2 = acc_fit[15].tonumpy()
        
        #call in jerk fit if polyfit for jerk directory given
        if (polyj!=None):
            jerk_fit = asciidata.open(self.root + polyj + 'fit.accelFormal')
            jerk_t0 = asciidata.open(self.root + polyj + 'fit.t0')
            j_xt0 = jerk_t0[1].tonumpy()
            j_yt0 = jerk_t0[2].tonumpy()
            j_names = jerk_fit[0].tonumpy()
            j_x0 = jerk_fit[1].tonumpy()
            j_xv = jerk_fit[2].tonumpy()
            j_xa= jerk_fit[3].tonumpy()
            j_xj = jerk_fit[4].tonumpy()
            j_x0e = jerk_fit[5].tonumpy()
            j_xve = jerk_fit[6].tonumpy()
            j_xae = jerk_fit[7].tonumpy()
            j_xje = jerk_fit[8].tonumpy()
            j_xchi2 = jerk_fit[9].tonumpy()
            j_y0 = jerk_fit[11].tonumpy()
            j_yv = jerk_fit[12].tonumpy()
            j_ya = jerk_fit[13].tonumpy()
            j_yj = jerk_fit[14].tonumpy()
            j_y0e = jerk_fit[15].tonumpy()
            j_yve = jerk_fit[16].tonumpy()
            j_yae = jerk_fit[17].tonumpy()
            j_yje = jerk_fit[18].tonumpy()
            j_ychi2 = jerk_fit[19].tonumpy()

        self.stars = np.array([])
        
        #cycle through stars in database
        for i in range(len(v_x0)):
            tmpname = str(v_names[i])
            tmpStar = one_star(tmpname,self.root,points)
            #set velocity terms
            tmpStar.set_vel(v_xt0[i], v_yt0[i], v_x0[i], v_y0[i], v_x0e[i], v_y0e[i], v_xv[i], v_yv[i], v_xve[i], v_yve[i], v_xchi2[i], v_ychi2[i])
            for j in range(len(a_x0)):
                accname = str(a_names[j])
                if (tmpname ==  accname):
                    #if has acceleration term, set it using one_star class
                    tmpStar.set_acc(a_xt0[j], a_yt0[j], a_x0[j], a_y0[j], a_x0e[j], a_y0e[j], a_xv[j], a_yv[j], a_xve[j], a_yve[j], a_xa[j], a_ya[j], a_xae[j], a_yae[j], a_xchi2[j], a_ychi2[j],pval=pval)
            if (polyj!=None):
                for k in range(len(j_x0)):
                    jerkname = str(j_names[k])
                    if (tmpname == jerkname):
                        #if star has jerk terms, set it using one_star class
                        tmpStar.set_jerk(j_xt0[k], j_yt0[k], j_x0[k], j_y0[k], j_x0e[k], j_y0e[k], j_xv[k], j_yv[k], j_xve[k], j_yve[k], j_xa[k], j_ya[k], j_xae[k], j_yae[k], j_xj[k], j_yj[k], j_xje[k], j_yje[k], j_xchi2[k], j_ychi2[k],pval=pval)

            self.stars = np.append(self.stars, tmpStar)





    def sample_Like(self,in_GCfield=True, chainsDir='efit/chains_lessrv/', addErr=True,file='/u/schappell/Downloads/NIRC2 radial mask/nirc2_gcows_2010_all_mask.fits',savefile='/u/schappell/code/c/gcows_field.dat',magCut=15.5, lmagCut=0.0, Rcut=5.0, accelRcut=2.5,epochs=3,sigmaCut = 3.0, chiCut = 7.0,label=''):
        #if param set, find which stars are in GCOWS footprint, and update errors
        if (in_GCfield == True):
            self.in_gcows(file=file, savefile=savefile)
        self.updateErr_all(chainsDir=chainsDir,addErr=addErr)
        
        samp_name = np.array([])
        samp_epoch = np.array([])
        samp_mag = np.array([])
        samp_pOld = np.array([])
        samp_R = np.array([])
        samp_x = np.array([])
        samp_xe = np.array([])
        samp_y = np.array([])
        samp_ye = np.array([])
        samp_xv = np.array([])
        samp_yv = np.array([])
        samp_xve = np.array([])
        samp_yve = np.array([])
        samp_fit = np.array([])
        samp_ar = np.array([])
        samp_are = np.array([])
        samp_at = np.array([])
        samp_ate = np.array([])
        samp_xchi2r = np.array([])
        samp_ychi2r = np.array([])
        samp_xchi2r_a = np.array([])
        samp_ychi2r_a = np.array([])
        samp_modelPL = np.array([])
        samp_xres = np.array([])
        samp_yres = np.array([])
        samp_xres_quad = np.array([])
        samp_yres_quad = np.array([])
        samp_t0 = np.array([])
        
        #begin tex file for output table
        out = open(tableRoot+'mn_accel_sample_table.tex','w')
        #out.write('\\documentclass{aastex} \n')
        out.write('\\setlength{\\tabcolsep}{4pt} \n')
        #out.write('\\usepackage{graphicx,longtable,pdflscape,threeparttablex} \n')
        #out.write('\\usepackage[labelsep=space]{caption} \n')
        #out.write('\\begin{document} \n')
        out.write('\\scriptsize \n')
        #out.write('\\begin{landscape} \n')
        out.write('\\begin{ThreePartTable} \n')
        out.write('\\begin{TableNotes} \n')
        out.write('\\item [a] Corrected for differential extinction to a mean extinction of 2.7 magnitudes \n')
        out.write('\\item [b] Number of epochs \n')
        #out.write('\\item [b] Only reported for stars with trusted measured accelerations, see section \\ref{sec:sample} \n')
        out.write('\\item [c] Probability of being late-type, reported in \\citet{do13l} \n')
        #out.write('\\item [d] Best kinematic fit, determined by F Tests \n')
        out.write('\\item [d] Average $\chi^2_r$ between $X$ and $Y$ direction for the best kinematic fit \n')
        out.write("\\item [e] Radial acceleration at a star's given projected radius, assuming line-of-sight distance is zero, divided by the star's error in radial acceleration \n")
        out.write('\\end{TableNotes} \n')
        out.write('\\begin{longtable}{*{14}{c}} \n')
        out.write("\\caption{Stars in Inner Sample}\\label{tab:like_accel_samp_data} \n")
        #out.write('\\hline \n')
        out.write('\\hline \n')
        #out.write("Star & K' & R$_{2D}$ & n\\tnote{a} & T$_0$ & X & Y & $v_x$ & $v_y$ & $a_r$ & $a_t$ & $j_x$ & $j_y$ & $p_{old}$\\tnote{b} & Fit\\tnote{c} & $\chi^2_r$\\tnote{d} & $a_r(R,z=0)/\sigma_R}$\\tnote{e}  \\\\ \n")
        #out.write("&    & (arcsec) &   & (yrs) & \\multicolumn{3}{c}{(arcsec)} & \\multicolumn{2}{c}{(mas/yr)} & \\multicolumn{2}{c}{($\mu$as/yr$^2$)} & \\multicolumn{2}{c}{($\mu$as/yr$^3$)} &   &   &    \\\\\ \n")
        out.write("Star & K'\\tnote{a} & n\\tnote{b} & X & Y & R$_{2D}$ & t_0 & $v_x$ & $v_y$ & $a_r$ & $a_t$ & $p_{old}$\\tnote{c} & $\chi^2_r$\\tnote{d} & $a_{r,min}/\sigma_r$\\tnote{e} \n")
        out.write(" & &  &   & \\multicolumn{3}{c}{(arcsec)} & (years) & \\multicolumn{2}{c}{($\mu$as/yr)} & \\multicolumn{2}{c}{($\mu$as/yr$^2$)} &   &   &   &    \n")
        #out.write('\\\\ \n')
        out.write('\\hline \n')
        out.write('\\midrule\\endhead \n')
        out.write('\\bottomrule\\endfoot \n')

#format for output data to table
#fmt_wj = '%15s  %1s  %5.1f  %1s  %5.1f  %1s  %5.1f  %1s  %5.1f  %1s  %2d  %1s  %6.2f  %5s  %6.2f  %1s  %6.2f  %5s  %6.2f  %1s  %6.2f  %5s  %6.2f  %1s  %6.2f  %5s  %6.2f  %1s  %6.1f  %5s  %6.1f  %1s  %6.1f  %5s  %6.1f  %1s  %6.2f  %1s  %5s  %1s  %6.2f  %4s\n'
#      fmt_noj = '%15s  %1s  %5.1f  %1s  %5.1f  %1s  %5.1f  %1s  %5.1f  %1s  %2d  %1s  %6.2f  %5s  %6.2f  %1s  %6.2f  %5s  %6.2f  %1s  %6.2f  %5s  %6.2f  %1s  %6.2f  %5s  %6.2f  %1s  %5s  %1s  %5s  %1s  %6.2f  %1s  %5s  %1s  %6.2f  %4s\n'
        fmt_wa = '%15s  %1s  %5.1f  %1s  %2d  %1s  %6.2f  %5s  %6.2f  %1s  %6.2f  %5s  %6.2f  %1s  %6.2f  %1s  %6.2f  %1s  %6.1f  %5s  %6d  %1s  %6.1f  %5s  %6d  %1s  %6.1f  %5s  %6d  %1s  %6.1f  %5s  %6d  %1s  %6.2f  %1s %6.1f  %1s  %6.2f  %4s\n'
        fmt_noa =  '%15s  %1s  %5.1f  %1s  %2d  %1s  %6.1f  %5s  %6.1f  %1s  %6.1f  %5s  %6.1f  %1s  %6.1f  %1s  %6.2f  %1s  %6.1f  %5s  %6d  %1s  %6.1f  %5s  %6d  %1s  %5s  %1s  %5s  %1s  %6.2f  %1s %6.1f  %1s  %5s  %4s\n'
            
        for tmpStar in self.stars:
            if (hasattr(tmpStar,'a_ar') & hasattr(tmpStar,'pOld')):
                if ((tmpStar.mag < magCut) & (tmpStar.mag >= lmagCut) & (tmpStar.R < Rcut) & (tmpStar.epoch > epochs) & (tmpStar.pOld > 0.0)):
                    #cycle through stars in inner sample
                    if (hasattr(tmpStar,'ar')):
                        tmp_ar = tmpStar.ar
                        tmp_are = tmpStar.are
                        tmp_at = tmpStar.at
                        tmp_ate = tmpStar.ate
                    else:
                        tmp_ar = tmpStar.a_ar
                        tmp_are = tmpStar.a_are
                        tmp_at = tmpStar.a_at
                        tmp_ate = tmpStar.a_ate
                    if (in_GCfield==True):
                        if (tmpStar.in_gcows==1):
                            
                            #for 5sig cut, be careful to only do this once and not to run it multiple times
                            #sig5dex = np.where((tmpStar.xres_sig >= 5.0) | (tmpStar.yres_sig >= 5.0))[0]
                            #if len(sig5dex) >= 1:
                            #    use5dex = np.where((np.abs(tmpStar.xres_sig) < 5.0) & (np.abs(tmpStar.yres_sig) < 5.0))[0]
                            #    np.savetxt('/g/ghez/align/schappell_14_06_18/points_5sig/'+str(tmpStar.name)+'.points',np.transpose([tmpStar.years[use5dex],tmpStar.x[use5dex],tmpStar.y[use5dex],tmpStar.xe[use5dex],tmpStar.ye[use5dex]]),delimiter=' ')
                            
                            
                            samp_name = np.append(samp_name,tmpStar.name)
                            samp_epoch = np.append(samp_epoch,tmpStar.epoch)
                            samp_mag = np.append(samp_mag,tmpStar.mag)
                            samp_pOld = np.append(samp_pOld,tmpStar.pOld)
                            samp_R = np.append(samp_R,tmpStar.R)
                            if (str(tmpStar.fit) == 'Vel'):
                                samp_fit = np.append(samp_fit,1.0)
                            elif (str(tmpStar.fit) == 'Acc'):
                                samp_fit = np.append(samp_fit,2.0)
                            elif (str(tmpStar.fit) == 'Jerk'):
                                samp_fit = np.append(samp_fit,3.0)
                            #add to array of eventual output radial acceleration, error, etc
                            samp_ar = np.append(samp_ar,tmp_ar)
                            samp_are = np.append(samp_are,tmp_are)
                            samp_at = np.append(samp_at,tmp_at)
                            samp_ate = np.append(samp_ate,tmp_ate)
                            samp_xchi2r = np.append(samp_xchi2r,tmpStar.xchi2r)
                            samp_ychi2r = np.append(samp_ychi2r,tmpStar.ychi2r)
                            samp_xv = np.append(samp_xv,tmpStar.xv)
                            samp_yv = np.append(samp_yv,tmpStar.yv)
                            samp_xe = np.append(samp_xe,tmpStar.x0e)
                            samp_ye = np.append(samp_ye,tmpStar.y0e)
                            samp_xve = np.append(samp_xve,tmpStar.xve)
                            samp_yve = np.append(samp_yve,tmpStar.yve)
                            samp_x = np.append(samp_x,tmpStar.x0)
                            samp_y = np.append(samp_y,tmpStar.y0)
                            samp_t0 = np.append(samp_t0,tmpStar.xt0)
                            samp_modelPL = np.append(samp_modelPL,tmpStar.modelPL)
                            samp_xres = np.append(samp_xres, tmpStar.xres_sig)
                            samp_yres = np.append(samp_yres, tmpStar.yres_sig)
                            samp_xres_quad = np.append(samp_xres_quad, math.sqrt(np.sum(tmpStar.xres_sig**2)))
                            samp_yres_quad = np.append(samp_yres_quad, math.sqrt(np.sum(tmpStar.yres_sig**2)))





                            #write to tex file row of desired info for each star

        samp_arz0 = GM_as_yr / samp_R**2
        tmpsort = np.argsort(samp_R)#-samp_arz0/samp_are)
        
        for i in tmpsort:
            if ((samp_epoch[i] > 27) & (samp_name[i] != 'S3-327') & (samp_name[i] != 'S5-53') & (samp_name[i] != 'S6-68') & (samp_name[i] != 'S6-61') & (samp_name[i] != 'S6-74') & (samp_name[i] != 'S6-53') & (samp_name[i] != 'S5-69') & (samp_name[i] != 'S3-136') & (samp_name[i] != 'S3-279') & (samp_name[i] != 'S3-125') & (samp_name[i] != 'S3-200') & (samp_name[i] != 'S3-15') & (samp_name[i] != 'S2-117') & (samp_name[i] != 'S1-45')):
                out.write(fmt_wa % (samp_name[i],'&',samp_mag[i],'&',samp_epoch[i],'&',samp_x[i],'$\pm$',samp_xe[i],'&',samp_y[i],'$\pm$',samp_ye[i],'&',samp_R[i],'&',samp_t0[i],'&',samp_xv[i]*1e6,'$\pm$',samp_xve[i]*1e6,'&',samp_yv[i]*1e6,'$\pm$',samp_yve[i]*1e6,'&',samp_ar[i]*1e6,'$\pm$',samp_are[i]*1e6,'&',samp_at[i]*1e6,'$\pm$',samp_ate[i]*1e6,'&',samp_pOld[i],'&',(samp_xchi2r[i]+samp_ychi2r[i])/2.0,'&',samp_arz0[i]/samp_are[i],'\\\\'))
            else:
                out.write(fmt_noa % (samp_name[i],'&',samp_mag[i],'&',samp_epoch[i],'&',samp_x[i],'$\pm$',samp_xe[i],'&',samp_y[i],'$\pm$',samp_ye[i],'&',samp_R[i],'&',samp_t0[i],'&',samp_xv[i]*1e6,'$\pm$',samp_xve[i]*1e6,'&',samp_yv[i]*1e6,'$\pm$',samp_yve[i]*1e6,'&','-','&','-','&',samp_pOld[i],'&',(samp_xchi2r[i]+samp_ychi2r[i])/2.0,'&','-','\\\\'))


        out.write('\\hline \n')
        out.write('\\insertTableNotes \n')
        out.write('\\end{longtable} \n')
        out.write('\\end{ThreePartTable} \n')
        out.write('\\end{landscape} \n')
        out.write('\\end{document} \n')
        out.close()
        #close tex table file
 
 
        #begin tex file for output table
        #out = open(tableRoot+'mn_inner_sample_table.tex','w')
        #out.write('\\begin{ThreePartTable} \n')
        #out.write('\\begin{TableNotes} \n')
        #out.write('\\item [a] Number of epochs \n')
        #out.write('\\item [b] Probability of being late-type, reported in \\citet{do13l} \n')
        #out.write('\\item [c] Best kinematic fit, determined by F Tests \n')
        #out.write('\\item [d] Average $\chi^2_r$ between $X$ and $Y$ direction for the best kinematic fit \n')
        #out.write('\\end{TableNotes} \n')
        #out.write('\\begin{longtable}{*{13}{c}} \n')
        #out.write("\\caption{Stars in Primary Sample}\\label{tab:like_samp_data} \n")
        #out.write('\\hline \n')
        #out.write("Star & K' & X & Y & R$_{2D}$ & n\\tnote{a} & $v_x$ & $v_y$ & $p_{old}$\\tnote{b} & Fit\\tnote{c} & $\chi^2_r$\\tnote{d} \n")
        #out.write("&&    & \\multicolumn{3}{c}{(arcsec)} &   & \\multicolumn{2}{c}{($\mu$as/yr)} &   &   &    \n")
        #out.write('\\hline \n')
        #out.write('\\midrule\\endhead \n')
        #out.write('\\bottomrule\\endfoot \n')
        #fmt_noa = '%15s  %1s  %5.1f  %1s  %5.1f  %1s  %5.1f  %1s  %5.1f  %1s  %2d  %1s  %6.1f  %5s  %6.1f  %1s  %6.1f  %5s  %6.1f  %1s  %6.2f  %1s  %5s  %1s  %6.2f  %4s\n'
        #out.write(fmt_noa % (tmpStar.name,'&',tmpStar.mag,'&',tmpStar.x0,'&',tmpStar.y0,'&',tmpStar.R,'&',tmpStar.epoch,'&',tmpStar.xv*1e6,'$\pm$',tmpStar.xve*1e6,'&',tmpStar.yv*1e6,'$\pm$',tmpStar.yve*1e6,'&',tmpStar.pOld,'&',tmpStar.fit,'&',(tmpStar.xchi2r+tmpStar.ychi2r)/2.0,'\\\\'))

        #out.write('\\hline \n')
        #out.write('\\insertTableNotes \n')
        #out.write('\\end{longtable} \n')
        #out.write('\\end{ThreePartTable} \n')
        #out.write('\\end{landscape} \n')
        #out.write('\\end{document} \n')
        #out.close()
        #close tex table file
        

        #selects stars with reduced chi^2 and unphysical accelerations with significance below set cuts
        #flagdex = np.where((samp_xchi2r < chiCut) & (samp_ychi2r < chiCut) & ((samp_ar/samp_are) < sigmaCut) & (np.abs(samp_at/samp_ate) < sigmaCut) & (samp_fit > 1.0))[0]
        forAG = np.where((samp_xchi2r < chiCut) & (samp_ychi2r < chiCut) & ((samp_ar/samp_are) < sigmaCut) & (np.abs(samp_at/samp_ate) < sigmaCut))[0]
        AG37 = np.where((samp_xchi2r < chiCut) & (samp_ychi2r < chiCut) & ((samp_ar/samp_are) < sigmaCut) & (np.abs(samp_at/samp_ate) < sigmaCut) & (samp_epoch > 37))[0]
        nonphys = np.where(((samp_ar/samp_are) >= sigmaCut) | (np.abs(samp_at/samp_ate) >= sigmaCut))[0]
        grt_chi2r = np.maximum(samp_xchi2r,samp_ychi2r)
        resdex = np.where((samp_name == 'S3-327') | (samp_name == 'S5-53') | (samp_name == 'S6-68') | (samp_name == 'S6-61') | (samp_name == 'S6-74') | (samp_name == 'S6-53') | (samp_name == 'S5-69') | (samp_name == 'S3-136') | (samp_name == 'S3-279') | (samp_name == 'S3-125') | (samp_name == 'S3-200') | (samp_name == 'S3-15') | (samp_name == 'S2-117') | (samp_name == 'S1-45'))[0]
        
        flagdex = np.where((samp_epoch > 27) & (samp_name != 'S3-327') & (samp_name != 'S5-53') & (samp_name != 'S6-68') & (samp_name != 'S6-61') & (samp_name != 'S6-74') & (samp_name != 'S6-53') & (samp_name != 'S5-69') & (samp_name != 'S3-136') & (samp_name != 'S3-279') & (samp_name != 'S3-125') & (samp_name != 'S3-200') & (samp_name != 'S3-15') & (samp_name != 'S2-117') & (samp_name != 'S1-45'))[0] #samp_modelPL > 0.128)[0]
        upLim = np.where((samp_ar + 3.0*samp_are) <= 0.0)[0]
        #accel_names = np.array([samp_name[ii] for ii in flagdex])
        
        #percent accel sample by radius
        perc_hist = np.zeros(5)
        perc_bins = np.array([0.0,1.0,2.0,3.0,4.0,5.0])
        for i in range(5):
            tmpdex_acc = np.where((samp_R[flagdex] >= perc_bins[i]) & (samp_R[flagdex] < perc_bins[i+1]))[0]
            tmpdex = np.where((samp_R >= perc_bins[i]) & (samp_R < perc_bins[i+1]))[0]
            perc_hist[i] = float(len(tmpdex_acc))/len(tmpdex)

        pdb.set_trace()
        py.close()
        py.bar([0.5,1.5,2.5,3.5,4.5],perc_hist,width=1.0)
        py.xlabel('Radius (arcsec)')
        py.ylabel('Percent in accel sample')
        py.savefig(plotRoot+'perc_accel_samp_R_hist.png')
        py.close()
        
        pass_ft = np.where(samp_fit > 1.0)[0]
        py.clf()
        py.plot(samp_R,samp_arz0/samp_are,'.',label='Inner Sample')
        py.plot(samp_R[pass_ft],samp_arz0[pass_ft]/samp_are[pass_ft],'.',label='Pass F-Test')
        py.xlabel('R$_{2D}$ (as)')
        py.ylabel(r'|a$_R$(z=0)| / $\sigma_R$')
        py.legend()
        py.savefig(plotRoot+'arz0_are_R2d.png')
        
        py.clf()
        py.errorbar(samp_at[flagdex]*1e6,samp_ar[flagdex]*1e6,xerr=samp_ate[flagdex]*1e6,yerr=samp_are[flagdex]*1e6,fmt='.')
        py.xlabel(r'a$_T$ ($\mu$as/yr$^2$)')
        py.ylabel(r'a$_R$ ($\mu$as/yr$^2$)')
        py.savefig(plotRoot+'at_ar_sample.png')
        py.clf()
        py.Circle((0,0),3)
        #py.plot(samp_at/samp_ate,samp_ar/samp_are,'.',label='Inner Sample')
        py.plot(samp_at[flagdex]/samp_ate[flagdex],samp_ar[flagdex]/samp_are[flagdex],'o',label='Accel Sample')
        py.xlabel(r'a$_T$ / $\sigma_T$')
        py.ylabel(r'a$_R$ / $\sigma_R$')
        #py.legend()
        py.savefig(plotRoot+'at_ar_sigma.png')
        py.clf()
        py.hist(samp_ar[flagdex]*1e6,bins=10)
        py.xlabel(r'a$_R$ ($\mu$as/yr$^2$)')
        py.savefig(plotRoot+'ar_hist_sample.png')
        py.clf()
        py.hist(samp_at[flagdex]*1e6,bins=10)
        py.xlabel(r'a$_T$ ($\mu$as/yr$^2$)')
        py.savefig(plotRoot+'at_hist_sample.png')
        py.clf()
        py.errorbar(samp_R[flagdex],samp_ar[flagdex]*1e6,yerr=samp_are[flagdex]*1e6,fmt='.')
        py.xlabel(r'R$_2D$ (as)')
        py.ylabel(r'a$_R$ ($\mu$as/yr$^2$)')
        py.savefig(plotRoot+'ar_R_samp.png')
        py.clf()
        py.errorbar(samp_R[flagdex],samp_at[flagdex]*1e6,yerr=samp_ate[flagdex]*1e6,fmt='.')
        py.xlabel(r'R$_2D$ (as)')
        py.ylabel(r'a$_T$ ($\mu$as/yr$^2$)')
        py.savefig(plotRoot+'at_R_samp.png')
        py.clf()
        
        rbins = np.arange(0.0,5.1,0.5)
        rmid = np.zeros(len(rbins)-1)
        ave_epoch = np.zeros(len(rbins)-1)
        for i in range(len(rbins)-1):
            rtdex = np.where((samp_R > rbins[i]) & (samp_R <= rbins[i+1]))[0]
            ave_epoch[i] = np.average(samp_epoch[rtdex])
            rmid[i] = (rbins[i] + rbins[i+1])/2.0
                
        py.clf()
        py.plot(rmid,ave_epoch)
        py.xlabel('R (as)')
        py.ylabel('Average epoch')
        py.savefig(plotRoot+'ave_epoch.png')
        py.clf()
        py.plot(samp_epoch,grt_chi2r,'.',label='Inner Sample')
        py.plot(samp_epoch[nonphys],grt_chi2r[nonphys],'o',label='Sig Non-phys')
        py.legend(numpoints=1)
        py.xlabel('Epochs')
        py.ylabel(r'$\tilde{\chi}^2$') #largest chi^2 reduced between X and Y directions
        py.savefig(plotRoot+'largest_chi2r_epoch_nonphys.png')
        py.clf()
        py.plot(samp_R,grt_chi2r,'.',label='Inner Sample')
        py.plot(samp_R[nonphys],grt_chi2r[nonphys],'o',label='Sig Non-phys')
        py.legend(numpoints=1)
        py.xlabel('R2d (as)')
        py.ylabel(r'$\tilde{\chi}^2$') #largest chi^2 reduced between X and Y directions
        py.savefig(plotRoot+'largest_chi2r_R_nonphys.png')
        py.clf()
        py.plot(samp_modelPL,grt_chi2r,'.',label='Inner Sample')
        py.plot(samp_modelPL[nonphys],grt_chi2r[nonphys],'o',label='Sig Non-phys')
        py.legend(numpoints=1)
        py.xlabel('Total Path Length (as)') #total path star covered across observations
        py.ylabel(r'$\tilde{\chi}^2$') #largest chi^2 reduced between X and Y directions
        py.savefig(plotRoot+'largest_chi2r_tot_pathlength_nonphys.png')
        py.clf()
        py.plot(samp_epoch,np.abs(samp_at/samp_ate),'.',label='Inner Sample')
        py.plot(samp_epoch[resdex],np.abs(samp_at[resdex]/samp_ate[resdex]),'X', ms=10,label='Near Resolved Star')
        py.plot(samp_epoch[nonphys],np.abs(samp_at[nonphys]/samp_ate[nonphys]),'o',label='Sig Non-phys')
        py.legend(numpoints=1)
        py.xlabel('Epochs')
        py.ylabel(r'$\sigma_T$')
        py.savefig(plotRoot+'sigma_t_epoch_nonphys.png')
        py.clf()
        py.plot(grt_chi2r,np.abs(samp_at/samp_ate),'.',label='Inner Sample')
        py.plot(grt_chi2r[nonphys],np.abs(samp_at[nonphys]/samp_ate[nonphys]),'o',label='Sig Non-phys')
        py.legend(numpoints=1)
        py.xlabel(r'$\tilde{\chi}^2$') #largest chi^2 reduced between X and Y directions
        py.ylabel(r'$\sigma_T$')
        py.savefig(plotRoot+'sigma_t_largest_chi2r_nonphys.png')
        py.clf()
        py.plot(samp_mag,np.abs(samp_at/samp_ate),'.',label='Inner Sample')
        py.plot(samp_mag[nonphys],np.abs(samp_at[nonphys]/samp_ate[nonphys]),'o',label='Sig Non-phys')
        py.legend(numpoints=1)
        py.xlabel("K' (mag)")
        py.ylabel(r'$\sigma_T$')
        py.savefig(plotRoot+'sigma_t_kmag_nonphys.png')
        py.clf()
        py.plot(samp_modelPL,np.abs(samp_at/samp_ate),'.',label='Inner Sample')
        py.plot(samp_modelPL[resdex],np.abs(samp_at[resdex]/samp_ate[resdex]),'X', ms=10,label='Near Resolved Star')
        py.plot(samp_modelPL[nonphys],np.abs(samp_at[nonphys]/samp_ate[nonphys]),'o',label='Sig Non-phys')
        py.legend(numpoints=1)
        py.xlabel('Total Path Length (as)') #total path star covered across observations
        py.ylabel(r'$\sigma_T$')
        py.savefig(plotRoot+'sigma_t_tot_pathlength_nonphys.png')
        py.clf()
        py.plot(samp_modelPL,samp_mag,'.',label='Inner Sample')
        py.plot(samp_modelPL[nonphys],samp_mag[nonphys],'o',label='Sig Non-phys')
        py.legend(numpoints=1)
        py.xlabel('Total Path Length (as)') #total path star covered across observations
        py.ylabel("K' (mag)")
        py.savefig(plotRoot+'kmag_pathlength_nonphys.png')
        py.clf()
        hist,bins,junk = py.hist(samp_mag,bins=6,label='Entire Sample')
        py.hist(samp_mag[flagdex],bins=bins,label='Accel Sample')
        py.yscale('log')
        py.legend()
        py.xlabel("K'")
        py.savefig(plotRoot+'KLF_accel_samplog.png')
        py.clf()
        hist,bins,junk = py.hist(samp_mag,bins=6,label='Entire Sample')
        py.hist(samp_mag[flagdex],bins=bins,label='Accel Sample')
        py.legend()
        py.xlabel("K'")
        py.savefig(plotRoot+'KLF_accel_samp.png')
        py.clf()
        posOnly = np.where(samp_modelPL <= 0.065)[0]
        hist,bins,junk = py.hist(samp_R[posOnly],bins=6,label='Pos Sample')
        py.hist(samp_R[flagdex],bins=bins,label='Accel Sample')
        py.legend()
        py.xlabel('Radius (arcsec)')
        py.savefig(plotRoot+'radius_accel_samp.png')
        py.clf()
        hist,bins,junk = py.hist(samp_pOld,bins=20,label='Entire Sample')
        py.hist(samp_pOld[flagdex],bins=bins,label='Accel Sample')
        py.legend()
        py.xlabel('Prob Late-Type')
        py.savefig(plotRoot+'pOld_accel_samp.png')
        py.clf()
        hist,bins,junk = py.hist(np.sqrt(samp_xchi2r**2 + samp_ychi2r**2),bins=30,label='Entire Sample')
        py.hist(np.sqrt(samp_xchi2r[flagdex]**2 + samp_ychi2r[flagdex]**2),bins=bins,label='Accel Sample')
        py.legend()
        py.xlabel(r'$\chi^2_{red}$')
        py.savefig(plotRoot+'rchi2_accel_samp.png')
        py.clf()
        bins = np.arange(0.0,51.0,1.0)
        hist,bins,junk = py.hist(samp_xchi2r,bins=bins,label='X',histtype='step')
        py.hist(samp_ychi2r,bins=bins,label='Y',histtype='step')
        py.plot([7,7],[0,61])
        py.ylim([0,60])
        py.legend()
        py.xlabel(r'$\chi^2_{red}$')
        py.savefig(plotRoot+'rchi2_XY_samp.png')
        py.clf()

        #to cgs units
        samp_R *= dist * cm_in_au
        #samp_pathLength *= dist * cm_in_au
        samp_ar *= asy_to_kms * 1e5 / sec_in_yr
        samp_are *= asy_to_kms * 1e5 / sec_in_yr
            
        sortRdex = np.argsort(samp_R)
        ar_z0 = -GM / samp_R[sortRdex[flagdex]]**2 #radial acceleration at a given projected radius if r = R or line of sight distance, z = 0, in cm/s^2
        ar_z0e = GMe / samp_R[sortRdex[flagdex]]**2 #error in radial acceleration at given r (assuming z=0), in cm/s^2
        pdb.set_trace()
        #py.clf()
        #py.fill_between(samp_R[sortRdex]/1e17, ar_z0-ar_z0e,ar_z0+ar_z0e,color='gray')
        #py.plot(samp_R[sortRdex]/1e17,ar_z0)
        #py.errorbar(samp_R/1e17,samp_ar,yerr=samp_are,fmt='.',label='Entire Sample')
        #py.errorbar(samp_R[forAG]/1e17,samp_ar[forAG],fmt='.',yerr=samp_are[forAG],label=r'Low $\chi^2$ + no unphysical accel')
        #py.errorbar(samp_R[flagdex]/1e17,samp_ar[flagdex],fmt='o',yerr=samp_are[flagdex],label='Accel Sample')
        #py.legend(numpoints=1)
        #py.xlabel(r'Radius (10$^17$ cm)')
        #py.ylabel('Radial acceleration (cm/s$^2$)')
        #py.savefig(plotRoot+'ar_accel_samp.png')
        py.clf()
        py.fill_between(samp_R[sortRdex[flagdex]]/1e17, ar_z0-ar_z0e,ar_z0+ar_z0e,color='gray')
        py.plot(samp_R[sortRdex[flagdex]]/1e17,ar_z0)
        #py.errorbar(samp_R/1e17,samp_ar,yerr=samp_are,fmt='.',label='Entire Sample')
        #tmpdex = np.where(samp_R <= 1.2e17)[0]
            #for ii in tmpdex:
            #py.annotate(samp_name[ii],(samp_R[ii]/1e17,samp_ar[ii]))
        #py.errorbar(samp_R[AG37]/1e17,samp_ar[AG37],fmt='.',yerr=samp_are[AG37],label=r'Low $\chi^2$ + no unphysical accel + N$>$37')
        py.errorbar(samp_R[flagdex]/1e17,samp_ar[flagdex],fmt='o',yerr=samp_are[flagdex],label='Accel Sample')
        #py.plot(samp_R[resdex]/1e17,samp_ar[resdex],'X',ms=15,label='Near Resolved Star')
#py.errorbar(samp_R[upLim]/1e17,(samp_ar[upLim]+3.0*samp_are[upLim]),yerr=0.003,fmt='.',uplims=True,label='Limits')
#py.legend(numpoints=1,loc=4)
        py.xlabel(r'Radius (10$^{17}$ cm)')
        py.ylabel('Radial acceleration (cm/s$^2$)')
        py.ylim([-0.04,0.02])
        py.savefig(plotRoot+'ar_accel_samp_zoom.png')
        py.clf()
        py.errorbar(samp_epoch,samp_ar+GM/samp_R**2,yerr=samp_are,fmt='.',label='Entire Sample')
        py.errorbar(samp_epoch[forAG],samp_ar[forAG]+GM / samp_R[forAG]**2,fmt='.',yerr=samp_are[forAG],label=r'Low $\chi^2$ + no unphysical accel')
        py.errorbar(samp_epoch[flagdex],samp_ar[flagdex]+GM / samp_R[flagdex]**2,fmt='o',yerr=samp_are[flagdex],label='Accel Sample')
        py.legend(numpoints=1)
        py.xlabel(r'Epochs')
        py.ylabel('Radial acceleration - $a_R$(z=0) (cm/s$^2$)')
        py.ylim([-0.1,0.1])
        py.savefig(plotRoot+'ar_accel-z0_epoch.png')
        py.clf()
        py.plot(samp_ar,samp_are,'.',label='Entire Sample')
        py.plot(samp_ar[flagdex],samp_are[flagdex],'.',label='Accel Sample')
        py.xlabel('Radial acceleration (cm/s$^2$)')
        py.ylabel('Acceleration Error (cm/s$^2$)')
        #py.xscale('log')
        py.xlim([-0.05,0.05])
        py.yscale('log')
        py.legend()
        py.savefig(plotRoot+'are_ar.png')
        py.clf()
        py.plot(samp_epoch,samp_pOld,'.',label='Entire Sample')
        py.plot(samp_epoch[flagdex],samp_pOld[flagdex],'.',label='Accel Sample')
        py.xlabel('Number of Epochs')
        py.ylabel('Prob Late-Type')
        py.legend()
        py.savefig(plotRoot+'pOld_epoch.png')
        py.clf()
        py.plot(samp_R/(dist * cm_in_au),samp_pOld,'.',label='Entire Sample')
        py.plot(samp_R[flagdex]/(dist * cm_in_au),samp_pOld[flagdex],'.',label='Accel Sample')
        py.xlabel('R (arcsec)')
        py.ylabel('Prob Late-Type')
        py.legend()
        py.savefig(plotRoot+'pOld_R.png')
        py.clf()
        py.plot(samp_R/cm_in_pc,samp_ar,'.')
        py.xlabel('Path Lenth (pc)')
        py.ylabel('Radial acceleration (cm/s$^2$)')
        py.savefig(plotRoot+'ar_pathLength.png')
        py.clf()
        py.plot(samp_R/cm_in_pc,samp_at,'.')
        py.xlabel('Path Lenth (pc)')
        py.ylabel('Tangential acceleration (cm/s$^2$)')
        py.savefig(plotRoot+'at_pathLength.png')
        py.clf()
        py.plot(samp_xchi2r,samp_ychi2r,'.',label='Entire Sample')
        py.plot(samp_xchi2r[flagdex],samp_ychi2r[flagdex],'o',label='Accel Sample')
        py.legend()
        py.xscale('log')
        py.yscale('log')
        py.xlabel(r'Reduced $\chi^2_X$')
        py.ylabel(r'Reduced $\chi^2_Y$')
        py.savefig(plotRoot+'redChi2_x_y.png')
        py.clf()
        py.plot(samp_epoch,np.sqrt(samp_xchi2r**2 + samp_ychi2r**2),'.',label='Entire Sample')
        py.plot(samp_epoch[forAG],np.sqrt(samp_xchi2r[forAG]**2 + samp_ychi2r[forAG]**2),'.',label=r'Low $\chi^2$ + no unphysical accel')
        py.plot(samp_epoch[flagdex],np.sqrt(samp_xchi2r[flagdex]**2 + samp_ychi2r[flagdex]**2),'o',label='Accel Sample')
        py.legend(numpoints=1)
        py.xlabel(r'Epochs')
        py.ylabel(r'$\chi^2_{red}$')
        #py.ylim([-0.04,0.02])
        py.savefig(plotRoot+'red_chi2_epoch.png')
        py.clf()
        py.plot(samp_epoch,samp_xchi2r,'.',label='X')
        py.plot(samp_epoch,samp_ychi2r,'.',label=r'Y')
        py.legend(numpoints=1)
        py.xlabel(r'Epochs')
        py.ylabel(r'$\chi^2_{red}$')
        #py.ylim([-0.04,0.02])
        py.savefig(plotRoot+'red_chi2_XY_epoch.png')
        py.clf()
        py.plot(samp_R/1e17,samp_epoch,'.',label='Entire Sample')
        py.plot(samp_R[forAG]/1e17,samp_epoch[forAG],'.',label=r'Low $\chi^2$ + no unphysical accel')
        py.plot(samp_R[flagdex]/1e17,samp_epoch[flagdex],'o',label='Accel Sample')
        py.legend(numpoints=1,loc=4)
        py.ylabel(r'Epochs')
        py.xlabel(r'R ($10^{17}$ cm)')
        #py.ylim([-0.04,0.02])
        py.savefig(plotRoot+'epoch_R.png')
        py.clf()
        
        
        #for stars outside of chi^2 and unphysical acceleraiton cut, set their error to -1, C++ code will not use their accelerations as we do not trust them
        use_are = np.zeros(len(samp_are)) - 1.0
        use_are[flagdex] = np.sqrt(samp_are[flagdex]**2 + samp_at[flagdex]**2)
        
        #save stars in dat file
        np.savetxt(cRoot+'stars_mn'+label+'.dat',np.transpose([samp_R,samp_ar,use_are,samp_pOld]),delimiter=' ')
        np.savetxt(cRoot+'xy_as'+label+'.dat',np.transpose([samp_x,samp_y]),delimiter=' ')
        for ii in flagdex:
            scYoung.plotStar(samp_name[ii],accel_fit=True,poly='polyfit_5sig/fit',points='points_5sig/')
            os.system('mv /u/schappell/plots/*'+str(samp_name[ii])+'*.png /u/schappell/plots/accelSamp')
        
        #imgFile='/u/ghezgroup/data/gc/09maylgs/combo/mag09maylgs_sgra_dim_kp.fits'
        imgFile='/u/ghezgroup/data/gc/14maylgs2/combo/mag14maylgs2_kp.fits'
        #sgra=[624.5,726.3]
        sgra=[569.3,672.3]#obtained from starfinder/align/aligh_kp_0.8.sgra
        scale = 0.00995
        img = pyfits.getdata(imgFile)
        imgsize = (img.shape)[0]
        pixL = np.arange(0,imgsize)
        xL = [-1*(xpos - sgra[0])*scale for xpos in pixL]
        yL = [(ypos - sgra[1])*scale for ypos in pixL]
        fig = py.figure(figsize=(8,8))
        fig.subplots_adjust(left=0.1,right=0.95,top=0.95)
        fig.clf()
        ax = fig.add_subplot(111)
        ax.imshow(np.log10(img+1), aspect='equal', interpolation='bicubic',
                  extent=[max(xL), min(xL), min(yL), max(yL)],vmin=2.2,vmax=5,
                  origin='lowerleft', cmap=py.cm.gray_r)
        #tmpdex = np.where((samp_xchi2r < chiCut) & (samp_ychi2r < chiCut))[0]
        #py.plot(samp_x[tmpdex],samp_y[tmpdex],'gs',ms=10,label=r'$\chi^2 < 10$')
        #tmpdex = np.where(np.abs(samp_at/samp_ate) < sigmaCut)[0]
        #py.plot(samp_x[tmpdex],samp_y[tmpdex],'rX',ms=9,label=r'|$\sigma_T$| $< $'+str(sigmaCut))
        #tmpdex = np.where((samp_ar/samp_are) < sigmaCut)[0]
        #py.plot(samp_x[tmpdex],samp_y[tmpdex],'bP',ms=8,label=r'$\sigma_R < +$'+str(sigmaCut))
        #py.plot(samp_x,samp_y,'k.',ms=7)
        #tmpdex = np.where(samp_fit > 1.0)[0]
        #py.plot(samp_x[tmpdex],samp_y[tmpdex],'y.',ms=7,label='Pass F Test')
        tmpdex = np.where((samp_xchi2r < chiCut) & (samp_ychi2r < chiCut) & (np.abs(samp_at/samp_ate) < sigmaCut) & ((samp_ar/samp_are) < sigmaCut) & (samp_epoch > 37))[0]
        py.plot(samp_x[flagdex],samp_y[flagdex],'.')
        #py.legend(numpoints=1)
        py.xlim([-5.5,5.5])
        py.ylim([-6,4])
        py.xlabel('RA (arcsec)')
        py.ylabel('DEC (arcsec)')
        py.savefig(plotRoot+'accel_samp_xy.png')
        py.clf()



    
    def schodel_samp(self,label='',innerCut=5.0,outerCut=15.0,magCut=17.75,lmagCut=0.0):
        R2d = np.array([])
        mag = np.array([])
        oldProb = np.array([])
        x = np.array([])
        y = np.array([])
        
        #to correct for extinction, calculate pixel location on fits file of dust for each star
        ext_scale = 0.02705 #arcsec/pixel
        ext_center = [808,611]
        extMap = pyfits.getdata('/u/schappell/Downloads/AKs_fg6.fits')
        database = mysql.connector.connect(host="galaxy1.astro.ucla.edu",user="dbread",passwd="t36fCEtw",db="gcg")
        cur = database.cursor()
        cur.execute('SELECT r2d,k,x,y FROM schoedel2010')
        for row in cur:
            
            Xext = int(round(row[2]/ext_scale))+ext_center[0]
            Yext = int(round(row[3]/ext_scale))+ext_center[0]
            if ((Xext < 0) | (Yext < 0)):
                print 'Something is wrong, star is calculated as being off extinction map'
                return
            else:
                if ((Xext < 1600) & (Yext < 1600)):
                    # set all stars in Schodel sample as being late-type (approx. true for larger distances), save R and corrected mag info
                    oldProb = np.append(oldProb,1.0)
                    R2d = np.append(R2d,row[0])
                    mag = np.append(mag,row[1] + 2.7 - extMap[Xext,Yext])
                    x = np.append(x,row[2])
                    y = np.append(y,row[3])

        #radius and mag cuts
        mdex = np.where((R2d > innerCut) & (R2d < outerCut) & (mag < magCut) & (mag > lmagCut))[0]
        pdb.set_trace()
        #into cgs units and save to file
        R2d *= dist * cm_in_au
        np.savetxt(cRoot+'maser_mn'+label+'.dat',np.transpose([R2d[mdex],oldProb[mdex],x[mdex],y[mdex]]),delimiter=' ')



    def run_MN(self,label_in='',label_out='',innerCut=5.0,Rcut=5.0,outerCut=15.0,mnAlpha=6.0,mnDelta=4.0,mnBreak=0.5,max_r=5.0,situation=3,nonRadial=1):
        #in units of pc
        outerCut *= dist / au_in_pc
        innerCut *= dist / au_in_pc
        Rcut *= dist / au_in_pc
        mnRoot = mnoutRoot+'data'+label_out+'_'
        
        #compile and run
        os.system('g++ sc_mn_FINAL.cpp gauss_legendre.c -o sc_mn -std=c++11 -lpthread -I'+cRoot+' -I/u/schappell/code/c/boost/config -L'+homeRoot+'multinest/MultiNest/lib -lmultinest')
        outCommand = './sc_mn '+mnRoot+' '+str(mnAlpha)+' '+str(mnDelta)+' '+str(mnBreak)+' '+str(max_r)+' '+str(Rcut)
        outCommand += ' '+str(situation)+' '+str(innerCut)+' '+str(outerCut)+' '+str(nonRadial)+' '+label_in
        os.system(outCommand)




    def in_gcows(self,file='/u/schappell/Downloads/NIRC2 radial mask/nirc2_gcows_2010_all_mask.fits',savefile='/u/schappell/code/c/gcows_field.dat'):
        #param for translating arcsec into pixels
        plate_scale = 0.00995
        gcows_zero = np.array([1500.0,1500.0])
        gcows = pyfits.getdata(file)
        gcows_dex = np.where(gcows > 0)
        if (savefile != None):
            #savefile with pixels in mask
            np.savetxt(savefile,np.transpose([gcows_dex[1],gcows_dex[0]]),delimiter=' ')
        for tmpStar in self.stars:
            try:
                x_pixels = int(round((tmpStar.x0 / plate_scale) + gcows_zero[0]))
                y_pixels = int(round((tmpStar.y0 / plate_scale) + gcows_zero[1]))
                ingcows = gcows[y_pixels,x_pixels]
            except:
                continue
            #if star is outside mask in negative direction, set it as being off footprint
            if ((x_pixels < 0.0) | (y_pixels < 0.0)):
                ingcows = 0
            #update for given star
            tmpStar.set_gcows(ingcows)



    def updateErr_all(self,chainsDir=None,addErr=False):
        #check, if chains directory to update errors with uncertainty in Sgr A* position and velocity, that this has not already been done
        if ((chainsDir!=None) & (hasattr(self,'chainsErrdone') == False)):
            print 'Updating position and velocity errors with errors from efit'
            origin_val = asciidata.open(self.root+chainsDir+'efit_summary.txt')
            ori_x0e = origin_val[25][0]
            ori_y0e = origin_val[26][0]
            ori_vxe = origin_val[27][0]
            ori_vye = origin_val[28][0]
            #values of uncertainty in chians
            t_0 = 2000.0

            for tmpStar in self.stars:
                #update undertainty in position and velocity for all stars
                tmpStar.x0e = math.sqrt(tmpStar.x0e**2 + ori_x0e**2 + ((tmpStar.xt0 - t_0)*ori_vxe)**2)
                tmpStar.y0e = math.sqrt(tmpStar.y0e**2 + ori_y0e**2 + ((tmpStar.yt0 - t_0)*ori_vye)**2)
                tmpStar.xve = math.sqrt(tmpStar.xve**2 + ori_vxe**2)
                tmpStar.yve = math.sqrt(tmpStar.yve**2 + ori_vye**2)
    
            #set this value so this procedure will not be done multiple times
            self.chainsErrdone = 1.0

        #if additive empirical error is going to run, check that it has not been done already
        if ((addErr==True) & (hasattr(self,'addError') == False)):
            print 'Updating error in radial and tangential acceleration with additive error'
            fit_mag = np.array([])
            fit_at = np.array([])
            fit_ate = np.array([])
            fit_are = np.array([])
            #cycle through stars, get magnitude and acceleration info
            for tmpStar in self.stars:
                if (hasattr(tmpStar,'a_at') & (hasattr(tmpStar,'mag'))):
                    if (tmpStar.mag < 16.5):
                        fit_mag = np.append(fit_mag, tmpStar.mag)
                        if (tmpStar.fit == 'Jerk'):
                            fit_at = np.append(fit_at, tmpStar.at)
                            fit_ate = np.append(fit_ate, tmpStar.ate)
                            fit_are = np.append(fit_are, tmpStar.are)
                        else:
                            fit_at = np.append(fit_at, tmpStar.a_at)
                            fit_ate = np.append(fit_ate, tmpStar.a_ate)
                            fit_are = np.append(fit_are, tmpStar.a_are)
            #call deltaError function to break stars into magnitude bins and fit empirical error for each bin
            deltaArr, atBins = deltaError(fit_mag,fit_at,fit_ate)
            #add array of empirical additive errors as attribute
            self.addError = deltaArr
            #ass limits of magnitude bins as attribute
            self.addMag = atBins

            ave_are = np.zeros(len(atBins)-1)
            for i in range(len(atBins)-1):
                tmpdex = np.where((fit_mag > atBins[i]) & (fit_mag <= atBins[i+1]))[0]
                ave_are[i] = np.median(fit_are[tmpdex])
            py.clf()
            mid_point = np.array([(atBins[ii]+atBins[ii+1])/2.0 for ii in range(len(deltaArr))])
            py.plot(mid_point,deltaArr*1e6,'o',label='Additive Error')
            py.plot(mid_point,ave_are*1e6,'o',label='Median $\sigma_{a,R}$')
            py.legend(numpoints=1)
            py.xlabel("Magnitude (K')")
            py.ylabel('Error ($\mu$as/yr$^2$)')
            py.savefig(plotRoot+'additive_error_magnitude.png')
            #plot additive error for every magnitude bin
            py.clf()
            py.figure()
            py.axes([0.08,0.1,0.38,0.82])
            hist,bins,junk=py.hist(fit_at/fit_ate,bins=400)
            py.xlim([-10,10])
            py.ylim([0,220])
            py.xlabel('$\sigma_T$')
            py.title('Before')
            #plot before and after of distribution of significance of tangential acceleration (acceleration / error)

            for i in range(len(fit_mag)):
                for j in range(len(atBins)-1):
                    if ((fit_mag[i] > atBins[j]) & (fit_mag[i] <= atBins[j+1])):
                        fit_ate[i] = math.sqrt(fit_ate[i]**2 + deltaArr[j]**2)

            py.axes([0.56,0.1,0.38,0.82])
            py.hist(fit_at/fit_ate,bins=bins)
            py.xlim([-10,10])
            py.ylim([0,220])
            py.xlabel('$\sigma_T$')
            py.title('After')
            py.savefig(plotRoot+'before_after_error_at.png')
            py.clf()

            for tmpStar in self.stars:
                for i in range(len(atBins)-1):
                    # update error for every star based on its magnitude
                    if (hasattr(tmpStar,'at') & hasattr(tmpStar,'mag')):
                        if ((tmpStar.mag > atBins[i]) & (tmpStar.mag <= atBins[i+1])):
                            tmpStar.ate = math.sqrt(tmpStar.ate**2 + deltaArr[i]**2)
                            tmpStar.are = math.sqrt(tmpStar.are**2 + deltaArr[i]**2)
                    if (hasattr(tmpStar,'a_at') & hasattr(tmpStar,'mag')):
                        if ((tmpStar.mag > atBins[i]) & (tmpStar.mag <= atBins[i+1])):
                            tmpStar.a_ate = math.sqrt(tmpStar.a_ate**2 + deltaArr[i]**2)
                            tmpStar.a_are = math.sqrt(tmpStar.a_are**2 + deltaArr[i]**2)
                    if (hasattr(tmpStar,'j_at') & hasattr(tmpStar,'mag')):
                        if ((tmpStar.mag > atBins[i]) & (tmpStar.mag <= atBins[i+1])):
                            tmpStar.j_ate = math.sqrt(tmpStar.j_ate**2 + deltaArr[i]**2)
                            tmpStar.j_are = math.sqrt(tmpStar.j_are**2 + deltaArr[i]**2)
                    #update all acceleration terms, so both in acceleration terms for acceleration and jerk fits and best fit
                    #value of acceleration which is attributed to the star


    def all_vz(self):
        #run get_vz function for every star in sample, gets info from database and saves it to attributes for given star
        for tmpStar in self.stars:
            tmpStar.get_vz()



    def plot_V_vs_R(self,inner=1.0):
        plot_R = np.array([])
        plot_V = np.array([])
        plot_Verr = np.array([])
        plot_yng = np.array([])
        plot_old = np.array([])

        #cycle through stars, if meet requirements of being outside radius cut, having RV measurements, and low chi^2_reduced
        for tmpStar in self.stars:
            if (tmpStar.R > inner):
                tmpStar.get_vz()
                if ((len(tmpStar.vz_err) > 0) & (tmpStar.xchi2r < 10.0) & (tmpStar.ychi2r < 10.0)):
                    plot_R = np.append(plot_R, tmpStar.R)
                    if (len(tmpStar.vz_err) > 1):
                        #if stars has more than one RV measurement, use measurement with lowest error
                        tmpdex = np.argmin(tmpStar.vz_err)
                        tmpVz = tmpStar.vz[tmpdex]
                        tmperr = tmpStar.vz_err[tmpdex]
                    else:
                        tmpVz = tmpStar.vz *1.0
                        tmperr = tmpStar.vz_err * 1.0
                    #calculate combined velocity and error
                    plot_V = np.append(plot_V, math.sqrt((tmpStar.xv*asy_to_kms)**2 + (tmpStar.yv*asy_to_kms)**2 + tmpVz**2))
                    plot_Verr = np.append(plot_Verr, math.sqrt((tmpStar.xve*asy_to_kms)**2 + (tmpStar.yve*asy_to_kms)**2 + tmperr**2))
                    
                    plot_yng = np.append(plot_yng,tmpStar.pYng)
                    plot_old = np.append(plot_old,tmpStar.pOld)

        py.clf()
        #phase space plots broken up for stars into early and late-type
        yngdex = np.where(plot_yng == 1.0)[0]
        py.hist2d(plot_R[yngdex],plot_V[yngdex],bins=4)
        py.colorbar()
        py.ylabel('Velocity (km/s)')
        py.xlabel(r'R$_{2D}$ (arcsec)')
        py.savefig(plotRoot+'yng_V_R.png')
        py.clf()
        py.errorbar(plot_R[yngdex],plot_V[yngdex],yerr=plot_Verr[yngdex],fmt='.')
        py.ylabel('Velocity (km/s)')
        py.xlabel(r'R$_{2D}$ (arcsec)')
        py.savefig(plotRoot+'yng_V_R_SCATTER.png')
        py.clf()
        olddex = np.where(plot_old == 1.0)[0]
        py.hist2d(plot_R[olddex],plot_V[olddex],bins=4)
        py.colorbar()
        py.ylabel('Velocity (km/s)')
        py.xlabel(r'R$_{2D}$ (arcsec)')
        py.savefig(plotRoot+'old_V_R.png')
        py.clf()
        py.errorbar(plot_R[olddex],plot_V[olddex],yerr=plot_Verr[olddex],fmt='.')
        py.ylabel('Velocity (km/s)')
        py.xlabel(r'R$_{2D}$ (arcsec)')
        py.savefig(plotRoot+'old_V_R_SCATTER.png')
        py.clf()
        print 'Total number of stars in yng: '+str(len(yngdex))+' and old: '+str(len(olddex))



    def plot_chi2r(self):
        pl_xchi2r = np.array([])
        pl_ychi2r = np.array([])
        pl_mag = np.array([])
        pl_epoch = np.array([])
        pl_fit = np.array([])
        pl_old = np.array([])
        pl_r2d = np.array([])
        #cycle through stars, get reduced chi^2, mag, epoch, etc information
        for tmpStar in self.stars:
            if (hasattr(tmpStar,'mag') & hasattr(tmpStar,'pOld')):
                pl_xchi2r = np.append(pl_xchi2r, tmpStar.xchi2r)
                pl_ychi2r = np.append(pl_ychi2r, tmpStar.ychi2r)
                pl_mag = np.append(pl_mag,tmpStar.mag)
                pl_epoch = np.append(pl_epoch,tmpStar.epoch)
                if (str(tmpStar.fit) == 'Vel'):
                    pl_fit = np.append(pl_fit,1)
                elif(str(tmpStar.fit) == 'Acc'):
                    pl_fit = np.append(pl_fit,2)
                elif(str(tmpStar.fit) == 'Jerk'):
                    pl_fit = np.append(pl_fit,3)
                pl_old = np.append(pl_old,tmpStar.pOld)
                pl_r2d = np.append(pl_r2d,tmpStar.R)



        py.clf()
        #plot chi^2 histograms
        usedex = np.where((pl_xchi2r < 100.0) &(pl_ychi2r < 100.0))[0]
        py.hist(pl_xchi2r[usedex],bins=100)
        py.xlabel(r'$\chi_{reduced}^2$')
        py.ylabel('Number of Stars')
        py.title(r'X $\chi_{reduced}^2$')
        py.savefig(plotRoot+'x_chi2r_hist.png')
        py.clf()
        py.hist(pl_ychi2r[usedex],bins=100)
        py.xlabel(r'$\chi_{reduced}^2$')
        py.ylabel('Number of Stars')
        py.title(r'Y $\chi_{reduced}^2$')
        py.savefig(plotRoot+'y_chi2r_hist.png')
        py.clf()
        veldex = np.where(pl_fit==1)[0]
        accdex = np.where(pl_fit==2)[0]
        jerkdex = np.where(pl_fit==3)[0]
        

        #plot chi^2 in X and Y directions against R, mag, and epoch
        #plots also show best fit for given star
        py.plot(pl_r2d[veldex],pl_xchi2r[veldex],'k.',label='Vel')
        py.plot(pl_r2d[accdex],pl_xchi2r[accdex],'b.',label='Accel')
        py.plot(pl_r2d[jerkdex],pl_xchi2r[jerkdex],'r.',label='Jerk')
        py.yscale('log')
        py.legend()
        py.xlabel('R (arcsec)')
        py.ylabel(r'X $\chi_{reduced}^2$')
        py.savefig(plotRoot+'xchi2r_R_wfit.png')
        py.clf()
        py.plot(pl_r2d[veldex],pl_ychi2r[veldex],'k.',label='Vel')
        py.plot(pl_r2d[accdex],pl_ychi2r[accdex],'b.',label='Accel')
        py.plot(pl_r2d[jerkdex],pl_ychi2r[jerkdex],'r.',label='Jerk')
        py.yscale('log')
        py.legend()
        py.xlabel('R (arcsec)')
        py.ylabel(r'Y $\chi_{reduced}^2$')
        py.savefig(plotRoot+'ychi2r_R_wfit.png')
        py.clf()
        py.plot(pl_mag[veldex],pl_xchi2r[veldex],'k.',label='Vel')
        py.plot(pl_mag[accdex],pl_xchi2r[accdex],'b.',label='Accel')
        py.plot(pl_mag[jerkdex],pl_xchi2r[jerkdex],'r.',label='Jerk')
        py.yscale('log')
        py.legend()
        py.xlabel("K' (mag)")
        py.ylabel(r'X $\chi_{reduced}^2$')
        py.savefig(plotRoot+'xchi2r_mag_wfit.png')
        py.clf()
        py.plot(pl_mag[veldex],pl_ychi2r[veldex],'k.',label='Vel')
        py.plot(pl_mag[accdex],pl_ychi2r[accdex],'b.',label='Accel')
        py.plot(pl_mag[jerkdex],pl_ychi2r[jerkdex],'r.',label='Jerk')
        py.yscale('log')
        py.legend()
        py.xlabel("K' (mag)")
        py.ylabel(r'Y $\chi_{reduced}^2$')
        py.savefig(plotRoot+'ychi2r_mag_wfit.png')
        py.clf()
        py.plot(pl_epoch[veldex],pl_xchi2r[veldex],'k.',label='Vel')
        py.plot(pl_epoch[accdex],pl_xchi2r[accdex],'b.',label='Accel')
        py.plot(pl_epoch[jerkdex],pl_xchi2r[jerkdex],'r.',label='Jerk')
        py.yscale('log')
        py.legend()
        py.xlabel('Epoch')
        py.ylabel(r'X $\chi_{reduced}^2$')
        py.savefig(plotRoot+'xchi2r_epoch_wfit.png')
        py.clf()
        py.plot(pl_epoch[veldex],pl_ychi2r[veldex],'k.',label='Vel')
        py.plot(pl_epoch[accdex],pl_ychi2r[accdex],'b.',label='Accel')
        py.plot(pl_epoch[jerkdex],pl_ychi2r[jerkdex],'r.',label='Jerk')
        py.yscale('log')
        py.legend()
        py.xlabel('Epoch')
        py.ylabel(r'Y $\chi_{reduced}^2$')
        py.savefig(plotRoot+'ychi2r_epoch_wfit.png')
        py.clf()

        #consider constrained sample with mag < 15.5 (K'), R < 5arcsec, and non-zero prob of being late-type
        usedex = np.where((pl_mag < 15.5) & (pl_r2d < 5.0) & (pl_old > 0.0))[0]
        py.hist(pl_xchi2r[usedex],bins=100)
        py.xlabel(r'$\chi_{reduced}^2$')
        py.ylabel('Number of Stars')
        py.title(r'X $\chi_{reduced}^2$')
        py.savefig(plotRoot+'x_chi2r_hist_constrain.png')
        py.clf()
        py.hist(pl_ychi2r[usedex],bins=100)
        py.xlabel(r'$\chi_{reduced}^2$')
        py.ylabel('Number of Stars')
        py.title(r'Y $\chi_{reduced}^2$')
        py.savefig(plotRoot+'y_chi2r_hist_constrain.png')
        py.clf()
        py.plot(pl_xchi2r[usedex],pl_ychi2r[usedex],'.')
        py.xlabel(r'X $\chi_{reduced}^2$')
        py.ylabel(r'Y $\chi_{reduced}^2$')
        py.savefig(plotRoot+'X_vs_Ychi.png')
        py.clf()

        #plot constrained sample as a function of number of epochs detected, with best fit shown with color
        usedex = np.where((pl_mag < 15.5) & (pl_r2d < 5.0) & (pl_old > 0.0) & (pl_fit==1.0))[0]
        py.plot(pl_epoch[usedex],(pl_xchi2r[usedex]+pl_ychi2r[usedex])/2.0,'.k',label='Vel')
        usedex = np.where((pl_mag < 15.5) & (pl_r2d < 5.0) & (pl_old > 0.0) & (pl_fit==2.0))[0]
        py.plot(pl_epoch[usedex],(pl_xchi2r[usedex]+pl_ychi2r[usedex])/2.0,'.b',label='Accel')
        usedex = np.where((pl_mag < 15.5) & (pl_r2d < 5.0) & (pl_old > 0.0) & (pl_fit==3.0))[0]
        py.plot(pl_epoch[usedex],(pl_xchi2r[usedex]+pl_ychi2r[usedex])/2.0,'.r',label='Jerk')
        py.legend()
        py.xlabel('Epochs')
        py.ylabel(r'$\chi_{reduced}^2$')
        py.savefig(plotRoot+'epoch_aveChi.png')
        py.clf()
        py.close('all')



'''
Purpose = for set of magnitudes and tangential acceleration, breaks sample into madnitude bins with equal number
        of stars, and for each bin, fit for empirical additive error and save that value for each bin. Best fit
        empirical error is the one which results in one sigma, or 68.27% of subsample in that bin is contained within
        -1 and 1 siga_tan (tangential acceleration / error). Distribution of significance of tangential acceleration,
        which is unphysical in both positive and negative direction, should follow an approx. gaussian function,
        centered on 0 with a spread of 1.0
        
Required input:
    mag = array of magnitudes
    at/e = array of tangential accelerations and errors, need to be same size and same saize as mag array, units also
        need to be consistent between two arrays
        
Optional input:
    numbins = number of magnitude bins, set to 8 unless otherwise stated
'''
def deltaError(mag, at, ate, numbins=8):
    #sort magnitudes in order to separate out into bins of equal size
    dex = np.argsort(mag)
    numstars = len(mag)/numbins
    delta_return = np.zeros(numbins)
    mag_return = np.zeros(numbins+1)
    mag_return[0] = np.min(mag)
    print "Delta errors: "
    for i in range(numbins):
        startdex = i * numstars
        enddex = startdex + numstars - 1
        #bins of equal size
        tmp_at = at[dex[startdex:enddex]]
        tmp_ate = ate[dex[startdex:enddex]]
        #solve for empirical error in this magnitude bin
        tmp_add = fsolve(findAdd, -1e-6,args=(tmp_at,tmp_ate))
        mag_return[i+1] = mag[dex[enddex]]
        delta_return[i] = tmp_add
        #save empirical error and magnitude bin limits
        print "From K mag "+str(mag_return[i])+" to "+str(mag_return[i+1])
        print tmp_add
    
    #return values, abs needed as sign degeneracy in empirical error
    return np.abs(delta_return), mag_return


'''
Purpose = for given set of tangential acceleration and errors, calculates errors with input additive error and
        resulting fraction of stars with significance in tangential acceleration  (accel_tan / error) between 
        -1 and +1 and compares that to 68.27% and returns difference, in percent. This function is used to fit 
        empirical error by deltaError funciton.
        
Reuqired input:
    addError = additive error, in the same units as error in acceleration
    needed = arrays of tangential acceleration and error, units and size of array need to be consistent
'''
def findAdd(addError,*needed):
    at, ate = needed
    value = np.array([])
    for tmperror in addError:
        #calculate significance with additive error
        sigma = at / np.sqrt(ate**2 + tmperror**2)
        #find fraction of stars with significance betwen -1 and +1
        spreadsig = np.where((sigma <= 1.0) & (sigma >= -1.0))[0]
        percent_in = 100.0 * float(len(spreadsig))/float(len(sigma))
        value = np.append(value,(68.27 - percent_in)**2)
    
    #return residual
    return value




#Compare two aligns by examining their errors in velocity and acceleration
def compareAlign(alignA='/g/ghez/align/siyao_18_01_12_OldHolo_18FebAbs_allEpoch/allEpoch/',polyA='polyfit_4_trim/',pointsA='points_4_trim/',labelA='Old Holo',alignB='/g/ghez/align/18_02_07_newSpeckle/',polyB='polyfit_4_trim/',pointsB='points_4_trim/',labelB='New Holo',radiusCut=2.0,l_radiusCut=0.4,magCut=16.0):
    starsA = stars(alignA,polyA,pointsA)
    starsB = stars(alignB,polyB,pointsB)
    #fetch the info from the two aligns and their specificied polyfits

    #cycle through stars, both in A and B aligns and find which ones are within the mag and the radii cut
    #and which stars are in both aligns
    epochA = np.array([])
    epochB = np.array([])
    magA = np.array([])
    x_A = np.array([])
    y_A = np.array([])
    vxe_A = np.array([])
    vye_A = np.array([])
    axe_A = np.array([])
    aye_A = np.array([])
    vx_A = np.array([])
    vy_A = np.array([])
    ax_A = np.array([])
    ay_A = np.array([])
    vxe_B = np.array([])
    vye_B = np.array([])
    axe_B = np.array([])
    aye_B = np.array([])
    vx_B = np.array([])
    vy_B = np.array([])
    ax_B = np.array([])
    ay_B = np.array([])
    for tmpStarA in starsA.stars:
        if (hasattr(tmpStarA, 'mag')):
            if ((tmpStarA.mag < magCut) & (tmpStarA.a_R < radiusCut) & (tmpStarA.a_R > l_radiusCut)):
                for tmpStarB in starsB.stars:
                    if (tmpStarA.name == tmpStarB.name):
                        epochA = np.append(epochA,tmpStarA.epoch)
                        epochB = np.append(epochB,tmpStarB.epoch)
                        magA = np.append(magA, tmpStarA.mag)
                        x_A = np.append(x_A, tmpStarA.a_x0)
                        y_A = np.append(y_A, tmpStarA.a_y0)
                        vxe_A = np.append(vxe_A, tmpStarA.a_xve)
                        vye_A = np.append(vye_A, tmpStarA.a_yve)
                        axe_A = np.append(axe_A, tmpStarA.a_xae)
                        aye_A = np.append(aye_A, tmpStarA.a_yae)
                        vxe_B = np.append(vxe_B, tmpStarB.a_xve)
                        vye_B = np.append(vye_B, tmpStarB.a_yve)
                        axe_B = np.append(axe_B, tmpStarB.a_xae)
                        aye_B = np.append(aye_B, tmpStarB.a_yae)

                        vx_A = np.append(vx_A, tmpStarA.a_xv)
                        vy_A = np.append(vy_A, tmpStarA.a_yv)
                        ax_A = np.append(ax_A, tmpStarA.a_xa)
                        ay_A = np.append(ay_A, tmpStarA.a_ya)
                        vx_B = np.append(vx_B, tmpStarB.a_xv)
                        vy_B = np.append(vy_B, tmpStarB.a_yv)
                        ax_B = np.append(ax_B, tmpStarB.a_xa)
                        ay_B = np.append(ay_B, tmpStarB.a_ya)


    #calculate difference between errors, ratio of the two, and the two added in quadrature
    vxe_ratio = vxe_A/vxe_B
    vxe_diff = vxe_A - vxe_B
    vxe_quad = np.sqrt(vxe_A**2 + vxe_B**2)
    vye_ratio = vye_A/vye_B
    vye_diff = vye_A - vye_B
    vye_quad = np.sqrt(vye_A**2 + vye_B**2)
    axe_ratio = axe_A/axe_B
    axe_diff = axe_A - axe_B
    axe_quad = np.sqrt(axe_A**2 + axe_B**2)
    aye_ratio = aye_A/aye_B
    aye_diff = aye_A - aye_B
    aye_quad = np.sqrt(aye_A**2 + aye_B**2)

    vx_diff = vx_A - vx_B
    vy_diff = vy_A - vy_B
    ax_diff = ax_A - ax_B
    ay_diff = ay_A - ay_B

    #Let's make some plots!
    py.clf()
    # Error ratio plots
    py.plot(epochA,vxe_ratio,'.',label='X')
    py.plot(epochA,vye_ratio,'.',label='Y')
    py.legend(numpoints=1)
    py.xlabel('Epoch ('+labelA+')')
    py.ylabel('Vel Error Ratio ('+labelA+' / '+labelB+')')
    py.savefig(plotRoot+''+labelA+'_'+labelB+'_veRatio_epoch.png')
    py.clf()
    py.plot(epochA,axe_ratio,'.',label='X')
    py.plot(epochA,aye_ratio,'.',label='Y')
    py.legend(numpoints=1)
    py.xlabel('Epoch ('+labelA+')')
    py.ylabel('Accel Error Ratio ('+labelA+' / '+labelB+')')
    py.savefig(plotRoot+''+labelA+'_'+labelB+'_aeRatio_epoch.png')
    py.clf()
    py.plot(magA,vxe_ratio,'.',label='X')
    py.plot(magA,vye_ratio,'.',label='Y')
    py.legend(numpoints=1)
    py.xlabel('Mag')
    py.ylabel('Vel Error Ratio ('+labelA+' / '+labelB+')')
    py.savefig(plotRoot+''+labelA+'_'+labelB+'_veRatio_mag.png')
    py.clf()
    py.plot(magA,axe_ratio,'.',label='X')
    py.plot(magA,aye_ratio,'.',label='Y')
    py.legend(numpoints=1)
    py.xlabel('Mag')
    py.ylabel('Accel Error Ratio ('+labelA+' / '+labelB+')')
    py.savefig(plotRoot+''+labelA+'_'+labelB+'_aeRatio_mag.png')
    py.clf()

    # Error diff plots, divided by quad averaged
    py.plot(epochA,vxe_diff/vxe_quad,'.',label='X')
    py.plot(epochA,vye_diff/vye_quad,'.',label='Y')
    py.legend(numpoints=1)
    py.xlabel('Epoch ('+labelA+')')
    py.ylabel(r'Vel Error Delta ( ('+labelA+' - '+labelB+') / ('+labelA+'$^2$ + '+labelB+'$^2$)$^{1/2}$ )')
    py.savefig(plotRoot+''+labelA+'_'+labelB+'_veDiff_epoch.png')
    py.clf()
    py.plot(epochA,axe_diff/axe_quad,'.',label='X')
    py.plot(epochA,aye_diff/aye_quad,'.',label='Y')
    py.legend(numpoints=1)
    py.xlabel('Epoch ('+labelA+')')
    py.ylabel('Accel Error Delta ( ('+labelA+' - '+labelB+') / ('+labelA+'$^2$ + '+labelB+'$^2$)$^{1/2}$ )')
    py.savefig(plotRoot+''+labelA+'_'+labelB+'_aeDiff_epoch.png')
    py.clf()
    py.plot(magA,vxe_diff/vxe_quad,'.',label='X')
    py.plot(magA,vye_diff/vye_quad,'.',label='Y')
    py.legend(numpoints=1)
    py.xlabel('Mag')
    py.ylabel('Vel Error Delta ( ('+labelA+' - '+labelB+') / ('+labelA+'$^2$ + '+labelB+'$^2$)$^{1/2}$ )')
    py.savefig(plotRoot+''+labelA+'_'+labelB+'_veDiff_mag.png')
    py.clf()
    py.plot(magA,axe_diff/axe_quad,'.',label='X')
    py.plot(magA,aye_diff/aye_quad,'.',label='Y')
    py.legend(numpoints=1)
    py.xlabel('Mag')
    py.ylabel('Accel Error Delta ( ('+labelA+' - '+labelB+') / ('+labelA+'$^2$ + '+labelB+'$^2$)$^{1/2}$ )')
    py.savefig(plotRoot+''+labelA+'_'+labelB+'_aeDiff_mag.png')
    py.clf()
    
    # Plots of difference between velocity and acceleration values
    py.figure(figsize=(8,5))
    py.errorbar(epochA,vx_diff*1e3,yerr=vxe_quad*1e3,fmt='.',label='X')
    py.errorbar(epochA,vy_diff*1e3,yerr=vye_quad*1e3,fmt='.',label='Y')
    #py.scatter(epochA,vx_diff*1e3,c=magA,cmap='Oranges',s=15)
    #py.scatter(epochA,vy_diff*1e3,c=magA,cmap='Blues',s=15,marker='^')
    py.legend(numpoints=1)
    #py.yscale('log')
    py.ylim([-0.2,0.2])
    py.xlabel('Epoch ('+labelA+')')
    py.ylabel(r'Vel '+labelA+' - '+labelB+' (mas/yr)')
    py.savefig(plotRoot+''+labelA+'_'+labelB+'_vDiff_epoch.png')
    py.clf()
    py.figure(figsize=(8,5))
    py.errorbar(epochA,ax_diff*1e3,yerr=axe_quad*1e3,fmt='.',label='X')
    py.errorbar(epochA,ay_diff*1e3,yerr=aye_quad*1e3,fmt='.',label='Y')
    #py.scatter(epochA,ax_diff*1e3,c=magA,cmap='Oranges',s=15,label='X')
    #py.scatter(epochA,ay_diff*1e3,c=magA,cmap='Blues',marker='^',s=15,label='Y')
    py.legend(numpoints=1)
    #py.yscale('log')
    py.ylim([-0.1,0.1])
    py.xlabel('Epoch ('+labelA+')')
    py.ylabel('Accel '+labelA+' - '+labelB+' (mas/yr$^2$)')
    py.savefig(plotRoot+''+labelA+'_'+labelB+'_aDiff_epoch.png')
    py.clf()
    py.figure(figsize=(8,5))
    py.errorbar(magA,vx_diff*1e3,yerr=vxe_quad*1e3,fmt='.',label='X')
    py.errorbar(magA,vy_diff*1e3,yerr=vye_quad*1e3,fmt='.',label='Y')
    #py.scatter(magA,vx_diff*1e3,c=epochA,cmap='Oranges',s=15,label='X')
    #py.scatter(magA,vy_diff*1e3,c=epochA,cmap='Blues',marker='^',s=15,label='Y')
    py.legend(numpoints=1)
    #py.yscale('log')
    py.ylim([-0.2,0.2])
    py.xlabel('Mag')
    py.ylabel('Vel '+labelA+' - '+labelB+' (mas/yr)')
    py.savefig(plotRoot+''+labelA+'_'+labelB+'_vDiff_mag.png')
    py.clf()
    py.figure(figsize=(8,5))
    py.errorbar(magA,ax_diff*1e3,yerr=axe_quad*1e3,fmt='.',label='X')
    py.errorbar(magA,ay_diff*1e3,yerr=aye_quad*1e3,fmt='.'
                
                ,label='Y')
    #py.scatter(magA,ax_diff*1e3,c=epochA,cmap='Oranges',s=15,label='X')
    #py.scatter(magA,ay_diff*1e3,c=epochA,cmap='Blues',marker='^',s=15,label='Y')
    py.legend(numpoints=1)
    #py.yscale('log')
    py.ylim([-0.1,0.1])
    py.xlabel('Mag')
    py.ylabel('Accel '+labelA+' - '+labelB+' (mas/yr$^2$)')
    py.savefig(plotRoot+''+labelA+'_'+labelB+'_aDiff_mag.png')
    py.clf()

    bins = np.arange(-20.0,20.0,0.5)
    py.hist(vxe_diff/vxe_quad,bins=bins,histtype='step',label='X')
    py.hist(vye_diff/vye_quad,bins=bins,histtype='step',label='Y')
    py.legend()
    py.xlabel('Vel Error Delta ( ('+labelA+' - '+labelB+') / ('+labelA+'$^2$ + '+labelB+'$^2$)$^{1/2}$ )')
    py.savefig(plotRoot+''+labelA+'_'+labelB+'_veDiff_hist.png')
    py.clf()
    py.hist(axe_diff/axe_quad,bins=bins,histtype='step',label='X')
    py.hist(aye_diff/aye_quad,bins=bins,histtype='step',label='Y')
    py.legend()
    py.xlabel('Accel Error Delta ( ('+labelA+' - '+labelB+') / ('+labelA+'$^2$ + '+labelB+'$^2$)$^{1/2}$ )')
    py.savefig(plotRoot+''+labelA+'_'+labelB+'_aeDiff_hist.png')
    
    py.clf()
    bins = np.arange(-5,5,0.1)
    py.hist(vx_diff/vxe_quad,bins=bins,histtype='step',label='X')
    py.hist(vy_diff/vye_quad,bins=bins,histtype='step',label='Y')
    py.legend()
    py.xlabel('Vel ( ('+labelA+' - '+labelB+') / ('+labelA+'$^2_e$ + '+labelB+'$^2_e$)$^{1/2}$ )')
    py.savefig(plotRoot+''+labelA+'_'+labelB+'_vDiff_hist.png')
    bins = np.arange(-5,5,0.2)
    py.clf()
    py.hist(ax_diff/axe_quad,bins=bins,histtype='step',label='X')
    py.hist(ay_diff/aye_quad,bins=bins,histtype='step',label='Y')
    py.legend()
    py.xlabel('Accel ( ('+labelA+' - '+labelB+') / ('+labelA+'$^2_e$ + '+labelB+'$^2_e$)$^{1/2}$ )')
    py.savefig(plotRoot+''+labelA+'_'+labelB+'_aDiff_hist.png')
    py.clf()

    py.plot([20,55],[20,55],'k')
    py.plot(epochB,epochA,'.')
    py.xlabel('Epoch ('+labelB+')')
    py.ylabel('Epoch ('+labelA+')')
    py.savefig(plotRoot+''+labelA+'_'+labelB+'_epoch.png')
    py.clf()
    py.plot([5e-6,1e-2],[5e-6,1e-2],'k')
    py.plot(vxe_B,vxe_A,'.',label='X')
    py.plot(vye_B,vye_A,'.',label='Y')
    py.legend()
    py.xscale('log')
    py.yscale('log')
    py.xlabel('Vel Error '+labelB+' (as/yr)')
    py.ylabel('Vel Error '+labelA+' (as/yr)')
    py.savefig(plotRoot+''+labelA+'_'+labelB+'_vel_error.png')
    py.clf()
    py.plot([1.5e-6,4e-3],[1.5e-6,4e-3],'k')
    py.plot(axe_B,axe_A,'.',label='X')
    py.plot(aye_B,aye_A,'.',label='Y')
    py.legend()
    py.xscale('log')
    py.yscale('log')
    py.xlabel('Accel Error '+labelB+' (as/yr$^2$)')
    py.ylabel('Accel Error '+labelA+' (as/yr$^2$)')
    py.savefig(plotRoot+''+labelA+'_'+labelB+'_accel_error.png')
    py.clf()

    py.errorbar(vx_B,vx_A,xerr=vxe_B,yerr=vxe_A,fmt='.',label='X')
    py.errorbar(vy_B,vy_A,xerr=vye_B,yerr=vye_A,fmt='.',label='Y')
    py.legend()
    py.xscale('log')
    py.yscale('log')
    py.xlabel('Vel '+labelB+' (as/yr)')
    py.ylabel('Vel '+labelA+' (as/yr)')
    py.savefig(plotRoot+''+labelA+'_'+labelB+'_vel.png')
    py.clf()
    py.errorbar(ax_B,ax_A,xerr=axe_B,yerr=axe_A,fmt='.',label='X')
    py.errorbar(ay_B,ay_A,xerr=aye_B,yerr=aye_A,fmt='.',label='Y')
    py.legend()
    py.xscale('log')
    py.yscale('log')
    py.xlabel('Accel '+labelB+' (as/yr$^2$)')
    py.ylabel('Accel '+labelA+' (as/yr$^2$)')
    py.savefig(plotRoot+''+labelA+'_'+labelB+'_accel.png')
    py.clf()

    velQ = py.quiver(x_A,y_A,vx_diff,vy_diff)
    py.quiverkey(velQ,1.5,1.7,5e-4,label=r'0.5 mas/yr',color='b',coordinates='data')
    py.xlabel('X (arcsec)')
    py.ylabel('Y (arcsec)')
    py.title('Velocity '+labelA+' - '+labelB)
    py.savefig(plotRoot+''+labelA+'_'+labelB+'_quiver_vel.png')
    py.clf()
    accelQ = py.quiver(x_A,y_A,ax_diff,ay_diff)
    py.quiverkey(accelQ,1.5,1.7,2e-4,label=r'0.2 mas/yr',color='b',coordinates='data')
    py.xlabel('X (arcsec)')
    py.ylabel('Y (arcsec)')
    py.title('Acceleration '+labelA+' - '+labelB)
    py.savefig(plotRoot+''+labelA+'_'+labelB+'_quiver_accel.png')
    py.clf()
    py.close('all')




def star_compAlign(starName, alignA='/g/ghez/align/schappell_14_06_18/',polyA='polyfit_nz/',pointsA='points_nz/',labelA='Old Holo',alignB='/g/ghez/align/18_02_07_newSpeckle/',polyB='polyfit_4_trim/',pointsB='points_4_trim/',labelB='New Holo'):

    pointsA = np.loadtxt(alignA+pointsA+starName+'.points')
    pointsB = np.loadtxt(alignB+pointsB+starName+'.points')
    
    py.clf()
    py.errorbar(pointsA[:,1],pointsA[:,2],xerr=pointsA[:,3],yerr=pointsA[:,4],fmt='.',label=labelA)
    py.errorbar(pointsB[:,1],pointsB[:,2],xerr=pointsB[:,3],yerr=pointsB[:,4],fmt='.',label=labelB)
    py.legend(numpoints=1)
    py.xlabel('X (as)')
    py.ylabel('Y (as)')
    py.title(starName)
    py.savefig(plotRoot+''+str(starName)+'_compareAlign_XY.png')
    py.clf()
    py.errorbar(pointsA[:,0],pointsA[:,2],yerr=pointsA[:,4],fmt='.',label=labelA)
    py.errorbar(pointsB[:,0],pointsB[:,2],yerr=pointsB[:,4],fmt='.',label=labelB)
    py.legend(numpoints=1)
    py.xlabel('Time (years)')
    py.ylabel('Y (as)')
    py.title(starName)
    py.savefig(plotRoot+''+str(starName)+'_compareAlign_tY.png')
    py.clf()
    py.errorbar(pointsA[:,0],pointsA[:,1],yerr=pointsA[:,3],fmt='.',label=labelA)
    py.errorbar(pointsB[:,0],pointsB[:,1],yerr=pointsB[:,3],fmt='.',label=labelB)
    py.legend(numpoints=1)
    py.xlabel('Time (years)')
    py.ylabel('X (as)')
    py.title(starName)
    py.savefig(plotRoot+''+str(starName)+'_compareAlign_tX.png')
    py.clf()
    py.close('all')




def print_ar_at(x0,x0e,y0,y0e,xa,xae,ya,yae):

    R = math.sqrt(x0**2 + y0**2)
    ar = ((xa*x0) + (ya*y0)) / R
    at = ((xa*y0) - (ya*x0)) / R
    are =  (xae*x0/R)**2 + (yae*y0/R)**2
    are += (y0*x0e*at/R**2)**2 + (x0*y0e*at/R**2)**2
    are =  math.sqrt(are)
    ate =  (xae*y0/R)**2 + (yae*x0/R)**2
    ate += (y0*x0e*ar/R**2)**2 + (x0*y0e*ar/R**2)**2
    ate =  math.sqrt(ate)

    print 'radial accel: '+str(ar)+' +/- '+str(are)+' as/yr'
    print 'significance: '+str(ar/are)
    print 'tangential accel: '+str(at)+' +/- '+str(ate)+' as/yr'
    print 'significance: '+str(at/ate)

