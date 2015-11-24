# ==============================================================================
# NORMAL : ANDRIEU AND ROBERTS 2009
# ==============================================================================

# Contact details :

# Georgios Karagiannis © 2012
# School of Mathematics, University of Bristol
# University Walk, Bristol, BS8 1TW, UK
# Email (current): Georgios.Karagiannis@pnnl.gov

# Christophe Andrieu
# School of Mathematics, University of Bristol
# University Walk, Bristol, BS8 1TW, UK
# Email: C.Andrieu@bristol.ac.uk

#     IMPORT MODULES

from numpy import *

#     DENSITY OF THE TARGET DISTRIBUTION ---------------------------------------

def Pdf( k, x ) :
      
      if k == 1 :
            Pr = (1./4.) \
                        *1/sqrt(2*pi) *exp(-0.5*x[0]**2)
      elif k == 2 :
            r = -0.9
            Pr = (3./4.) \
                        *1/(2*pi*sqrt(1-r**2)) \
                        *exp( -0.5/(1-r**2)*(x[0]**2+x[1]**2-2*r*x[0]*x[1]) )
      
      return Pr

#     LOG DENSITY OF THE TARGET DISTRIBUTION -----------------------------------

def LogPdf( k, x ) : # LOG DENSITY OF THE TARGET DISTRIBUTION
      
      if k == 1 :
            logPr = log(1./4.) \
                        -0.5*log(2*pi) -0.5*x[0]**2
      elif k == 2 :
            r = -0.9
            logPr = log(3./4.) \
                        -log(2*pi) \
                        -0.5*log(1-r**2) \
                        -0.5/(1-r**2)*(x[0]**2+x[1]**2-2*r*x[0]*x[1])
      
      return logPr

#     GRADIENT OF THE LOG DENSITY OF THE TARGET DISTRIBUTION -------------------

def GradLogPdf( k, x ) :
      
      if k == 1 :
            grd = array([-x[0]])
      elif k == 2 :
            r = -0.9
            grd = array([ \
                        -x[0]/(1-r**2) +r/(1-r**2)*x[1] \
                        ,-x[1]/(1-r**2) +r/(1-r**2)*x[0] \
                        ])
      
      return grd






