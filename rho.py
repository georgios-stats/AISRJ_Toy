# ==============================================================================
# RHO DISTRIBUTION FUNCTIONS
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
import target

#     LOG DENSITY OF THE RHO DISTRIBUTION

def LogPdf( k1, k2, x, maux, saux, gt ) :
      
      logPr1 = target.LogPdf( k1, x[:k1] )
      
      if k1==1 and k2==2 :
            logProp12 = -0.5*log(2*pi) -0.5*log(saux**2) -0.5*(x[1]-maux)**2/saux**2
      else :
            logProp12 = 0.0
      
      logPr2 = target.LogPdf( k2, x[:k2] )
      
      if k1==2 and k2==1 :
            logProp21 = -0.5*log(2*pi) -0.5*log(saux**2) -0.5*(x[1]-maux)**2/saux**2
      else :
            logProp21 = 0.0
      
      logRho = (1-gt)*logPr1 +(1-gt)*logProp12 +gt*logPr2 +gt*logProp21
      
      return logRho

#     GRADIENT OF LOG DENSITY OF THE RHO DISTRIBUTION

def GradLogPdf( k1, k2, x, maux, saux, gt ) :
      
      grd1ex = target.GradLogPdf( k1, x[:k1] )
      
      if k1==1 and k2==2 :
            grd1ex = append( grd1ex, -(x[1]-maux)/saux**2 )
      
      grd2ex = target.GradLogPdf( k2, x[:k2] )
      
      if k1==2 and k2==1 :
            grd2ex = append( grd2ex, -(x[1]-maux)/saux**2 )
      
      grd = (1-gt)*grd1ex +gt*grd2ex
      
      return grd



