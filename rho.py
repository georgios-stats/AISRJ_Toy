
# ==============================================================================
#  
# Copyrigtht 2012 Georgios Karagiannis
# 
# This file is part of AISRJ_Toy.
# 
# AISRJ_Toy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation version 2 of the License.
# 
# AISRJ_Toy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
#
# ==============================================================================


# ==============================================================================
# RHO DISTRIBUTION FUNCTIONS
# ==============================================================================

# References :

# Karagiannis, G., & Andrieu, C. (2013). 
# Annealed importance sampling reversible jump MCMC algorithms. 
# Journal of Computational and Graphical Statistics, 22(3), 623-648.

# Georgios Karagiannis
# School of Mathematics, University of Bristol
# University Walk, Bristol, BS8 1TW, UK
# Email : Georgios.Karagiannis@pnnl.gov
# Email (current): georgios-stats@gmail.com

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



