
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
# along with AISRJ_Toy.  If not, see <http://www.gnu.org/licenses/>.
#
# ==============================================================================


# ==============================================================================
# NORMAL : ANDRIEU AND ROBERTS 2009
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






