# ==============================================================================
#     ANNEALED IMPORTANCE SAMPLING REVERSIBLE JUMP
# ==============================================================================

# Contact details :

# Georgios Karagiannis
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
import rho

# ==============================================================================
#     ANNEALING UPDATE SWEEP AT TIME T
# ==============================================================================

def AnnealingUpdateSweep( k1, k2, x, maux, saux, scl, gt) :
      
#      DRAW A PROPOSAL ---------------------------------------------------------
      
      grd = rho.GradLogPdf( k1, k2, x[:2], maux, saux, gt ) 
      xN = random.normal(size=2)
      xN = x +0.5*(scl**2)*grd +scl*xN
      
#     COMPUTE THE LOG RHO DENSITY ----------------------------------------------
      
      logRho = rho.LogPdf( k1, k2, x[:2], maux, saux, gt )
      
#     COMPUTE THE LOG PROPOSAL DENSITY NEW -------------------------------------
      
      logPropN = sum( \
                        -0.5*log(2*pi) \
                        -0.5*log(scl**2) \
                        -0.5*( xN[:2] -x[:2] -0.5*(scl**2)*grd )**2 /scl**2 \
                  )
      
#     COMPUTE THE LOG RHO DENSITY NEW ------------------------------------------
      
      logRhoN = rho.LogPdf( k1, k2, xN[:2], maux, saux, gt )
      
#     COMPUTE THE LOG PROPOSAL DENSITY -----------------------------------------
      
      grd = rho.GradLogPdf( k1, k2, xN[:2], maux, saux, gt ) 
      
      logProp = sum( \
                        -0.5*log(2*pi) \
                        -0.5*log(scl**2) \
                        -0.5*( x[:2] -xN[:2] -0.5*(scl**2)*grd )**2 /scl**2 \
                  )
      
#      COMPUTE THE ACCEPTANCE PROBABILITY --------------------------------------
      
      accpr = exp( min( 0, logRhoN +logProp -logRho -logPropN ) )
      
#      ACCEPT / REJECT ---------------------------------------------------------
      
      u = random.uniform()
      if accpr>u :
            return(xN,accpr)
      else :
            return(x,accpr)

# ==============================================================================
#     LOG IMPORTANCE WEIGHT AT TIME T
# ==============================================================================

def LogImportanceWeight( k1, k2, x, maux, saux ) :
      
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
      
      logIW = logPr2 +logProp21 -logPr1 -logProp12 
      
      return logIW

# ==============================================================================
# THE PROGRAM
# ==============================================================================

if __name__ == "__main__" :
      
      import sys
      
#     PARAMETERS ---------------------------------------------------------------
      
#     marginal move proposal      
      Qrj = array([[0,1],[1,0]],dtype=float64)
      
#     mean and std of dimentional matching proposal
      maux = 3.0
      saux = 1.0
      
#     scale parametero of AIS proposal
      scl = 1.0
      
#     number of iterations and burni-in of MCMC     
      iterations = 10**5
      nburnin = 0
      
#     number of intermediate distributions
      Tau = 1
      
      lab = str(Tau)
      
      k0 = 1
      x0 = zeros(2,dtype=float64)
      
      lenin = len(sys.argv)
      for ia in range(0,lenin) :
            word = sys.argv[ia]
            if sys.argv[ia][0:3] == '-k=':
                  k0 = int(sys.argv[ia][3:])
            if sys.argv[ia][0:3] == '-x=':
                  x0 = eval('array('+sys.argv[ia][3:]+',dtype=float64)')
            elif sys.argv[ia][0:8] == '-niters=':
                  iterations = int(sys.argv[ia][8:])
            elif sys.argv[ia][0:9] == '-nburnin=':
                  nburnin = int(sys.argv[ia][9:])
            elif sys.argv[ia][0:5] == '-scl=':
                  scl = float(sys.argv[ia][5:])
            elif sys.argv[ia][0:5] == '-Tau=':
                  Tau = int(sys.argv[ia][5:])
                  lab = sys.argv[ia][5:]
            elif sys.argv[ia][0:5] == '-Qrj=':
                  Qrj = eval('array('+sys.argv[ia][5:]+',dtype=float64)')
            elif sys.argv[ia][0:6] == '-maux=':
                  maux = float(sys.argv[ia][6:])
            elif sys.argv[ia][0:6] == '-saux=':
                  saux = float(sys.argv[ia][6:])
      
#     PRINT THE PARAMETERS -----------------------------------------------------
      
      print('')
      print('PARAMETERS')
      print('----------')
      print('k0               : ' + str(k0))
      print('x0               : ' + str(x0))
      print('maux             : ' + str(maux))
      print('saux             : ' + str(saux))
      print('iterations       : ' + str(iterations))
      print('nburnin          : ' + str(nburnin))
      print('scl              : ' + str(scl))
      print('Qrj              : ' + str(Qrj))
      print('Tau              : ' + str(Tau))
      print('lab              : ' + str(lab))
      
#     DECLARE VECTORS AND MATRICES ---------------------------------------------
      
      ksample = zeros( iterations+1, dtype=int16 )
      xsample = zeros( (iterations+1,2), dtype=float64 )
      movesample = zeros( (iterations+1,2), dtype=int16 )
      accprsample = zeros( iterations+1, dtype=float64 )
      logAIWsample = zeros( iterations+1, dtype=float64 )
      
      x = zeros( (Tau,2), dtype=float64 )
      
#     STORE THE SEEDS ----------------------------------------------------------
      
      ksample[0] = k0
      xsample[0] = x0
      movesample[0] = zeros( 2,dtype=int16 )
      accprsample[0] = 0.0
      logAIWsample[0] = 0.0
      
#     INITIALIZE THE COUNTERS --------------------------------------------------
      
      ExpAccPrBirthDeath = 0.0
      nBirth = 0
      ExpAccPrBirth = 0.0
      nDeath = 0
      ExpAccPrDeath = 0.0
      
      ExpAccPrXais = 0.0
      
#     INITIALIZE THE SWEEP -----------------------------------------------------
      
      xais = zeros( 2, dtype=float64 )
      
      k = k0
      x = x0
      
      print('')
      print(' *** START THE ITERATIONS : ***')
      
      for itr in range(-nburnin, iterations+1) :
            
            if (itr+nburnin) % ((iterations+nburnin)/ 10.) == 0 :
                  print( \
                     repr((iterations+nburnin-itr)/((iterations+nburnin)/10.)).rjust(3)[:6] \
                        )
            
#           LOAD THE VARIABLES
            
            k1 = k
            xais[0:k1] = x[0:k1]
            
#           DRAW A MODEL LABEL -------------------------------------------------
            
            #k1 = k
            u = random.uniform()
            k2 = 1
            sumQrj = 0.0
            while True :
                  sumQrj = sumQrj +Qrj[k1-1,k2-1]
                  if sumQrj>u :
                        break
                  k2 = k2+1
            
#           DIMENSION MATCHING STEP --------------------------------------------
            
            #xais[0:k1] = x[0:k1]
            if k1==1 and k2==2 :
                  xais[1] = maux +saux*random.normal()
            
#           ANEALING IMPORTANCE WEIGHT
            
            logAIW = LogImportanceWeight( k1, k2, xais[:2], maux, saux) / Tau
            
#           ANNEALING ----------------------------------------------------------
            
            for t in range( 1, Tau ) :
                  
#                 ANNEALING TIME
                  
                  gt = float(t)/Tau
                  
#                 ANNEALING UPDATE
                  
                  ( xais, accpr ) = AnnealingUpdateSweep(k1,k2,xais[:2],maux,saux,scl,gt)
                  
#                 ANEALING IMPORTANCE WEIGHT
                  
                  logAIW = logAIW +LogImportanceWeight(k1,k2,xais[:2],maux,saux)/Tau
                  
#                 RECORD
                  
                  if itr>0 :
                        ExpAccPrXais = ExpAccPrXais+accpr
            
#           RJ ACCEPT REJECT ---------------------------------------------------
            
            accprrj = exp( \
                              min( \
                                    0.0, \
                                    log( Qrj[k2-1,k1-1] ) \
                                          -log( Qrj[k1-1,k2-1] ) \
                                                +logAIW \
                              ) \
                        )
            
            u = random.uniform()
            if accprrj>u :
                  k = k2
                  x[0:k] = xais[0:k]
            
#           RECORD -------------------------------------------------------------
            
            if itr>0 :
                  ExpAccPrBirthDeath = ExpAccPrBirthDeath +accprrj
                  if k1==1 and k2==2 :
                        nBirth = nBirth+1
                        ExpAccPrBirth = ExpAccPrBirth +accprrj
                  elif k1==2 and k2==1 :
                        nDeath = nDeath+1
                        ExpAccPrDeath = ExpAccPrDeath +accprrj
            
#           SAVE THE SAMPLE ----------------------------------------------------
            
            if itr>0 :
                  ksample[itr] = k
                  xsample[itr,0:k] = x[0:k]
                  movesample[itr] = array([k1,k2],dtype=int16)
                  accprsample[itr] = accprrj
                  logAIWsample[itr] = logAIW
      
#     REFINE THE COUNTERS ------------------------------------------------------
      
      ExpAccPrBirthDeath = ExpAccPrBirthDeath/(nBirth+nDeath)
      ExpAccPrBirth = ExpAccPrBirth/nBirth
      ExpAccPrDeath = ExpAccPrDeath/nDeath
      if Tau>1 :
            ExpAccPrXais = ExpAccPrXais/((Tau-1)*(nBirth+nDeath))
      
#     PRINT THE RESULTS OF THE COUNTERS ----------------------------------------
      
      print('')
      print('COUNTERS' )
      print('--------' )
      print('ExpAccPrBirthDeath     : ' + str(ExpAccPrBirthDeath) )
      print('ExpAccPrBirth          : ' + str(ExpAccPrBirth) )
      print('ExpAccPrDeath          : ' + str(ExpAccPrDeath) )
      print('--------' )
      print('ExpAccPrXais           : ' + str(ExpAccPrXais) )
      print('--------' )
      print('mean(k)                : ' + str(mean(ksample)) )
      print('--------')
      print('Pr(k=1|RJ)             : ' + str(mean(ksample==1)) )
      print('Pr(k=2|RJ)             : ' + str(mean(ksample==2)) )
      print('--------' )
      print('Pr(k=2)/Pr(k=1)|IS     : ' + \
                        str( \
                        mean( \
                              exp( \
                                    logAIWsample[ all( movesample==[1,2],1 ) ] \
                                    ) \
                              ) \
                           ) \
            )
      print('Pr(k=1)/Pr(k=2)|IS     : ' + \
                        str( \
                        mean( \
                              exp( \
                                    logAIWsample[ all( movesample==[2,1],1 ) ] \
                                    ) \
                              ) \
                           ) \
            )
      
#     SAVE THE SAMPLE ----------------------------------------------------------
      
      savetxt('./results/k.T'+lab,ksample[1:iterations+1], fmt='%1i')
      savetxt('./results/x.T'+lab,xsample[1:iterations+1])
      savetxt('./results/move.T'+lab,movesample[1:iterations+1], fmt='%1i %1i')
      savetxt('./results/accpr.T'+lab,accprsample[1:iterations+1])
      savetxt('./results/logAIW.T'+lab,logAIWsample[1:iterations+1])

