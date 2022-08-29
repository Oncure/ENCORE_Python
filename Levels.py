import math
import numpy as np
from scipy.integrate import quad as integrate
from scipy.optimize import root_scalar
import scipy.special as spf

from Encore import Encore

# ==================================================================================
# Merger:
# ==================================================================================
class Merger:
    """
    ...

    Variables:
    Freq  :: 64b float [N]   | ...
    w     :: 64b float [N]   | ...
    a     :: 64b float [N]   | ...
    N     ::  8b   int       | ...

    Z     :: 64b float [N,N] | ...
    ZB    :: 64b float [N]   | ...
    xMax  :: 64b float       | ...
    xMaxB :: 64b float       | ...
    """

    pid4       = math.pi/4
    sqrtpid4   = math.sqrt(pid4)

    deg        = 150
    multiplier = 1e10 # the upper bound on the integral and xMax search

    def __init__(self, Freq, LevelSpacingFunc:str='Wigner', BrodyParam = None, err=1e-9):
        """
        ...
        """

        self.Freq = Freq.reshape(1,-1)
        self.N    = self.Freq.shape[1]

        self.LevelSpacingDist = LevelSpacingFunc
        if LevelSpacingFunc not in ('Wigner','Brody'):
            raise NotImplementedError(f'Level-spacing function, "{LevelSpacingFunc}", is not implemented yet.')

        if LevelSpacingFunc == 'Brody':
            if BrodyParam == None:
                self.w = np.array([1.0]*self.N, dtype='f8').reshape(1,-1)
            else:
                self.w = BrodyParam

            w1i = 1/(self.w+1)
            self.a = (w1i*spf.gamma(w1i)*self.Freq)**(self.w+1)

        xMaxLimit  = self.__xMaxLimit(err) # xMax must be bounded by the error for spin-group alone

        if self.N != 1:
            self.Z, self.ZB = self.__FindZ(xMaxLimit)

        self.xMax, self.xMaxB = self.__FindxMax(err, xMaxLimit)

    @property
    def FreqTot(self):
        """
        Gives the total frequency for the combined group.
        """
        return np.sum(self.Freq)
    @property
    def MeanLS(self):
        """
        Gives the combined mean level-spacing for the group.
        """
        return 1.0 / np.sum(self.Freq)

    def FindLevelSpacings(self, E, EB, Prior, verbose=False):
        """
        ...
        """

        L  = E.shape[0]
        
        # Calculating the approximation thresholds:
        iMax = np.full((L+2,2), -1, dtype='i4')
        for j in range(L):
            if E[j] - EB[0] >= self.xMaxB:
                iMax[0,0]    = j
                iMax[:j+1,1] = 0
                break
        for i in range(L-1):
            for j in range(iMax[i,0]+1,L):
                if E[j] - E[i] >= self.xMax:
                    iMax[i+1,0] = j
                    iMax[iMax[i-1,0]:j+1,1] = i+1
                    break
            else:
                iMax[i:,0] = L+1
                iMax[iMax[i-1,0]:,1] = i+1
                break
        for j in range(L-1,-1,-1):
            if EB[1] - E[j] >= self.xMaxB:
                iMax[-1,1] = j
                iMax[j:,0] = L+1
                break
        
        # Level-spacing calculation:
        LS = np.zeros((L+2,L+2),'f8')
        if self.N == 1:
            for i in range(L-1):
                X = E[i+1:iMax[i+1,0]-1] - E[i]
                LS[i+1,i+2:iMax[i+1,0]] = self.LevelSpacingFunc(X)
            LS[0,1:-1]  = self.LevelSpacingFunc(E - EB[0], True)
            LS[1:-1,-1] = self.LevelSpacingFunc(EB[1] - E, True)
        else:
            for i in range(L-1):
                X = E[i+1:iMax[i+1,0]-1] - E[i]
                PriorL = np.tile(Prior[i,:], (iMax[i+1,0]-i-2, 1))
                PriorR = Prior[i+1:iMax[i+1,0]-1,:]
                LS[i+1,i+2:iMax[i+1,0]] = self.__LSMerge(X, PriorL, PriorR)
            LS[0,1:-1]  = self.__LSMergeBoundaries(E - EB[0], Prior)
            LS[1:-1,-1] = self.__LSMergeBoundaries(EB[1] - E, Prior)

        if verbose: print('Finished level-spacing calculations')
        return LS, iMax

    def __LSMerge(s, X, PriorL, PriorR):
        """
        ...
        """

        c, r1, r2 = s.__FindParts(X.reshape(-1,1))
        d = r2 * (r1 - r2)

        Norm = np.matmul(PriorL.reshape(-1,1,s.N), np.matmul(s.Z.reshape(1,s.N,s.N), PriorR.reshape(-1,s.N,1))).reshape(-1,)
        return ((c / Norm) * ( \
                    np.sum(PriorL*r2, axis=1) * \
                    np.sum(PriorR*r2, axis=1) + \
                    np.sum(PriorL*PriorR*d, axis=1)
                )).reshape(X.shape)

    def __LSMergeBoundaries(s, X, Prior):
        """
        ...
        """

        c, r2 = s.__FindParts(X.reshape(-1,1))[::2]
        Norm = np.sum(s.ZB * Prior, axis=1)
        return ((c / Norm) * \
                    np.sum(Prior*r2, axis=1) \
                ).reshape(X.shape)

    def LevelSpacingFunc(s, X, Boundary=False):
        if s.LevelSpacingDist == 'Wigner':
            coef = (s.sqrtpid4 * s.Freq)**2
            if Boundary:
                return np.exp(-coef * X**2)
            else:
                return 2 * coef * X * np.exp(-coef * X**2)
        elif s.LevelSpacingDist == 'Brody':
            aXw = s.a * X**s.w
            if Boundary:
                return np.exp(-aXw*X)
            else:
                return (s.w+1) * np.exp(-aXw*X)
    
    def __FindParts(s, X):
        """
        ...
        """
        
        if s.LevelSpacingDist == 'Wigner':
            fX = s.sqrtpid4 * s.Freq * X
            R1 = 2 * s.sqrtpid4 * s.Freq * fX
            R2 = s.Freq / spf.erfcx(fX)     # "erfcx(x)" is "exp(x^2)*erfc(x)"
            F2 = np.exp(-fX*fX) / R2        # Getting the "erfc(x)" from "erfcx(x)"
        elif s.LevelSpacingDist == 'Brody':
            w1i = 1.0 / (s.w+1)
            aXw = s.a*X**s.w
            aXw1 = aXw*X
            R1 = (s.w+1)*aXw
            F2 = (w1i*(s.a)**w1i)*spf.gammaincc(w1i,aXw1)
            R2 = np.exp(-aXw1) / F2
        C  = np.prod(F2, axis=1)
        return C, R1, R2
   
    def __FindZ(s, xMaxLimit):
        """
        ...
        """

        def OffDiag(x, i:int, j:int):
            C, R2 =  s.__FindParts(x)[::2]
            return C[0] * R2[0,i] * R2[0,j]
        def MainDiag(x, i:int):
            C, R1, R2 = s.__FindParts(x)
            return C[0] * R1[0,i] * R2[0,i]
        def Boundaries(x, i:int):
            C, R2 = s.__FindParts(x)[::2]
            return C[0] * R2[0,i]

        # Level Spacing Normalization Matrix:
        Z = np.zeros((s.N,s.N), dtype='f8')
        for i in range(s.N):
            for j in range(i):
                Z[i,j] = integrate(lambda _x: OffDiag(_x,i,j), a=0.0, b=min(*xMaxLimit[0,[i,j]]))[0]
                Z[j,i] = Z[i,j]
            Z[i,i] = integrate(lambda _x: MainDiag(_x,i), a=0.0, b=xMaxLimit[0,i])[0]
        
        # Level Spacing Normalization at Boundaries:
        ZB = np.zeros((1,s.N), dtype='f8')
        ZB[0,:] = [integrate(lambda _x: Boundaries(_x,i), a=0.0, b=xMaxLimit[0,i])[0] for i in range(s.N)]
        return Z, ZB
    
    def __FindxMax(self, err:float, xMaxLimit):
        """
        ...
        """

        mthd = 'brentq'

        bounds = [self.MeanLS, max(*xMaxLimit)]
        if self.N == 1:
            xMax  = root_scalar(lambda x: self.LevelSpacingFunc(x) - err     , method=mthd, bracket=bounds).root
            xMaxB = root_scalar(lambda x: self.LevelSpacingFunc(x,True) - err, method=mthd, bracket=bounds).root
        else:
            # Bounding the equation from above:
            def UpperBoundLS(x):
                c, r1, r2 = self.__FindParts(x)

                # Minimum normalization possible. This is the lower bound on the denominator with regards to the priors:
                Norm_LB = np.amin(self.Z)

                # Finding maximum numerator for the upper bound on the numerator:
                c1 = np.amax(r1*r2, axis=1)
                c2 = np.amax(r2**2)
                return (c / Norm_LB) * max(c1, c2)

            def UpperBoundLSBoundary(x):
                c, r2 = self.__FindParts(x)[:3:2]

                # Minimum normalization possible for lower bound:
                Norm_LB = np.amin(self.ZB)
                return (c / Norm_LB) * np.max(r2, axis=1) # Bounded above by bounding the priors from above
            xMax  = root_scalar(lambda x: UpperBoundLS(x)-err        , method=mthd, bracket=bounds).root
            xMaxB = root_scalar(lambda x: UpperBoundLSBoundary(x)-err, method=mthd, bracket=bounds).root
        return xMax, xMaxB

    def __xMaxLimit(s, err:float):
        if s.LevelSpacingDist == 'Wigner':
            xMax = np.sqrt(-np.log(err)) / (s.sqrtpid4 * s.Freq)
        elif s.LevelSpacingDist == 'Brody':
            xMax = (-np.log(err) / s.a) ** (1.0 / (s.w+1))
        return 2*xMax

# =================================================================================================
#     WigBayes Partition Master:
# =================================================================================================

class RunMaster:

    def __init__(self, E, EB, Prior, TPPrior, Freq, LevelSpacingDist:str='Wigner', err:float=1e-9, BrodyParam=None):
        self.E       = E
        self.EB      = EB
        self.Prior   = Prior
        self.TPPrior = TPPrior
        self.Freq    = Freq

        self.LevelSpacingDist = LevelSpacingDist
        self.err     = err
        self.w       = BrodyParam

        self.L = E.shape[0]
        self.N = Freq.shape[1] - 1

    def MergePartitioner(s, Partitions:list):
        """
        ...
        """

        n = len(Partitions)

        # Merged level-spacing calculation:
        LS   = np.zeros((s.L+2, s.L+2, n), 'f8')
        iMax = np.zeros((s.L+2,     2, n), 'i4')
        for g, group in enumerate(Partitions):
            if s.w == None:
                merge = Merger(s.Freq[:,group], s.LevelSpacingDist, BrodyParam=None, err=s.err)
            else:
                merge = Merger(s.Freq[:,group], s.LevelSpacingDist, BrodyParam=s.w[:,group], err=s.err)
            LS[:,:,g], iMax[:,:,g] = merge.FindLevelSpacings(s.E, s.EB, s.Prior[:,group])

        # Merged prior calculation:
        PriorMerge = np.zeros((s.L, n+1), 'f8')
        for g, group in enumerate(Partitions):
            if hasattr(group, '__iter__'):
                PriorMerge[:,g] = np.sum(s.Prior[:,group], axis=1)
            else:
                PriorMerge[:,g] = s.Prior[:,group]
        PriorMerge[:,-1] = s.Prior[:,-1]

        return LS, iMax, PriorMerge

    def WigBayesPartitionMaster(s, log_total_probability=False, verbose=False):
        """
        ...
        """
        
        # Only 2 spin-groups (merge not needed):
        if s.N == 2:
            if verbose: print(f'Preparing level-spacings')
            LS_g, iMax_g, Prior_g = s.MergePartitioner([[0], [1]])
            if verbose: print(f'Finished level-spacing calculations')
            ENCORE = Encore(Prior_g, LS_g, iMax_g)
            if verbose: print(f'Finished CP calculation')
            Probs = ENCORE.WigBayes()
            if verbose: print(f'Finished WigBayes calculation')
            if log_total_probability:
                ltp = ENCORE.LogTotProb(s.EB, s.Freq[0,-1], s.TPPrior)
                return Probs, ltp
            else:
                return Probs

        # More than 2 spin-groups (Merge needed):
        else:
            Probs = np.zeros((s.L,3,s.N),dtype='f8')
            if log_total_probability:
                ltp = np.zeros(s.N, dtype='f8')
            for g in range(s.N):
                partition = [[g], [g_ for g_ in range(s.N) if g_ != g]]
                if verbose: print(f'Preparing for Merge group, {g}')
                LS_g, iMax_g, Prior_g = s.MergePartitioner(partition)
                if verbose: print(f'Finished spin-group {g} level-spacing calculation')
                ENCORE = Encore(Prior_g, LS_g, iMax_g)
                if verbose: print(f'Finished spin-group {g} CP calculation')
                Probs[:,:,g] = ENCORE.WigBayes()
                if verbose: print(f'Finished spin-group {g} WigBayes calculation')

                if log_total_probability:
                    ltp[g] = ENCORE.LogTotProb(s.EB, s.Freq[0,-1])

            # Combine probabilities for each merge case:
            NewProbs = s.prob_combinator(Probs)
            if log_total_probability:
                if verbose: print(f'Preparing for Merge group, 1000!!!')
                LS1, iMax1, Prior1 = s.MergePartitioner([tuple(range(s.N))])
                if verbose: print(f'Finished spin-group 1000 level-spacing calculation')
                ENCORE = Encore(Prior1, LS1, iMax1)
                if verbose: print(f'Finished spin-group {g} CP calculation')
                base_LogProb = ENCORE.LogTotProb(s.EB, s.Freq[0,-1])
                NewLTP = s.ltp_combinator(ltp, base_LogProb)
                if verbose: print('Finished!')
                return NewProbs, NewLTP
            else:
                if verbose: print('Finished!')
                return NewProbs
           
    def prob_combinator(self, Probs):
        """
        ...
        """

        S, NS, F = range(3)

        CombProbs = np.zeros((self.L,self.N+1), dtype='f8')
        for g in range(self.N):
            CombProbs[:,g] = Probs[:,S,g]
        CombProbs[:,-1] = np.prod(Probs[:,S,:], axis=1) * self.Prior[:,-1] ** (1-self.N)
        CombProbs[self.Prior[:,-1]==0.0,  -1] = 0.0
        CombProbs /= np.sum(CombProbs, axis=1).reshape((-1,1))

        # =========================================================

        # CombProbs = np.zeros((self.L,self.N+1), dtype='f8')
        # notPrior  = np.zeros((self.L,self.N)  , dtype='f8')

        # CombProbs[:,-1] = self.Prior[:,-1] ** (1-self.N)
        # for g in range(self.N):
        #     notPrior[:,g]  = np.sum(self.Prior[:,:g]) + np.sum(self.Prior[:,g+1:-1])
        #     CombProbs[:,g]   = Probs[:,S,g] * notPrior[:,g] / Probs[:,NS,g]
        #     CombProbs[:,-1] *= Probs[:,F,g] * notPrior[:,g] / Probs[:,NS,g]

        # # Corrections:
        # CombProbs[self.Prior[:,-1]==0.0,  -1] = 0.0

        # # Normalization:
        # CombProbs /= np.sum(CombProbs, axis=1).reshape((-1,1))

        # # Corrections:
        # CombProbs[self.Prior[:,-1]==1.0, :-1] = 0.0
        # CombProbs[self.Prior[:,-1]==1.0,  -1] = 1.0
        # for g in range(self.N):
        #     CombProbs[self.Prior[:,g]==1.0, :] = 0.0
        #     CombProbs[self.Prior[:,g]==1.0, g] = 1.0
        #     CombProbs[Probs[:,S,g]==1.0, :] = 0.0
        #     CombProbs[Probs[:,S,g]==1.0, g] = 1.0

        return CombProbs

    def ltp_combinator(self, LogProbs, base_LogProb):
        """
        ...
        """

        return np.sum(LogProbs) - (self.N-1)*base_LogProb