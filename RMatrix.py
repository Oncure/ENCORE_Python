import math
import numpy as np
from numpy import ndarray
from scipy.stats import chi2

__doc__ = """
This file contains the nuclear data theory from R-matrix theory from the ENDF and SAMMY manuals.
"""

def NuclearRadius(A:float) -> float:
    """
    eq. D.14 in ENDF manual
    """
    return 1.23 * A**(1/3) + 0.8 # fm = 10^-15 m

def Rho(A:float, ac:float, E:ndarray, E_thres:float=0.0) -> ndarray:
    """
    eq. II A.9 in the SAMMY manual
    """
    CONSTANT = 0.002197; # sqrt(2Mn)/hbar; units of (10^(-12) cm sqrt(eV)^-1)
    return CONSTANT*ac*(A/(A+1))*np.sqrt(E-E_thres)

def PenetrationFactor(rho:ndarray, l) -> ndarray:
    """
    Table II A.1 in the SAMMY manual
    """

    def PF(rho:ndarray, l:int):
        rho2 = rho**2
        if   l == 0:
            return rho
        elif l == 1:
            return rho*rho2    / (  1 +    rho2)
        elif l == 2:
            return rho*rho2**2 / (  9 +  3*rho2 +   rho2**2)
        elif l == 3:
            return rho*rho2**3 / (225 + 45*rho2 + 6*rho2**2 + rho2**3)
        else: # l >= 3
            # l = 3:
            Denom = (225 + 45*rho2 + 6*rho2**2 + rho2**3)
            P = rho*rho2**3 / Denom
            S = -(675 + 90*rho2 + 6*rho2**2) / Denom

            # Iteration equation:
            for l_iter in range(4,l+1):
                Mult = rho2 / ((l_iter-S)**2 + P**2)
                P = Mult*P
                S = Mult*S - l_iter
            return P

    if hasattr(l, '__iter__'):
        Pen = np.zeros((rho.shape[0],l.shape[1]))
        for t, lt in enumerate(l[0,:]):
            Pen[:,t] = PF(rho,lt)
    else:
        Pen = np.array(PF(rho,l))
    return Pen

# =================================================================================================
#    Width Probability Distributions
# =================================================================================================

def ReduceFactor(E:ndarray, l, A:float, ac:float) -> ndarray:
    """
    Multiplication factor to go from neutron width to reduced neutron width
    """

    rho = Rho(A, ac, E)
    return 1.0 / (2.0*PenetrationFactor(rho,l))

def PTBayes(Res, MeanParam, FalseWidthDist = None, Prior = None, GammaWidthOn:bool = False):
    """
    ...
    """
    
    if Prior == None:
        prob = MeanParam.FreqAll/np.sum(MeanParam.FreqAll)
        Prior = np.repeat(prob, repeats=Res.E.size, axis=0)
    Posterior = Prior

    MultFactor = (MeanParam.nDOF/MeanParam.Gnm) * ReduceFactor(Res.E, MeanParam.L, MeanParam.A, MeanParam.ac)
    Posterior[:,:-1] *= MultFactor * chi2.pdf(MultFactor * Res.Gn.reshape(-1,1), MeanParam.nDOF)

    if GammaWidthOn:
        MultFactor = MeanParam.gDOF/MeanParam.Ggm
        Posterior[:,:-1] *= MultFactor * chi2.pdf(MultFactor * Res.Gg.reshape(-1,1), MeanParam.gDOF)

    if MeanParam.FreqF != 0.0:
        Posterior[:,-1]  *= FalseWidthDist(Res.E, Res.Gn, Res.Gg)

    Total_Probability = np.sum(Posterior, axis=1)
    Posterior = Posterior / Total_Probability.reshape(-1,1)
    return Posterior, Total_Probability

# =================================================================================================
#    Sampling
# =================================================================================================

def SampleNeutronWidth(E:ndarray, Gnm:float, df:float, l:int, A:float, ac:float) -> ndarray:
    """
    ...
    """

    MultFactor = (df/Gnm)*ReduceFactor(np.array(E), l, A, ac)
    return np.random.chisquare(df, (len(E),1)) / MultFactor.reshape(-1,1)

def SampleGammaWidth(L:int, Ggm, df):
    """
    ...
    """
    
    MultFactor = df/Ggm
    return np.random.chisquare(df, (L,1)) / MultFactor

def WigSemicircleCDF(x):
    """
    CDF of Wigner's semicircle law distribution
    """
    return (x/np.pi) * np.sqrt(1.0 - x**2) + np.arcsin(x)/np.pi + 0.5

def SampleEnergies(EB, Freq, w, ensemble='NNE'):
    """
    ...
    """

    MULTIPLIER = 5
    
    if w == None:       w = 1.0
    if (ensemble in ('GOE','GUE','GSE')) and (w != 1.0):
        raise NotImplementedError(f'Cannot sample "{ensemble}" with Brody parameters')

    if ensemble == 'NNE': # Nearest Neighbor Ensemble
        # Sig = 6.0
        # wig_std = 0.522723200877 # Normalized Wigner distribution standard deviation
        if w == 1.0:
            L_Guess =  Freq * (EB[1] - EB[0]) * MULTIPLIER
            #L_Guess *= 1.5
            #L_Guess += Sig * wig_std * math.sqrt(L_Guess)
            L_Guess = round(L_Guess)

            LS = np.ndarray(L_Guess+1, dtype='f8')
            LS[0]  = EB[0] + abs(np.random.normal()) * (2/(math.pi * Freq ** 2))
            LS[1:] = np.sqrt((-4/math.pi) * np.log(np.random.uniform(size=L_Guess))) / Freq
            E = np.cumsum(LS)
            E = [e for e in E if e <= EB[1]]
        else:
            raise NotImplementedError('No functionality for Brody Distribution yet.')

    elif ensemble == 'GOE':
        # Since the eigenvalues do not follow the semicircle distribution
        # exactly, there is a small chance for some values that would never
        # occur with semicircle distribution. Therefore, we make extra
        # eigenvalues and use the ones that are needed. As extra precaution,
        # we select eigenvalues within a margin of the edges of the semicircle
        # distribution.
        margin = 0.1
        N_res_est = Freq*(EB[1]-EB[0])
        N_Tot = round((1 + 2*margin) * N_res_est)

        if seed is None:
            seed = np.random.randint(10000)
        rng = np.random.default_rng(seed)

        H = rng.normal(size=(N_Tot,N_Tot)) / math.sqrt(2)
        H += H.T
        H += math.sqrt(2) * np.diag(rng.normal(size=(N_Tot,)) - np.diag(H))
        eigs = np.linalg.eigvals(H) / (2*np.sqrt(N_Tot))
        eigs.sort()
        eigs = eigs[eigs >= -1.0+margin]
        eigs = eigs[eigs <=  1.0-margin]

        E = EB[0] + N_Tot * (WigSemicircleCDF(eigs) - WigSemicircleCDF(-1.0+margin)) / Freq
        return E[E < EB[1]]

    elif ensemble == 'Poisson':
        NumSamples = np.random.poisson(Freq * (EB[1]-EB[0]))
        E = np.random.uniform(low=EB[0], high=EB[1], size=(NumSamples, 1))

    E.sort()
    return E