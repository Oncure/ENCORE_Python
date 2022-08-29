import numpy as np

def WigSemicircleCDF(x):
    """
    CDF of Wigner's semicircle law distribution
    """
    return (x/np.pi) * np.sqrt(1.0 - x**2) + np.arcsin(x)/np.pi + 0.5

def SampleGOE(meanLevelSpacing:float, N_res:int, E0:float=0.0, seed:int=None):
    """
    Sample from the Gaussian Orthogonal Ensemble (GOE)
    """

    # Since the eigenvalues do not follow the semicircle distribution
    # exactly, there is a small chance for some values that would never
    # occur with semicircle distribution. Therefore, we make extra
    # eigenvalues and use the ones that are needed. As extra precaution,
    # we select eigenvalues within a margin of the edges of the semicircle
    # distribution.
    margin = 0.1
    N_Tot = round((1 + 2*margin) * N_res + 25)

    if seed is None:
        seed = np.random.randint(10000)
    rng = np.random.default_rng(seed)

    # Get random matrix and eigenvalues:
    sqrt2 = np.sqrt(2)
    H = rng.normal(size=(N_Tot,N_Tot)) / sqrt2
    H += H.T
    H += sqrt2 * np.diag(rng.normal(size=(N_Tot,)) - np.diag(H))
    eigs = np.linalg.eigvals(H) / (2*np.sqrt(N_Tot))
    eigs.sort()
    eigs = eigs[eigs > -1.0+margin]
    eigs = eigs[eigs <  1.0-margin]

    # If for some reason there is a large number of resonances
    # refusing to follow the semicircle law, try again.
    if len(eigs) < N_res:
        return SampleGOE(meanLevelSpacing, N_res, seed)

    # Make energies uniformly spaced using semicircle law transformation:
    E = E0 + N_Tot * meanLevelSpacing * (WigSemicircleCDF(eigs[:N_res]) - WigSemicircleCDF(-1.0+margin))
    return E

def SampleGOE2(Freq:float, EB:tuple, seed:int=None):
    """
    Sample from the Gaussian Orthogonal Ensemble (GOE)
    """

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

    sqrt2 = np.sqrt(2)
    H = rng.normal(size=(N_Tot,N_Tot)) / sqrt2
    H += H.T
    H += sqrt2 * np.diag(rng.normal(size=(N_Tot,)) - np.diag(H))
    eigs = np.linalg.eigvals(H) / (2*np.sqrt(N_Tot))
    eigs.sort()
    eigs = eigs[eigs > -1.0+margin]
    eigs = eigs[eigs <  1.0-margin]

    E = EB[0] + N_Tot * (WigSemicircleCDF(eigs) - WigSemicircleCDF(-1.0+margin)) / Freq
    return E[E < EB[1]]

# ========================================================================================
# Validation:
import matplotlib.pyplot as plt
def WignerDist(X, MLS):
    return (np.pi/(2*MLS**2)) * X * np.exp(-np.pi/(4*MLS**2) * X**2)
def PlotECDF(X, E0=0.0):
    N = len(X)
    plt.plot([E0, X[0]], [0.0, 0.0], '-k')
    [plt.plot([X[idx-1],X[idx]], [idx/N, idx/N], '-k') for idx in range(1,N)]
    [plt.plot([x, x], [idx/N, (idx+1)/N], '-k') for idx, x in enumerate(X)]

MLS = 2.0
# E = SampleGOE(MLS, 1000)
E = SampleGOE2(1/MLS, (0.0, 1000*MLS))

plt.figure()
PlotECDF(E)
plt.plot([0.0, np.max(E)], [0.0, 1.0], '--b')
plt.show()

LS = np.diff(E)
plt.figure()
plt.hist(LS, density=True)
X = np.linspace(0.0, np.max(LS), 1000)
Y = WignerDist(X, MLS)
plt.plot(X, Y, '-k')
plt.show()




