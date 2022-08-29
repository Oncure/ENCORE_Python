import numpy as np

import Resonances

def readENDF(file):
    with open(file, 'r') as contents:
        pass

def readSammyPar(file):
    with open(file, 'r') as contents:
        header = 'RESONANCES'.split(contents)[0]
        resonances = np.array([res_txt.split() for res_txt in '\n'.split('RESONANCES'.split(contents)[1])[1:]])
    E      = resonances[:, 0]
    Gg     = resonances[:, 1]
    Gn     = resonances[:, 1]
    SGType = resonances[:,-1] - 1

    return Resonances.Resonances(E=E, Gn=Gn, Gg=Gg), SGType
