import json
from math import pi, exp
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

__doc__ = """
Results module ...

...
"""

def PrintScore(Probability:ndarray, Answer:ndarray, run_name:str='', metric:str='best guess'):
    """
    ...
    """

    metric = metric.lower()

    if   metric == 'best guess':
        Guess = np.argmax(Probability, axis=1)
        Score = np.count_nonzero(Answer == Guess) / np.size(Answer)
    elif metric == 'probability score':
        Score = np.mean(Probability[:,Answer])

    if run_name:    print(f'{run_name} score = {100*Score:7.2f}%')
    else:           print(f'Score = {100*Score:7.2f}%')

def ProbCorrPlot(Prob_Guess_All, Answer, sg_name, figName:str='', figPath:str=''):
    """
    ...
    """

    nBin = round(np.sqrt(len(Answer)))
    edges = np.linspace(0.0, 1.0, nBin+1)
    X = (edges[:-1] + edges[1:])/2
    for t in range(Prob_Guess_All.shape[1]):
        Prob_Guess_Type = Prob_Guess_All[:,t]
        Prob_Guess_Cor  = Prob_Guess_Type[Answer == t]
        Count_All = np.histogram(Prob_Guess_Type, bins=edges)[0]
        Count_Cor = np.histogram(Prob_Guess_Cor , bins=edges)[0]

        X2         = X[Count_All != 0]
        Count_All2 = Count_All[Count_All != 0]
        Count_Cor2 = Count_Cor[Count_All != 0]
        Prob_Ans_Cor = Count_Cor2/Count_All2
        Prob_Ans_Cor_SD = 1.0 / np.sqrt(Count_All2)

        # Plotting:
        plt.errorbar(X2,Prob_Ans_Cor,Prob_Ans_Cor_SD,capsize=3,ls='none',c='k')
        plt.scatter(X2,Prob_Ans_Cor,marker='.',s=14,c='k')
        plt.plot([0,1],[0,1],':b',lw=2,label='Ideal Correlation')
        Norm = 0.25/np.max(Count_All)
        plt.fill_between(X,Norm*Count_All,alpha=0.8,color=(0.1,0.5,0.1),label='Probability Density')
        
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.title(f'Probability Accuracy for Spin Group {sg_name[t]}',fontsize=18)
        plt.xlabel('Predicted Probability',fontsize=15)
        plt.ylabel('Fraction of Correct Assignments',fontsize=15)
        plt.legend(fontsize=10)

        if figName:     plt.savefig(f'{figPath}{figName.format(sgn=sg_name[t])}.png')
        else:           plt.show()

def Plot(Data:ndarray, *args, **kwargs):
    
    def arg_manager(name:str, required=True, default=None):
        if name in kwargs.keys():
            return kwargs[name]
        elif len(args) >= 1:
            return args.pop(0)
        elif required:
            raise ValueError(f'Missing "{name}" argument.')
        else:
            return default

    distribution = arg_manager('distribution').lower()
    hist         = arg_manager('hist').lower()
    
    if   distribution == 'level spacing':
        if len(Data.shape) == 2:
            mls = float(arg_manager('mean', False, np.mean(Data[:,0])))
        else:
            mls = float(arg_manager('mean', False, np.mean(Data)))
        w    = float(arg_manager('w', False, 1.0))
        xlim = tuple(arg_manager('xlim', False, (0., max(*Data))))

        Data /= mls
        X = np.linspace(xlim)

        plt.xlim(*xlim)

        if   hist == 'pdf':
            Y = (pi/2) * X*exp(-(pi/4) * X**2)
        elif hist == 'cdf':
            Y = 1. - exp(-(pi/4) * X**2)
        elif hist == 'sf':
            Y = exp(-(pi/4) * X**2)
        else:
            raise ValueError('Unknown value for "hist".')
        
    elif distribution == 'energy':
        pass
    elif distribution == 'width':
        pass
    else:
        raise ValueError('Unknown distribution.')

def PlotECDF(X, E0=None, E_end=None):
    if E0 is not None:
        plt.plot([E0, X[0]], [0.0, 0.0], '-k')
    if E_end is not None:
        plt.plot([X[-1], E_end], [1.0, 1.0], '-k')

    N = len(X)
    [plt.plot([X[idx-1],X[idx]], [idx/N, idx/N], '-k') for idx in range(1,N)]
    [plt.plot([x, x], [idx/N, (idx+1)/N], '-k') for idx, x in enumerate(X)]
    plt.plot([E0, X[0]], [0.0, 0.0], '-k')


        
