#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from copy import copy
import math

import Levels
import FileReader
import Results
from Encore import Encore
from RMatrix import PTBayes
import Resonances
from SpinGroups import SpinGroups
# import Statistics

np.set_printoptions(precision=6, edgeitems=9, linewidth=130)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # verbosity controls
    parser.add_argument('-v', dest='verbose', default=False, action='store_true', help="Enable verbose output.")
    
    # Mean parameter information:
    parser.add_argument('--meanParams', type=str, default=None, help='JSON file with mean parameter information.')
    
    # Resonance infomration:
    parser.add_argument('--ENDFreader', type=str, default=None, help='ENDF file with the resonance information.')
    parser.add_argument('--sample', type=bool, default=False, action='store_true', help='Samples resonances from the mean parameter file.')

    # Running:
    parser.add_argument('--run', type=str, default=None, choices=['score', 'sample'], help='What sequence to run on the resonances.')

    # Results:
    parser.add_argument('--saveResults', type=str, default=None, help='Saves the probability scores or resampling to the given file.')
    parser.add_argument('--corrPlot', type=str, default=None, help='Gives a correlation plot on the probabilities for checking.')

    # ...

    return parser.parse_args()

#######################################################################################################################
# 							Main								#
#######################################################################################################################
if __name__ == "__main__":
    #args = parse_args()

    A    = 238
    EB   = (1000, 1100)

    FreqF = 0
    Freq = [2, 1]
    Gnm  = [8, 1]
    Ggm  = [1, 1]
    dfn  = [1, 1]
    dfg  = [100, 100]
    l    = [0, 1]
    j    = [0.5, 0.5]

    # Freq = [1.5, 3, 2, 0]
    # Gnm  = [5, 4, 3]
    # Ggm  = [1, 1, 1]
    # dfn  = [1, 1, 1]
    # dfg  = [100, 100, 100]
    # l    = [0, 1, 1]
    # j    = [0.5, 0.5, 1.5]

    SGs = SpinGroups.make(l, j)
    MP_True = Resonances.MeanParameters(Freq = Freq, Gnm = Gnm, nDOF = dfn, Ggm = Ggm, gDOF = dfg, A = A, sg = SGs, EB = EB, FreqF=FreqF)
    Res, Types, Missed_Res, Missed_Types = MP_True.sample()

    N = 30
    # A2s = np.zeros(N)
    # Ts  = np.zeros(N)
    # KSs = np.zeros(N)

    # GNMs = np.zeros(N)
    LTPs   = np.zeros(N)
    MLTPs  = np.zeros(N)
    LTPs2  = np.zeros(N)
    Freqs  = np.zeros(N)
    for i in range(N):
        # GNMs[i] = 25*np.random.rand()
        # Gnm2 = copy(Gnm)
        # Gnm2[0] = GNMs[i]

        Freqs[i] = np.sum(Freq[:2])*(0.1+0.8*(i/(N-1)))
        Freq2 = copy(Freq)
        Freq2[0] = Freqs[i]
        Freq2[1] = np.sum(Freq[:2]) - Freq2[0]

        MP_Guess = Resonances.MeanParameters(Freq = Freq2, Gnm = Gnm, nDOF = dfn, Ggm = Ggm, gDOF = dfg, A = A, sg = SGs, EB = EB, FreqF = FreqF)
        Prior, TPPrior = PTBayes(Res, MP_Guess)
        runMaster = Levels.RunMaster(Res.E, MP_Guess.EB, Prior, TPPrior, MP_Guess.FreqAll)
        Posterior, LTP = runMaster.WigBayesPartitionMaster(True, verbose=True)
        LTPs[i] = LTP
        MLTPs[i] = Encore.MeanLogTotProb(MP_Guess)
        # ===========================================

        Res2, Types2, Missed_Res2, Missed_Types2 = MP_Guess.sample()
        Prior2, TPPrior2 = PTBayes(Res2, MP_Guess)
        runMaster2 = Levels.RunMaster(Res2.E, MP_Guess.EB, Prior2, TPPrior2, MP_Guess.FreqAll)
        Posterior2, LTP2 = runMaster2.WigBayesPartitionMaster(True, verbose=True)
        LTPs2[i] = LTP2
        # ===========================================

        # runMaster = Levels.RunMaster(Res.E, Res.EB, Prior, MP_Guess.Freq)
        # LSMerge, iMaxMerge, PriorMerge = runMaster.MergePartitioner([[0],[1]])
        # ENCORE = Encore(PriorMerge, LSMerge, iMaxMerge)
        # Samples = ENCORE.WigSample(50)
        # A2, T, KS = Statistics.LSGoodnessFit(Res, MP_Guess, Samples)
        # A2s[i] = A2
        # Ts[i]  = T
        # KSs[i] = KS

    import matplotlib.pyplot as plt

    # print(LTPs)
    # print(GNMs)
    # print(Gnm[0])

    # plt.plot(GNMs, A2s, '.k')
    # plt.plot(GNMs, Ts , '.b')
    # plt.plot(GNMs, KSs, '.r')
    fig, ax = plt.subplots()
    ax.plot(Freqs, LTPs , '.k', label='Main data')
    ax.plot(Freqs, LTPs2, '.b', label='Sampled data')
    ax.plot(Freqs, MLTPs, '.r', label='Est. mean sampled data')
    ax.axvline(Freq[0], color='blue', label='True mean level-spacing')
    ax.axvline(Freq[1], color='green', label='Conjugate mean level-spacing')
    plt.xlabel('Reciprocal of Mean Level-Spacing', fontsize=15)
    plt.ylabel('Total Probability', fontsize=15)

    plt.legend(fontsize=10)

    limits = (min(*LTPs, *LTPs2, *MLTPs), max(*LTPs, *LTPs2, *MLTPs))
    ytickPos = np.round(np.linspace(*limits, 10))
    tick_labels = [rf'$10^{{{int(value)}}}$' for value in ytickPos]
    plt.yticks(ytickPos, tick_labels)

    plt.tight_layout()
    plt.show()

    # print(Probs)

    # Results.PrintScore(Prior, Types, 'Best-Guess Porter-Thomas', metric='best guess')
    # Results.PrintScore(Posterior, Types, 'Best-Guess ENCORE', metric='best guess')

    # Results.PrintScore(Prior, Types, 'Prob-Score Porter-Thomas', metric='probability score')
    # Results.PrintScore(Posterior, Types, 'Prob-Score ENCORE', metric='probability score')
    # Results.ProbCorrPlot(Posterior, Types, ['A', 'B', 'C', 'False'])

    # Merge = Levels.Merger(MP_True.Freq[:,:-1])
    # PriorM, iMaxM = Merge.FindLevelSpacings(Res.E, Res.EB, Prior[:,:-1], True)





    # Levels = Levels.Levels(Res.E, Res.EB, Prior, MP_True.Freq)

    # LS = Levels.LevelSpacings(1e-9)

    # ENCORE = Encore(Prior, LS)

    # if any(ENCORE.PW == np.nan):
    #     raise RuntimeError('ENCORE failed to run properly')
    # Probs = ENCORE.WigBayes()

    # Sample = ENCORE.WigSample(2)

    # ltp = ENCORE.LogTotProb(EB, Freq[-1])
    # print(f'Log Total Probability = {ltp:.5e}')

    # Results.PrintScore(Probs, Types)
    # Results.ProbCorrPlot(Probs, Types, ['A', 'B', 'F'])

    # print(np.array([Probs[j,Types[j]] for j in range(ENCORE.Length)]))

    # print(f'L = {ENCORE.Length}')
    # # print(f'SP = {Probs}')
    # # print(f'TP = {ENCORE.TP}')
    # print(f'RATIO = {1-Probs[:,-1]}')

    # print(f'Probs | Answers =\n{np.concatenate((Probs[:,:-1], Types.reshape(-1,1)), axis=1)}')


    # ============================================================================================

    # Current issues:
    #   * Merging has not been fully implemented
    #   * "WigSample" has not been tested
    #   * Plots and results have not been fully implemented


