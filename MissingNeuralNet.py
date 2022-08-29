import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

BG = 30.0
N = 20000
NP = 25
S = 25
Ma = 0.05
Mb = 0.01

dE = 0.05
MLS = 1

DE = N*dE

tail = int((NP-1)/2)
E = dE * (np.arange(N + 2*tail) - tail)

def Noise(X):
    return np.random.poisson(S*X+BG)

NumRes = np.random.poisson(DE/MLS)
El = np.random.uniform(low=0.0, high=DE, size=NumRes).reshape(1,-1)
Ga = Ma * abs(np.random.normal(size=NumRes)).reshape(1,-1)
Gb = Mb * abs(np.random.normal(size=NumRes)).reshape(1,-1)
TrueData = np.sum(Ga*Gb / ((E.reshape(-1,1)-El)**2 + ((Ga+Gb)/2)**2), axis=1)
Resonance = [(idx == np.round(El/dE)).any() for idx in np.round(E[tail:-tail]/dE)]
RawData = Noise(TrueData)
Seen = [RawData.tolist()[idx:idx+NP] for idx in range(N)]

NeuralNet = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 3))
NeuralNet.fit(Seen, Resonance)

# =====================================

NumRes = np.random.poisson(DE/MLS)
El = np.random.uniform(low=0.0, high=DE, size=NumRes).reshape(1,-1)
Ga = Ma * abs(np.random.normal(size=NumRes)).reshape(1,-1)
Gb = Mb * abs(np.random.normal(size=NumRes)).reshape(1,-1)
TrueData2 = np.sum(Ga*Gb / ((E.reshape(-1,1)-El)**2 + ((Ga+Gb)/2)**2), axis=1)
Resonance2 = [(idx == np.round(El/dE)).any() for idx in np.round(E[tail:-tail]/dE)]
RawData2 = Noise(TrueData2)
Seen2 = [RawData2.tolist()[idx:idx+NP] for idx in range(N)]

Probs = np.array(NeuralNet.predict_proba(Seen2))[:,1]

def ProbCorrPlot(Prob_Guess_All, Answer):
    nBin = round(np.sqrt(len(Answer)))
    edges = np.linspace(0.0, 1.0, nBin+1)
    X = (edges[:-1] + edges[1:])/2
    Prob_Guess_Type = Prob_Guess_All
    Prob_Guess_Cor  = Prob_Guess_Type[Answer]
    Count_All = np.histogram(Prob_Guess_Type, bins=edges)[0]
    Count_Cor = np.histogram(Prob_Guess_Cor , bins=edges)[0]

    X2         = X[Count_All != 0]
    Count_All2 = Count_All[Count_All != 0]
    Count_Cor2 = Count_Cor[Count_All != 0]
    Prob_Ans_Cor = Count_Cor2/Count_All2
    Prob_Ans_Cor_SD = 1.0 / np.sqrt(Count_All2)

    # Plotting:
    plt.figure()
    plt.errorbar(X2,Prob_Ans_Cor,Prob_Ans_Cor_SD,capsize=3,ls='none',c='k')
    plt.scatter(X2,Prob_Ans_Cor,marker='.',s=14,c='k')
    plt.plot([0,1],[0,1],':b',lw=2,label='Ideal Correlation')
    Norm = 0.25/np.max(Count_All)
    plt.fill_between(X,Norm*Count_All,alpha=0.8,color=(0.1,0.5,0.1),label='Probability Density')
    
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.title('Probability Accuracy',fontsize=18)
    plt.xlabel('Predicted Probability',fontsize=15)
    plt.ylabel('Fraction of Correct Assignments',fontsize=15)
    plt.legend(fontsize=10)

    plt.show()

print(f'Fraction existing = {np.sum(Resonance2)/len(Resonance2)}')
print(f'Score = {(np.sum(Probs[Resonance2]) + np.sum(1.0-Probs[[~r for r in Resonance2]]))/N}')

ProbCorrPlot(Probs, Resonance2)

plt.figure()
plt.plot(E, TrueData, '-b')
plt.plot(E, (RawData-BG)/S, '.k')
plt.title('Cross-section',fontsize=18)
plt.xlabel('Energy',fontsize=15)
plt.ylabel('Cross-section',fontsize=15)
plt.show()
