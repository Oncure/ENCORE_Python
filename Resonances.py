import numpy as np
import pandas as pd

import SpinGroups
import RMatrix

# =================================================================================================
#    Mean Parameters:
# =================================================================================================
class MeanParameters:
    """
    ...
    """

    DEFAULT_GDOF = 500

    def __init__(self,**kwargs):
        """
        ...
        """

        ParamNames = set(kwargs.keys())
        def ParamsIn(*Names: str, **options: str):
            if 'type' in options.keys():    ParamType = options['type']
            else:                           ParamType = 'array1'
            ParamSet = (ParamNames & set(Names))
            ParamList = list(ParamSet)
            if len(ParamList) >= 2:
                ParamStr = '\"' + '\", \"'.join(ParamList[:-1]) + f', and \"{ParamList[-1]}\"'
                raise ValueError(f'Cannot accept multiple parameters, {ParamStr}.')
            elif len(ParamList) == 1:
                param = ParamList[0]
                if ParamType == 'array1':
                    paramValue = np.array(kwargs[param]).reshape(1,-1)
                elif ParamType == 'float':
                    paramValue = float(kwargs[param])
                elif ParamType == 'int':
                    paramValue = int(kwargs[param])
                elif ParamType == 'halfint':
                    paramValue = SpinGroups.halfint(kwargs[param])
                elif ParamType == 'tuple':
                    paramValue = tuple(kwargs[param])
                elif ParamType == 'pass':
                    paramValue = kwargs[param]
                return paramValue, True
            else:
                return None, False

        # Isotope Spin:
        self.I, IExists = ParamsIn('I', 'S', 'spin', 'isotope_spin', type='halfint')

        # Atomic Number:
        self.Z, ZExists = ParamsIn('Z', 'atomic_number', type='int')

        # Atomic Mass:
        self.A, AExists = ParamsIn('A', 'atomic_mass', type='int')

        # Mass:
        value, massExists = ParamsIn('mass', 'Mass', 'm', 'M', type='float')
        if massExists:
            self.mass = value
        elif AExists:
            self.mass = float(self.A)

        # Atomic Radius:
        value, acExists = ParamsIn('Ac', 'ac', 'atomic_radius', 'scatter_radius', type='float')
        if acExists:
            self.ac = value
        elif AExists:
            self.ac = RMatrix.NuclearRadius(self.A)

        # Energy Range:
        value, EBExists = ParamsIn('EB', 'energy_bounds', 'energy_range', type='tuple')
        if EBExists:
            self.EB = value
            if len(self.EB) != 2:
                raise ValueError('"EB" can only have two values for an interval')
            elif self.EB[0] > self.EB[1]:
                raise ValueError('"EB" must be a valid increasing interval')

        # Spin-Groups:
        self.sg, sgExists = ParamsIn('sg', 'SG', 'spin-group', type='pass')

        # Frequencies:
        self.Freq, FreqExists = ParamsIn('freq', 'Freq', 'frequency', 'Frequency')

        # Mean Level Spacings:
        value, MLSExists = ParamsIn('mean_spacing', 'mean_level_spacing', 'mls')
        if FreqExists & MLSExists:
            raise ValueError('Cannot have both mean level spacing and frequencies')
        elif MLSExists:
            self.Freq = 1.0 / value

        # Mean Neutron Widths:
        self.Gnm, GnmExists = ParamsIn('mean_neutron_width', 'Gnm')

        # Neutron Channel Degrees of Freedom:
        value, nDOFExists = ParamsIn('nDOF', 'nDF')
        if nDOFExists:
            self.nDOF = value
        elif GnmExists:
            self.nDOF = [1] * self.Gnm.shape[1] # Assume that there is one DOF

        # Mean Gamma Widths:
        self.Ggm, GgmExists = ParamsIn('mean_gamma_width', 'Ggm')

        # Gamma Channel Degrees of Freedom:
        value, gDOFExists = ParamsIn('gDOF', 'gDF')
        if gDOFExists:
            self.gDOF = value
        elif GgmExists:
            self.gDOF = [self.DEFAULT_GDOF] * self.Ggm.shape[1] # Arbitrarily high DOF
        
        # Brody Parameter:
        self.w, wExists = ParamsIn('w', 'brody', 'Brody', 'brody_parameter')

        # False Frequency:
        value, FalseFreqExists = ParamsIn('freqF', 'FreqF', 'false_frequency', type='float')
        if FalseFreqExists:
            self.FreqF = value
        else:
            self.FreqF = 0.0

    @property
    def n(self):        return self.Freq.shape[1]
    @property
    def L(self):        return np.array(self.sg.L).reshape(1,-1)
    @property
    def J(self):        return np.array(self.sg.J).reshape(1,-1)
    @property
    def S(self):        return np.array(self.sg.S).reshape(1,-1)
    @property
    def MLS(self):      return 1.0 / self.Freq
    @property
    def FreqAll(self):  return np.append(self.Freq, np.array(self.FreqF, ndmin=2), axis=1)
            
    def sample(self, missingFactor=0.0, ensemble='NNE'):
        """
        ...
        """

        n = self.Freq.shape[1]
        if self.w is None:
            Et = [RMatrix.SampleEnergies(self.EB, self.Freq[0,g], w=None, ensemble=ensemble) for g in range(n)]
        else:
            Et = [RMatrix.SampleEnergies(self.EB, self.Freq[0,g], w=self.w[0,g], ensemble=ensemble) for g in range(n)]
        
        if self.FreqF != 0.0:
            raise NotImplementedError('The width sampling for false resonances has not been worked out yet.')
            Et += [RMatrix.SampleEnergies(self.EB, self.FreqF, ensemble='Poisson')]
    

        E = np.array([e for Etg in Et for e in Etg])
        idx = np.argsort(E)
        E = E[idx]

        Gn = np.array([ele for g in range(n) for ele in RMatrix.SampleNeutronWidth(Et[g], self.Gnm[0,g], self.nDOF[0,g], self.L[0,g], self.A, self.ac)])[idx]
        Gg = np.array([ele for g in range(n) for ele in RMatrix.SampleGammaWidth(len(Et[g]), self.Ggm[0,g], self.gDOF[0,g])])[idx]

        spingroups = np.array([g for g in range(n) for e in Et[g]])[idx]

        missed_idx = np.random.rand(*E.shape) < missingFactor
        resonances = Resonances(E=E, Gn=Gn, Gg=Gg)
        return resonances[~missed_idx], spingroups[~missed_idx], resonances[missed_idx], spingroups[missed_idx]

# =================================================================================================
# Resonances:
# =================================================================================================
class Resonances:
    """
    ...
    """

    def __init__(self, E, *args, **kwargs):
        self.properties = ['E']

        Idx = np.argsort(E)
        self.E = np.array(E).reshape(-1)[Idx]

        if 'Gn' in kwargs.keys():
            self.Gn = np.array(kwargs['Gn']).reshape(-1)[Idx]
            self.properties.append('Gn')
        elif len(args) >= 2:
            self.Gn = np.array(args[1]).reshape(-1)[Idx]
            self.properties.append('Gn')

        if 'Gg' in kwargs.keys():
            self.Gg = np.array(kwargs['Gg']).reshape(-1)[Idx]
            self.properties.append('Gg')
        elif len(args) >= 3:
            self.Gg = np.array(args[2]).reshape(-1)[Idx]
            self.properties.append('Gg')

        if 'GfA' in kwargs.keys():
            self.GfA = np.array(kwargs['GfA']).reshape(-1)[Idx]
            self.properties.append('GfA')
        elif len(args) >= 4:
            self.GfA = np.array(args[3]).reshape(-1)[Idx]
            self.properties.append('GfA')
        
        if 'GfB' in kwargs.keys():
            self.GfB = np.array(kwargs['GfB']).reshape(-1)[Idx]
            self.properties.append('GfB')
        elif len(args) >= 5:
            self.GfB = np.array(args[4]).reshape(-1)[Idx]
            self.properties.append('GfB')

        if 'SG' in kwargs.keys():
            self.SG = np.array(kwargs['SG']).reshape(-1)[Idx]
            self.properties.append('SG')
        elif len(args) >= 6:
            self.SG = np.array(args[5]).reshape(-1)[Idx]
            self.properties.append('SG')

    # Get resonances by indexing the "Resonances" object:
    def __getitem__(self, Idx):
        kwargs = {}
        for property in self.properties:
            if   property == 'E'   :    kwargs['E']   = self.E[Idx]
            elif property == 'Gn'  :    kwargs['Gn']  = self.Gn[Idx]
            elif property == 'Gg'  :    kwargs['Gg']  = self.Gg[Idx]
            elif property == 'GfA' :    kwargs['GfA'] = self.GfA[Idx]
            elif property == 'GfB' :    kwargs['GfB'] = self.GfB[Idx]
            elif property == 'SG'  :    kwargs['SG']  = self.SG[Idx]
        return Resonances(**kwargs)

    # Print the resonance data as a table:
    def __str__(self):
        Data = []
        for name in self.properties:
            if   name == 'E':
                Data.append(self.E)
            elif name == 'Gn':
                Data.append(self.Gn)
            elif name == 'Gg':
                Data.append(self.Gg)
            elif name == 'GfA':
                Data.append(self.GfA)
            elif name == 'GfB':
                Data.append(self.GfB)
            elif name == 'SG':
                Data.append(self.SG)
        Data = np.array(Data).reshape(len(self.properties), self.L).T
        table_str = '\n'.join(str(pd.DataFrame(data=Data, columns=self.properties)).split('\n')[:-2])
        return table_str

    @property
    def L(self):
        return self.E.size