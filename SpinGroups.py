from numpy import uint8

class halfint:
    """
    Data type for half-integers to represent spins
    """

    def __init__(self, value):
        if (2*value) % 1:
            raise ValueError(f'The number, {value}, is not a half-integer.')
        self.__2xValue = uint8(2*value)
    
    @property
    def parity(self):   return '+' if self.__2xValue >= 0 else '-'
    @property
    def value(self):    return self.__2xValue / 2

    def __str__(self):
        if self.__2xValue % 2 == 0:
            return '{:2i}'.format(self.__2xValue//2)
        else:
            return '{:2i}/2'.format(self.__2xValue)

    # Arithmetic:
    def __eq__(self, other) -> bool:
        if type(other) == self.__class__:
            return self.value == other.value
        else:
            return self.value == other
    def __lt__(self, other) -> bool:
        if type(other) == self.__class__:
            return self.value < other.value
        else:
            return self.value < other
    def __le__(self, other) -> bool:
        if type(other) == self.__class__:
            return self.value <= other.value
        else:
            return self.value <= other
    def __gt__(self, other) -> bool:
        if type(other) == self.__class__:
            return self.value > other.value
        else:
            return self.value > other
    def __ge__(self, other) -> bool:
        if type(other) == self.__class__:
            return self.value >= other.value
        else:
            return self.value >= other
    def __ne__(self, other) -> bool:
        if type(other) == self.__class__:
            return self.value != other.value
        else:
            return self.value != other
    def __add__(self, other):
        if type(other) == self.__class__:
            return self.__class__(self.value + other.value)
        else:
            return self.value + other
    def __radd__(self, other):
        return self.value + other
    def __sub__(self, other):
        if type(other) == self.__class__:
            return self.__class__(self.value - other.value)
        else:
            return self.value - other
    def __rsub__(self, other):
        return other - self.value
    def __mul__(self, other):
        return float(self.value * other)

class SpinGroup:
    """
    ...
    """

    def __init__(self, l, j, s):
        self.L = uint8(l)
        self.J = halfint(j)
        if s != None:   self.S = halfint(s)
        else:           self.S = None

    def name(self, option='j^pi') -> str:
        """
        ...
        """

        if   option.lower() == 'j^pi':
            if self.L % 2:  return f'{self.J}-'
            else:           return f'{self.J}+'
        elif option.lower() == '(l,j)':
            return f'({self.L},{self.J})'
        elif option.lower() == '(l,j,s)':
            return f'({self.L},{self.J},{self.S})'
        else:
            raise ValueError('Unknown option.')

    def __str__(self):
        return self.name('j^pi')

class SpinGroups:
    """
    ...
    """

    def __init__(self, SGs:list, l_max:int):
        self.SGs   = SGs
        self.l_max = l_max

    @classmethod
    def make(cls, Ls, Js, Ss = None):
        if (Ss != None) and (len(Ls) == len(Js) == len(Ss)):
            sgs = [SpinGroup(Ls[g], Js[g], Ss[g]) for g in range(len(Ls))]
        elif (len(Ls) == len(Js)) and (Ss == None):
            sgs = [SpinGroup(Ls[g], Js[g], None) for g in range(len(Ls))]
        else:
            raise ValueError('The number of "L", "J", and "S" values for spin-groups are not equal.')
        l_max = max(*Ls)
        return cls(sgs, l_max)

    @classmethod
    def find(cls, spin_target, spin_proj, l_max:int=1):
        l_max = int(l_max)
        sgs   = []
        for l in range(l_max+1):
            for s2 in range(abs(2*(spin_target - spin_proj)), 2*(spin_target + spin_proj+1), 2):
                for j2 in range(abs(s2 - 2*l),s2 + 2*l+2, 2):
                    sgs.append(SpinGroup(l, j2/2, s2/2))
        return cls(sgs, l_max)

    @property
    def L(self):        return [sg.L for sg in self.SGs]
    @property
    def J(self):        return [sg.J for sg in self.SGs]
    @property
    def S(self):        return [sg.S for sg in self.SGs]
    @property
    def num_sgs(self):  return len(self.SGs)