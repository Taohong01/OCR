from pylab import *
from scipy.integrate import odeint
import time as tmstamp
import numpy as np
import pandas as pd
import sys
from copy import copy, deepcopy
import itertools

class FundamentalPhysicalConstants(object):
    """ definion for Fundamental Physical Constants per (2006 CODATA recommended values:
    P. J. Mohr, B. N. Taylor, and D. B. Newell, “The 2006 CODATA Recommended Values of the Fundamental
    Physical Constants, Web Version 5.1,” available at http://physics.nist.gov/constants 
    (National Institute of Standards and Technology, Gaithersburg, MD 20899, 31 December 2007).)
    """
    def __init__(self):
        # general format: self.constant = dict(Value = , Units='  '  , Name = '  ') 
        self.c = dict(Value = 2.99792458e8, Units='m/s', Name = 'Speed of Light') 
        self.mu0 = dict(Value = 4*np.pi * 1.0e-7, Units=' N/A^2 '  , Name = ' Permeability of Vacuum ')  
        self.epsilon0 = dict(Value = 8.854187817e-12, Units='  F/m '  , Name = ' Permittivity of Vacuum ') 
        self.h = dict(Value = 6.62606896e-34, Units=' J*s '  , Name = ' Planck’s Constant ') 
        self.hbar = dict(Value = self.h['Value'], Units= self.h['Units'], Name = self.h['Name']) 
        self.u = dict(Value = 1.660538782e-27 , Units=' kg '  , Name = ' Atomic Mass Unit  ') 
        self.kB = dict(Value = 1.3806504e-23, Units=' J/K '  , Name = ' Boltzmann’s Constant  ') 
                 
                 
                 
class AtomicElement(object):
    
    def __init__(self, Name = 'Rubidium', Mass= (1.443160648e-25, 'kg'), AtomicNumber = 37, 
                 NuclearSpin = '3/2', TotalNucleons = 87, RelativeNaturalAbundance = 0.2783,
                 D2LineWaveLength =(780.241209686, 'nm'), RecoilVelocity = (5.8845, 'mm/s'),
                 RecoilFrequency_Omegar = (2*pi*3.7710, 'kHz')   ):
        self.Name = Name
        self.Mass = Mass
        self.AtomicNumber = AtomicNumber
        self.NuclearSpin = NuclearSpin
        self.TotalNucleons = TotalNucleons
        self.RelativeNaturalAbundance = RelativeNaturalAbundance
        
        self.D2LineWaveLength = D2LineWaveLength
        self.RecoilVelocity  = RecoilVelocity
        self.RecoilFrequency_Omegar = RecoilFrequency_Omegar 
        self.TwoPhotonWaveVector_keff = (2.0 * 2.0 * pi / self.D2LineWaveLength[0] * 1.0e9, 'm^-1') 
          
        
        
class CommonElementList(object):
    Rb87 = AtomicElement()
    Cs133 = AtomicElement(Name='Cesium', Mass = (2.20694650e-25, 'kg'), AtomicNumber = 55, NuclearSpin= '7/2', 
                          TotalNucleons = 133, RelativeNaturalAbundance = 1.0,
                          D2LineWaveLength =(852.347, 'nm'), RecoilVelocity = (3.5225, 'mm/s'),
                          RecoilFrequency_Omegar = (2*pi*2.0663, 'kHz') )
    
    Na23 = AtomicElement(Name='Cesium', Mass = (2.20694650e-25, 'kg'), AtomicNumber = 55, NuclearSpin= '7/2', 
                          TotalNucleons = 133, RelativeNaturalAbundance = 1.0,
                          D2LineWaveLength =(852.347, 'nm'), RecoilVelocity = (3.5225, 'mm/s'),
                          RecoilFrequency_Omegar = (2*pi*2.0663, 'kHz') )
    
    K39 = AtomicElement(Name='Cesium', Mass = (2.20694650e-25, 'kg'), AtomicNumber = 55, NuclearSpin= '7/2', 
                          TotalNucleons = 133, RelativeNaturalAbundance = 1.0,
                          D2LineWaveLength =(852.347, 'nm'), RecoilVelocity = (3.5225, 'mm/s'),
                          RecoilFrequency_Omegar = (2*pi*2.0663, 'kHz') )
    
    Li = AtomicElement(Name='Cesium', Mass = (2.20694650e-25, 'kg'), AtomicNumber = 55, NuclearSpin= '7/2', 
                          TotalNucleons = 133, RelativeNaturalAbundance = 1.0,
                          D2LineWaveLength =(852.347, 'nm'), RecoilVelocity = (3.5225, 'mm/s'),
                          RecoilFrequency_Omegar = (2*pi*2.0663, 'kHz') )
    Fr = AtomicElement(Name='Cesium', Mass = (2.20694650e-25, 'kg'), AtomicNumber = 55, NuclearSpin= '7/2', 
                          TotalNucleons = 133, RelativeNaturalAbundance = 1.0,
                          D2LineWaveLength =(852.347, 'nm'), RecoilVelocity = (3.5225, 'mm/s'),
                          RecoilFrequency_Omegar = (2*pi*2.0663, 'kHz') )

        
class NumberMismatchingException(Exception):
    """Generating NumberMismathingException
    Application example:
    
    source:
    raise NumberMismatchingException(message ='\nError: Invalid internal state initialization: 
         number of internal states do not match the number of internal state name definition.')      
         
    tryandcatch:
    try: 
        newstates= QuantumAtomicInternalStates(ProbAmps = np.array([1.0+0j, 0.0+0j, 0.0]), Names = ['|1>','|2>', '|3>'])
    except NumberMismatchingException, e:
        print e.message
    """
    
    def __init__(self, value =-1, message='Number mistaching is found!'):
        self.value = value
        print message
        
    def __str__(self):
        return self.value

class GeneralQuantumStates(object):
    """include NumberMismathingException
    Application example:
    try: 
        newstates= GeneralQuantumStates(ProbAmps = np.array([1.0+0j, 0.0+0j, 1.0]), Names = ['|1>','|2>', '|3>'])
    except NumberMismatchingException, e:
        print e.message
    """
    
    def __init__(self, ProbAmps = np.array([1.0+0j, 0.0+1.0j]), Names = ['|1>','|2>'], EigenValues = np.array([0.0, 0.0])):
        if len(ProbAmps) != len(Names) or len(Names) != len(EigenValues) :
            raise NumberMismatchingException(message ='\nError: Invalid internal state initialization: number of internal states do not match the number of internal state name definition.')                        
        self.StatesAmplitude = pd.Series(ProbAmps, index = Names, name='ProbabilityAmplitudes')
        self.EigenValues = pd.Series(EigenValues, index = Names, name='EigenValues')
        self.TotalProbability()
        pass
    
    def TotalProbability(self):
        self.TotalProbability = np.linalg.norm(self.States.values)**2
        
        
    def Normalization(self):
        SqRootTotalProbability = np.linalg.norm(self.States.values)
        normlizationfunction = lambda x: x/SqRootTotalProbability
        self.States = pd.DataFrame(self.States.map(normlizationfunction))
        self.TotalProbability = 1.0
        return self.States
    
    def Normalize(self, InputStates):
        # better add a type judgement and error exception if the type is not pandas.Seriers type
        # this function is supposed to return a normalized state probability amplitude
        pass

    def GenerateArbitraryNumberOfStates(self): 
        """input integer indices for the internal states and then probability amplitudes"""
        pass
    
class QuantumAtomicInternalStates(object):
    """include NumberMismathingException
    Application example:
    try: 
        newstates= QuantumAtomicInternalStates(ProbAmps = np.array([1.0+0j, 0.0+0j, 1.0]), Names = ['|1>','|2>', '|3>'])
    except NumberMismatchingException, e:
        print e.message
    """
    
    def __init__(self, ProbAmps = np.array([1.0+0j, 0.0+1.0j]), Names = ['|1>','|2>']):
        if len(ProbAmps) != len(Names):
            raise NumberMismatchingException(message ='\nError: Invalid internal state initialization: number of internal states do not match the number of internal state name definition.')                        
        self.States = pd.Series(ProbAmps, index = Names, name='Probability Amplitudes')
        self.TotalProbability()
        pass
    
    def TotalProbability(self):
        self.TotalProbability = np.linalg.norm(self.States.values)**2
        
        
    def Normalization(self):
        SqRootTotalProbability = np.linalg.norm(self.States.values)
        normlizationfunction = lambda x: x/SqRootTotalProbability
        self.States = pd.DataFrame(self.States.map(normlizationfunction))
        self.TotalProbability = 1.0
        return self.States
    
    def Normalize(self, InputStates):
        # better add a type judgement and error exception if the type is not pandas.Seriers type
        # this function is supposed to return a normalized state probability amplitude
        pass

    def GenerateAtomicInternalStates(self): 
        """input integer indices for the internal states and then probability amplitudes"""
        pass
    
    
    
class Velocity(object):
    def __init__(self, vec_v = np.array([0.0, 0.0, 0.0]) ):
        self._x, self._y, self._z = vec_v
        self._vec = vec_v
        
    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, value):
        self._x = value
        #self._r = np.array([value, self._y, self._z])
        self._vec[0] =self._x
       
    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, value):
        self._vec[1] = value
        self._y = value
        
    @property
    def z(self):
        return self._z
    
    @z.setter
    def z(self, value):
        self._vec[2] = value
        self._z = value

    @property
    def vec(self):
        return self._vec
    
    @vec.setter
    def vec(self, value):
        self._vec = value
        self._x = value[0]
        self._y = value[1]
        self._z = value[2] 
        

class SpatialCoordinates(object):
    
    def __init__(self, vec_r = np.array([0.0, 0.0, 0.0]) ):
        self._x, self._y, self._z = vec_r
        self._vec = vec_r
        
    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, value):
        self._x = value
        #self._r = np.array([value, self._y, self._z])
        self._vec[0] =self._x
       
    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, value):
        self._vec[1] = value
        self._y = value
        
    @property
    def z(self):
        return self._z
    
    @z.setter
    def z(self, value):
        self._vec[2] = value
        self._z = value

    @property
    def vec(self):
        return self._vec
    
    @vec.setter
    def vec(self, value):
        self._vec = value
        self._x = value[0]
        self._y = value[1]
        self._z = value[2] 
       
    def UpdateDueTo(self, Velocity = Velocity(), timeduration=0.0):
        self._vec = self._vec + Velocity.vec * timeduration
        self._x, self._y, self._z = self._vec
       


    
class QuantumAtomicExternalMomentumStates(GeneralQuantumStates):
    
    def __init__(self):
        pass   
    
class QuantumAtomicInter_ExternalCoupledStates(object):
    """include NumberMismathingException
    Application example:
    try: 
        newstates= GeneralQuantumStates(ProbAmps = np.array([1.0+0j, 0.0+0j, 1.0]), Names = ['|1>','|2>', '|3>'])
    except NumberMismatchingException, e:
        print e.message
    """
    
    def __init__(self, 
                 Labels = list(          ['|2,+3hbarkeff>','|1,+2hbarkeff>','|2, +1hbarkeff>','|1, 0hbarkeff>', '|2,-1hbarkeff>','|1,-2hbarkeff>','|2,-3hbarkeff>']), 
                 StateValues = np.array([[1.0,             0.0,             1.0,               0.0,              1.0,             0.0,            1.0],
                                         [9.0,             4.0,             1.0,               0.0,              1.0,             4.0,            9.0],
                                         [3.0,             2.0,             1.0,               0.0,              -1.0,            -2.0,           -3.0],
                                         [0.0+0j,          0.0+0j,          .0+0j,            1.0+0j,           0.0+0j,          0.0+0j,         0+0.0j]]),
                 Names=list(['Einternal', 'Ekinetic', 'Momentum', 'ProbAmp'])):
        if StateValues.shape != (len(Names), len(Labels)) :
            raise NumberMismatchingException(message ='\nError: Invalid state initialization: Dimension of labels do not match the number of states.')                        
        self.States = pd.DataFrame(StateValues.T, index = Labels, columns= Names)
        self.States[['Einternal', 'Ekinetic', 'Momentum']] = self.States[['Einternal', 'Ekinetic', 'Momentum']].astype('float64')
        self.States[['Einternal', 'Ekinetic', 'Momentum', 'ProbAmp']] = self.States[['Einternal', 'Ekinetic', 'Momentum', 'ProbAmp']].astype('object')
        self.GetTotalProbability()
        self.StateVec = np.concatenate([StateValues[3,:].real, StateValues[3,:].imag])
        self.StateDimension = len(Labels)
    
    def InCaseOf2hbark(self):
        Labels = list(          ['|2, +1hbarkeff>','|1, 0hbarkeff>', '|2,-1hbarkeff>']) 
        StateValues = np.array([[1.0,               0.0,               1.0],
                                [1.0,               0.0,               1.0],
                                [1.0,               0.0,              -1.0],
                                [0.0+0j,            1.0+0j,           0.0+0j]])
        Names=list(['Einternal', 'Ekinetic', 'Momentum', 'ProbAmp'])
        self.States = pd.DataFrame(StateValues.T, index = Labels, columns= Names)
        self.States[['Einternal', 'Ekinetic', 'Momentum']] = self.States[['Einternal', 'Ekinetic', 'Momentum']].astype('float64')
        self.States[['Einternal', 'Ekinetic', 'Momentum', 'ProbAmp']] = self.States[['Einternal', 'Ekinetic', 'Momentum', 'ProbAmp']].astype('object')
        self.GetTotalProbability()
        self.StateVec = np.concatenate([StateValues[3,:].real, StateValues[3,:].imag])
        self.StateDimension = len(Labels)
        
    def InCaseOf12hbark(self):
        Labels = list(          ['|1,+6hbarkeff>','|2, +5hbarkeff>','|1,+4hbarkeff>', '|2,+3hbarkeff>','|1,+2hbarkeff>','|2, +1hbarkeff>','|1, 0hbarkeff>', '|2,-1hbarkeff>','|1,-2hbarkeff>','|2,-3hbarkeff>']) 
        StateValues = np.array([[0.0,             1.0,               0.0,               1.0,              0.0,             1.0,               0.0,              1.0,             0.0,            1.0],
                                [36.0,            25.0,              16.0,              9.0,              4.0,             1.0,               0.0,              1.0,             4.0,            9.0],
                                [6.0,             5.0,               4.0,               3.0,              2.0,             1.0,               0.0,              -1.0,            -2.0,           -3.0],
                                [0.0+0j,          0.0+0j,            0.0+0j,            0.0+0j,           0.0+0j,          0.0+0j,            1.0+0j,           0.0+0j,          0.0+0j,         0+0.0j]])
        Names=list(['Einternal', 'Ekinetic', 'Momentum', 'ProbAmp'])
        self.States = pd.DataFrame(StateValues.T, index = Labels, columns= Names)
        self.States[['Einternal', 'Ekinetic', 'Momentum']] = self.States[['Einternal', 'Ekinetic', 'Momentum']].astype('float64')
        self.States[['Einternal', 'Ekinetic', 'Momentum', 'ProbAmp']] = self.States[['Einternal', 'Ekinetic', 'Momentum', 'ProbAmp']].astype('object')
        self.GetTotalProbability()
        self.StateVec = np.concatenate([StateValues[3,:].real, StateValues[3,:].imag])
        self.StateDimension = len(Labels)
    
    def InCaseOf8hbark(self):
        Labels = list(          ['|1,+4hbarkeff>', '|2,+3hbarkeff>','|1,+2hbarkeff>','|2, +1hbarkeff>','|1, 0hbarkeff>', '|2,-1hbarkeff>','|1,-2hbarkeff>','|2,-3hbarkeff>']) 
        StateValues = np.array([[0.0,               1.0,              0.0,             1.0,               0.0,              1.0,             0.0,            1.0],
                                 [16.0,              9.0,              4.0,             1.0,               0.0,              1.0,             4.0,            9.0],
                                 [4.0,               3.0,              2.0,             1.0,               0.0,              -1.0,            -2.0,           -3.0],
                                 [0.0+0j,            0.0+0j,           0.0+0j,          0.0+0j,            1.0+0j,           0.0+0j,          0.0+0j,         0+0.0j]])
        Names=list(['Einternal', 'Ekinetic', 'Momentum', 'ProbAmp'])
        self.States = pd.DataFrame(StateValues.T, index = Labels, columns= Names)
        self.States[['Einternal', 'Ekinetic', 'Momentum']] = self.States[['Einternal', 'Ekinetic', 'Momentum']].astype('float64')
        self.States[['Einternal', 'Ekinetic', 'Momentum', 'ProbAmp']] = self.States[['Einternal', 'Ekinetic', 'Momentum', 'ProbAmp']].astype('object')
        self.GetTotalProbability()
        self.StateVec = np.concatenate([StateValues[3,:].real, StateValues[3,:].imag])
        self.StateDimension = len(Labels)
        
    
    def GetTotalProbability(self):
        self.TotalProbability = np.abs(np.linalg.norm(self.States['ProbAmp'].values)**2)
        return self.TotalProbability
        
        
    def ProbNormalization(self):       
        SqRootTotalProbability = np.sqrt(self.GetTotalProbability())
        normlizationfunction = lambda x: x/SqRootTotalProbability
        self.States[['ProbAmp']] = normlizationfunction(self.States[['ProbAmp']])
        self.TotalProbability = 1.0
        return self.States
    
    
    
    def Normalize(self, InputStates):
        # better add a type judgement and error exception if the type is not pandas.Seriers type
        # this function is supposed to return a normalized state probability amplitude
        pass

    def GenerateArbitraryNumberOfStates(self): 
        """input integer indices for the internal states and then probability amplitudes"""
        pass
    
    def ComplexVecConvertedFromStateVec(self, InputStateVec):
        ComplexVec= InputStateVec[0:(len(InputStateVec)/2)]+1.0j*InputStateVec[(len(InputStateVec)/2):len(InputStateVec)]
        return ComplexVec
    
    def StateVecConvertedFromComplexVec(self, ComplexVec):
        StateVec = np.array(list(ComplexVec.real) + list(ComplexVec.imag))
        return StateVec
                 
class SingleAtom(object):
    
    def __init__(self, Time=0.0, QuatumStates=QuantumAtomicInter_ExternalCoupledStates(), 
                 Position = SpatialCoordinates(),
                 Velocity = Velocity(), 
                 R0 = SpatialCoordinates(),
                 V0 = Velocity(),
                 AtomDefinition=CommonElementList().Rb87):
        # THSE: add R0 and V0
        
        self.FPC = FundamentalPhysicalConstants()
        
        self.Time = Time
        self.Definition = AtomDefinition
        self.Quantum = QuatumStates
        self.Position = Position
        self.Velocity = Velocity
        self.R0 = R0 # THSE: add initial offset,  assuming already normalized by length unit
        self.V0 = V0 # THSE: add initial offset, assuming already normalized by velocity unit
        self.units =['mass', 'velocity', 'position']
        
                
        #univeral units in this calculation
        self.LengthUnit = (2.0 / self.Definition.TwoPhotonWaveVector_keff[0], 'm')
        self.KeffUnit = (self.Definition.TwoPhotonWaveVector_keff[0], 'm^-1')
        self.TimeUnit = (1.0/ (4.0 * self.Definition.RecoilFrequency_Omegar[0] *10**3), 's')
        self.AngularFrequencyUnit = (4 * self.Definition.RecoilFrequency_Omegar[0] *10**3, 's^-1')
        self.EnergyUnit = (self.FPC.hbar['Value'] * 4 * self.Definition.RecoilFrequency_Omegar[0]*1.0e3, 'J*s*Hz')
        #some convenient units derived from the above univeral units 
        self.OneCentimeter = (1.0e-2/self.LengthUnit[0], 'LengthUnit')
        self.kbTk3nK = self.FPC.kB['Value'] * 3.0e-9/ self.EnergyUnit[0] # in units of 'J/K*3nK/(J*s*Hz)  = 3x10^-9', 
        # Usage example, for an arbitrary temperature, say, 5 nK in SI units, and we should assigne a value of 
        # 5/3 * self.kbTk3nK to the temperature variable T used for simulation. When we want to convert the number of T
        # back to SI units, we should use T * self.kbTk3nK * self.EnergyUnit to get value in SI units. 
        ###print '----kbTk3nK corresponds to Tk=3 nK and kbTk is in unites of (kb * 3nK/( hbar* 4 recoil freqeuncy)):', kbTk3nK
        self.hbar = 1.0
        self.Mass = 1.0/2

        #self.Delta = 1000000.0
        #self.pipulse = 0.785 * 2 /17.197 * 21 /20.2287880384 *61
        
        self.Delta = -500.0e6 * 2 * np.pi / self.AngularFrequencyUnit[0] # THSE: change to 500MHz blue
        print 'Delta:         ', self.Delta
        self.pipulse = 6.0e-6 / self.TimeUnit[0] # THSE: change to 6.0 mus
        print 'pipulse        ', self.pipulse
        
        ##print '------------------pi pulse duration', self.pipulse *(self.TimeUnit[0])*10**6, 'micro-seconds' 
        self.interpulse = self.pipulse * 1.0  #3.0 * 10 * 3 / 16.266 * 14.0
        
        self.keff = 1.0
        self.RabiFreqmax = sqrt(np.pi*np.abs(self.Delta)/2.0/self.pipulse)   #in unites of (hbar*keff)^2/(2*M*habar) = 4 * recoil frequency = 4* 2π • 3.6325 kHz
        ##print '--------------effective two photo Rabi frequency 2*Omega^2/Delta/2pi is', 2*self.RabiFreqmax**2/self.Delta*(self.AngularFrequencyUnit[0])/2.0/np.pi/10**3, 'KHz'     
        ## THSE: correct the expression for RabiFreqmax
        self.BeamWaistAt1OverESquared = 3.5/10 * self.OneCentimeter[0] # THSE: change to 7mm in diameter 
        self.CloudRadiusAt1OverESquared = 0.5 * self.OneCentimeter[0] /10 #previous value was 3.0, the name should be fullwidth instead of radius, because it is divided by 2
        ###print '-----BeamWaistAt1OverESquared = 3.0 * OneCentimeter[0]:',BeamWaistAt1OverESquared 
        ###print '-----CloudRadiusAt1OverESquared = 3.0 * OneCentimeter[0] /10:', CloudRadiusAt1OverESquared 
        # THSE: changed to Cloud Radius to 0.5mm 
        
        
        
        self.kbTk = self.kbTk3nK /3.0 * 8.0 # assuming we want to get 8 nK temperature atom cloud 
            
        self.Omega10=0.0
        self.Omega20=0.0
        self.Transitions =          [(3,2), (2,1), (1,0), (0,1), (1,2), (2,3)] + [ (2,1), (1,0), (0,1), (1,2), (2,3)]
        self.keffsign    =          [+1,    -1,    +1,    +1,     -1,    +1  ] + [ -1,    +1,    +1,    -1,    +1  ]

        self.pulsetiming = np.array([0,     1.0,   2.0,   13.5,   15.0,  16.5,     18.0,  19.5,  31.0,  32.0,  33.0])* self.interpulse
        ##print '---------------------interrogation time:', self.pulsetiming[-1]/2.0*(self.TimeUnit[0]), 'seconds'
        self.pulseduration=np.array([0.5,   1.0,   1.0,   1.0,    1.0,   1.0,      1.0,   1.0,   1.0,   1.0,   0.5]) * self.pipulse 
        #pulsetiming = array([0, interpulse, 2.0*interpulse, 13.5*interpulse, 15.0*interpulse,    16.5*interpulse,   
        #                   18.0*interpulse, 19.5*interpulse, 31.0*interpulse, 32.0*interpulse, 33.0*interpulse ])
        
        self.StatesFilterUpperArm = np.array([self.StateFilterFunction([0, 0, 0, 1, 0, 0, 0,  0, 0, 0, 1, 0, 0, 0], ShiftIndex = SF) 
                                                for SF in [0, -1, -2, -3, -2, -1, 0, 0, 0, 0, 0] ])

        self.StatesFilterLowerArm = np.array([self.StateFilterFunction([0, 0, 0, 1, 0, 0, 0,  0, 0, 0, 1, 0, 0, 0], ShiftIndex = SF)  
                                                for SF in [0, 0, 0, 0, 0, 0, -1, -2, -2, -1, 0] ])
    def InCaseOf6hbark(self):
        self.RabiFreqmax = sqrt(np.pi*np.abs(self.Delta)/2.0/self.pipulse)   #in unites of (hbar*keff)^2/(2*M*habar) = 4 * recoil frequency = 4* 2π
        # THSE: correct the RabiFreqmax expression
        self.Transitions =          [(3,2), (2,1), (1,0), (0,1), (1,2), (2,3)] + [ (2,1), (1,0), (0,1), (1,2), (2,3)]
        self.keffsign    =          [+1,    -1,    +1,    +1,     -1,    +1  ] + [ -1,    +1,    +1,    -1,    +1  ]

        self.pulsetiming = np.array([0,     1.0,   2.0,   13.5,   15.0,  16.5,     18.0,  19.5,  31.0,  32.0,  33.0])* self.interpulse
        ##print '---------------------interrogation time:', self.pulsetiming[-1]/2.0*(self.TimeUnit[0]), 'seconds'
        self.pulseduration=np.array([0.5,   1.0,   1.0,   1.0,    1.0,   1.0,      1.0,   1.0,   1.0,   1.0,   0.5]) * self.pipulse 
        #pulsetiming = array([0, interpulse, 2.0*interpulse, 13.5*interpulse, 15.0*interpulse,    16.5*interpulse,   
        #                   18.0*interpulse, 19.5*interpulse, 31.0*interpulse, 32.0*interpulse, 33.0*interpulse ])
        
        self.StatesFilterUpperArm = np.array([self.StateFilterFunction([0, 0, 0, 1, 0, 0, 0,  0, 0, 0, 1, 0, 0, 0], ShiftIndex = SF) 
                                                for SF in [0, -1, -2, -3, -2, -1, 0, 0, 0, 0, 0] ])

        self.StatesFilterLowerArm = np.array([self.StateFilterFunction([0, 0, 0, 1, 0, 0, 0,  0, 0, 0, 1, 0, 0, 0], ShiftIndex = SF)  
                                                for SF in [0, 0, 0, 0, 0, 0, -1, -2, -2, -1, 0] ])
    """    
    def InCaseOf2hbark(self):
        self.RabiFreqmax = sqrt(np.pi*self.Delta/2.0/self.pipulse)   #in unites of (hbar*keff)^2/(2*M*habar) = 4 * recoil frequency = 4* 2π
        self.Transitions =          [(3,2), (2,3), (2,3)]
        self.keffsign    =          [+1,     +1,    +1  ]

        self.pulsetiming = np.array([0,     16.5,   33.0])* self.interpulse
        ##print '---------------------interrogation time:', self.pulsetiming[-1]/2.0*(self.TimeUnit[0]), 'seconds'
        self.pulseduration=np.array([0.5,   1.0,   0.5]) * self.pipulse 
        #pulsetiming = array([0, interpulse, 2.0*interpulse, 13.5*interpulse, 15.0*interpulse,    16.5*interpulse,   
        #                   18.0*interpulse, 19.5*interpulse, 31.0*interpulse, 32.0*interpulse, 33.0*interpulse ])
        
        self.StatesFilterUpperArm = np.array([self.StateFilterFunction([0, 0, 0, 1, 0, 0, 0,  0, 0, 0, 1, 0, 0, 0], ShiftIndex = SF) 
                                                for SF in [0, -1,  0] ])

        self.StatesFilterLowerArm = np.array([self.StateFilterFunction([0, 0, 0, 1, 0, 0, 0,  0, 0, 0, 1, 0, 0, 0], ShiftIndex = SF)  
                                                for SF in [0, -1,  0] ])
        self.StatesFilterUpperArm = np.array([[0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]])
        self.StatesFilterLowerArm = np.array([[0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]])
        print 'upperfilter', self.StatesFilterUpperArm
        print 'lowerfilter', self.StatesFilterLowerArm
    """    
    def InCaseOf2hbark(self):
        self.RabiFreqmax = sqrt(np.pi*np.abs(self.Delta)/2.0/self.pipulse)   #in unites of (hbar*keff)^2/(2*M*habar) = 4 * recoil frequency = 4* 2π
        # THSE: correct RabiFreqmax expression
        self.Transitions =          [(1,0), (0,1),  (0,1)]
        self.keffsign    =          [+1,     +1,     +1  ]

        self.pulsetiming = np.array([0,     16.5,   33.0])* self.interpulse
        ##print '---------------------interrogation time:', self.pulsetiming[-1]/2.0*(self.TimeUnit[0]), 'seconds'
        self.pulseduration=np.array([0.5,   1.0,     0.5]) * self.pipulse 
        #pulsetiming = array([0, interpulse, 2.0*interpulse, 13.5*interpulse, 15.0*interpulse,    16.5*interpulse,   
        #                   18.0*interpulse, 19.5*interpulse, 31.0*interpulse, 32.0*interpulse, 33.0*interpulse ])
        
        self.StatesFilterUpperArm = np.array([self.StateFilterFunction([0, 1, 0,  0, 1, 0], ShiftIndex = SF) 
                                                for SF in [0, -1, 0] ])

        self.StatesFilterLowerArm = np.array([self.StateFilterFunction([0, 1, 0,  0, 1, 0], ShiftIndex = SF)  
                                                for SF in [0, -1, 0] ])
        self.StatesFilterUpperArm = np.array([[1, 1, 0, 1, 1, 0], [1, 0, 0, 1, 0, 0], [1, 1, 0, 1, 1, 0]])
        self.StatesFilterLowerArm = np.array([[1, 1, 0, 1, 1, 0], [0, 1, 0, 0, 1, 0], [1, 1, 0, 1, 1, 0]])
 
    def InCaseOf2hbarkWithInitialDelay(self):
        # THSE: fix RabiFreqmax to the case of pi pulse = 6 mus
        self.RabiFreqmax = sqrt(np.pi*np.abs(self.Delta)/2.0/(6.0e-6 / self.TimeUnit[0]))   #in unites of (hbar*keff)^2/(2*M*habar) = 4 * recoil frequency = 4* 2π
        # THSE: correct RabiFreqmax expression
        self.Transitions =          [(1,0),  (1,0), (0,1),  (0,1)]
        self.keffsign    =          [+1,     +1,     +1,     +1  ]

        self.pulsetiming = np.array([-2.93,   0,     5.0,   10.0])* self.interpulse
        ##print '---------------------interrogation time:', self.pulsetiming[-1]/2.0*(self.TimeUnit[0]), 'seconds'
        self.pulseduration=np.array([1.0e-6, 0.5,   1.0,     0.5]) * self.pipulse 
        #pulsetiming = array([0, interpulse, 2.0*interpulse, 13.5*interpulse, 15.0*interpulse,    16.5*interpulse,   
        #                   18.0*interpulse, 19.5*interpulse, 31.0*interpulse, 32.0*interpulse, 33.0*interpulse ])
        
        self.StatesFilterUpperArm = np.array([self.StateFilterFunction([0, 1, 0,  0, 1, 0], ShiftIndex = SF) 
                                                for SF in [0, 0, -1, 0] ])

        self.StatesFilterLowerArm = np.array([self.StateFilterFunction([0, 1, 0,  0, 1, 0], ShiftIndex = SF)  
                                                for SF in [0, 0, -1, 0] ])
        self.StatesFilterUpperArm = np.array([[1, 1, 1, 1, 1, 1], [1, 1, 0, 1, 1, 0], [1, 0, 0, 1, 0, 0], [1, 1, 0, 1, 1, 0]])
        self.StatesFilterLowerArm = np.array([[1, 1, 1, 1, 1, 1], [1, 1, 0, 1, 1, 0], [0, 1, 0, 0, 1, 0], [1, 1, 0, 1, 1, 0]])
        
         
    def InCaseOf12hbark(self):
        self.RabiFreqmax = sqrt(np.pi*np.abs(self.Delta)/2.0/self.pipulse)   #in unites of (hbar*keff)^2/(2*M*habar) = 4 * recoil frequency = 4* 2π
        # THSE: correct RabiFreqmax expression
        self.Transitions =          [(6,5), (5,4), (4,3), (3,2), (2,1), (1,0), (0,1), (1,2), (2,3), (3,4), (4,5), (5,6)] + [(5,4), (4,3), (3,2), (2,1), (1,0), (0,1), (1,2), (2,3), (3,4), (4,5), (5,6)]
        self.keffsign    =          [+1,    -1,     +1,   -1,     +1,    -1,    -1,    +1,    -1,    +1,   -1,     +1  ] + [ -1,    +1,    -1,    +1,    -1,    -1,    +1,   -1,    +1,    -1,     +1  ]

        self.pulsetiming = np.array([0,     1.0,   2.0,   3.0,   4.0,   5.0,   9.0,    10.5,  12.0,  13.5,   15.0,  16.5,    18.0,  19.5,  21.0,  22.5,  24.0,  28.0,  29.0,  30.0,  31.0,  32.0,  33.0])* self.interpulse
        ##print '---------------------interrogation time:', self.pulsetiming[-1]/2.0*(self.TimeUnit[0]), 'seconds'
        self.pulseduration=np.array([0.5,   1.0,   1.0,   1.0,    1.0,   1.0,   1.0,    1.0,   1.0,   1.0,    1.0,   1.0,    1.0,   1.0,   1.0,   1.0,   1.0,   1.0,    1.0,   1.0,   1.0,   1.0,   0.5]) * self.pipulse 
        #pulsetiming = array([0, interpulse, 2.0*interpulse, 13.5*interpulse, 15.0*interpulse,    16.5*interpulse,   
        #                   18.0*interpulse, 19.5*interpulse, 31.0*interpulse, 32.0*interpulse, 33.0*interpulse ])
        
        self.StatesFilterUpperArm = np.array([self.StateFilterFunction([0, 0, 0, 0, 0, 0, 1, 0, 0, 0,  0, 0, 0, 0, 0, 0, 1, 0, 0, 0], ShiftIndex = SF) 
                                                for SF in [0, -1, -2, -3, -4, -5, -5, -4, -3, -2, -1, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ])

        self.StatesFilterLowerArm = np.array([self.StateFilterFunction([0, 0, 0, 0, 0, 0, 1, 0, 0, 0,  0, 0, 0, 0, 0, 0, 1, 0, 0, 0], ShiftIndex = SF)  
                                                for SF in [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  -1, -2, -3, -4, -5, -5, -4, -3, -2, -1, 0] ])
        
    
    def InCaseOf8hbark(self):
        self.RabiFreqmax = sqrt(np.pi*np.abs(self.Delta)/2.0/self.pipulse)   #in unites of (hbar*keff)^2/(2*M*habar) = 4 * recoil frequency = 4* 2π
        # THSE: correct RabiFreqmax expression
        self.Transitions =         [(4,3), (3,2), (2,1), (1,0), (0,1), (1,2), (2,3), (3,4)] + [(3,2), (2,1), (1,0), (0,1), (1,2), (2,3), (3,4)]
        self.keffsign    =         [+1,    -1,     +1,   -1,     -1,    +1,   -1,     +1  ] + [ -1,    +1,    -1,    -1,    +1,    -1,     +1  ]

        self.pulsetiming = np.array([0,     1.0,   2.0,   3.0,   12.0,  13.5,   15.0,  16.5,    18.0,  19.5,  21.0,  30.0,  31.0,  32.0,  33.0])* self.interpulse
        ##print '---------------------interrogation time:', self.pulsetiming[-1]/2.0*(self.TimeUnit[0]), 'seconds'
        self.pulseduration=np.array([0.5,   1.0,    1.0,   1.0,    1.0,   1.0,   1.0,   1.0,   1.0,   1.0,    1.0,   1.0,   1.0,   1.0,   0.5]) * self.pipulse 
        #pulsetiming = array([0, interpulse, 2.0*interpulse, 13.5*interpulse, 15.0*interpulse,    16.5*interpulse,   
        #                   18.0*interpulse, 19.5*interpulse, 31.0*interpulse, 32.0*interpulse, 33.0*interpulse ])
        
        self.StatesFilterUpperArm = np.array([self.StateFilterFunction([0, 0, 0, 0, 1, 0, 0, 0,  0, 0, 0, 0, 1, 0, 0, 0], ShiftIndex = SF) 
                                                for SF in [0, -1, -2, -3, -3, -2, -1, 0,  0, 0, 0, 0, 0, 0, 0] ])

        self.StatesFilterLowerArm = np.array([self.StateFilterFunction([0, 0, 0, 0, 1, 0, 0, 0,  0, 0, 0, 0, 1, 0, 0, 0], ShiftIndex = SF)  
                                                for SF in [0, 0, 0, 0, 0, 0, 0, 0,  -1, -2, -3, -3, -2, -1, 0] ])
        
    
    
    def StateFilterFunction(self, state0 = [0, 0, 0, 1, 0, 0, 0,  0, 0, 0, 1, 0, 0, 0], ShiftIndex=0):
        state0 = [0] + state0 
        stateL=self.movetoleft(copy(state0))
        stateR=self.movetoright(copy(state0))
        statesum = list(np.array(state0) + np.array(stateL) + np.array(stateR))
        if ShiftIndex == 0:
            statesum.pop(0)
            return np.array(statesum)
        elif ShiftIndex >0:
            for ii in np.arange(ShiftIndex):
                statesum = self.movetoright(statesum)
            statesum.pop(0)
            return np.array(statesum)
        else:
            for ii in np.arange(abs(ShiftIndex)):
                statesum = self.movetoleft(statesum)
            statesum.pop(0)
            return np.array(statesum)
               
    def movetoleft(self, inputlist=list([1, 2, 3, 4])):
        inputlist.append(inputlist.pop(0))
        return inputlist
    
    def movetoright(self, inputlist=list([1, 2, 3, 4])):
        inputlist.insert(0, inputlist.pop(len(inputlist)-1))
        return inputlist
    

    def FreePropagationOverTime(self, PropagationTime = 0):
        self.Position.UpdateDueTo(self.Velocity, PropagationTime)
        self.Time += PropagationTime
        
    def RadomizePositionAndMomentumAccordingToTempAndCloudSize(self):
        # THSE: needs modification to take into account of initial velocity and position offsets
        psigma=sqrt(self.kbTk/2.0)
        rsigma= self.CloudRadiusAt1OverESquared/2.0
        self.Velocity.vec=self.V0.vec + np.random.normal(0,psigma,3)/self.Mass
        self.Position.vec=self.R0.vec + np.random.normal(0,rsigma,3)
        

        
class LaserPulse(object):
    
    def __init__(self, InteractingAtom=SingleAtom()):
        pass
             
    def AmplitudeNormalizedPulse(self, time, timestart, timeend):
        return self.rectpulse(time, timestart, timeend)
    
    
    def LaserBeamShape(self):
        pass
      
    
    def rectpulse(self, time,time0,time1):
        if (time-time0< 0) or (time-time1 >0):
            return 0.0
        else:
            return 1.0
    
class LaserAtomInteraction(object):
    
    def __init__(self, InputAtom=SingleAtom(), InputLaserPulse=LaserPulse(), InteractionPeriodIndex = 0, ArmFilterIndex = 'upper', Laser2PhoDetuningFactor=1.0):
        self.Atom  = InputAtom
        self.Laser = InputLaserPulse
        
        self.IntDex = InteractionPeriodIndex
        self.PulseTimeStart = self.Atom.pulsetiming[self.IntDex]
        self.PulseTimeEnd = self.PulseTimeStart + self.Atom.pulseduration[self.IntDex]
        self.FlightDuration =  0.0 if self.IntDex == 0 else self.Atom.pulsetiming[self.IntDex]-self.Atom.pulsetiming[self.IntDex-1]
        self.TwoPhotonDetuning = self.TwoPhotonResonanceFrequency(TransitionPair = self.Atom.Transitions[self.IntDex]) # this is the 2 photon resonance frequency of the two momentum states that we want to select
        self.delta2Laser = Laser2PhoDetuningFactor * self.Atom.keffsign[self.IntDex] * self.TwoPhotonDetuning #This is the laser frequency we want to set in order to enhance the transition pair we attompt to enhance and suppress others
        #THSE: EEEE
        #print self.delta2Laser
        self.Delta = self.Atom.Delta
        self.StateFilter = self.Atom.StatesFilterUpperArm if ArmFilterIndex == 'upper' else self.Atom.StatesFilterLowerArm
        self.N = 500 # number of points to descritize the integration
        # THSE: add beam distance
        self.intrabeamDistance = 23.0/10 * self.Atom.OneCentimeter[0] # 23 mm

    def LaserPulse(self, time):
        return self.Laser.AmplitudeNormalizedPulse(time, self.PulseTimeStart, self.PulseTimeEnd)

    # Raman laser field configuration and time sequence definition:
    def Omega1(self, time, Omega10):
        Omega1t = Omega10 * self.LaserPulse(time)
        return Omega1t #in unites of (hbar*keff)^2/(2*M*habar) = 4 * recoil frequency = 4* 2π • 3.6325 kHz

    def Omega2(self, time, Omega20):
        Omega2t = Omega20 * self.LaserPulse(time)
        return Omega2t #in unites of (hbar*keff)^2/(2*M*habar) = 4 * recoil frequency = 4* 2π • 3.6325 kHz  
    
    def TwoPhotonResonanceFrequency(self, TransitionPair =(3,2)):        
        p0, p1 = (self.Atom.Quantum.States['Momentum'][TransitionPair[0]], self.Atom.Quantum.States['Momentum'][TransitionPair[1]])
        DetuningDueToKineticEnergyDifference = np.abs(p0**2 - p1**2)/(2.0*self.Atom.Mass*self.Atom.hbar)
        return DetuningDueToKineticEnergyDifference
    
    # AC Stark Shifts + Effective 2-photon Rabi Frequency + 2-photon deturning with 2-k Doppler effect
    def ACStarkShift1(self, Omega1):
        return 1.0j*abs(Omega1)**2/self.Delta

    def ACStarkShift2(self, Omega2):
        return 1.0j*abs(Omega2)**2/self.Delta

    def Omegaeff(self, Omega1, Omega2):
        return 1.0j*Omega1*conjugate(Omega2)/self.Delta
    
    # this is the laser detuning in regarding to a transition from arbitrary momentum state pi to pj
    def delta(self, pi, pj):
        M=self.Atom.Mass
        hbar=self.Atom.hbar
        return abs(pi)**2/(2*M*hbar)-abs(pj)**2/(2*M*hbar) + self.delta2Laser

    # Raman coupling and Hamiltonian
    def RamanCoupling(self, pi,pj,time, Omega10, Omega20):
        return  self.Omegaeff(self.Omega1(time, Omega10), self.Omega2(time, Omega20))*np.exp(1.0j*self.delta(pi,pj)*time) 

    
    def Hamiltonian12HbarK(self, time, Omega10, Omega20):
        """This 12HbarK means expendable, the dimension of H depends on the self.Atom.Quantum.StateDemension
        """
        hbar=self.Atom.hbar
        p = self.Atom.Velocity.z * self.Atom.Mass
        
        H= 1.0j*zeros([self.Atom.Quantum.StateDimension,self.Atom.Quantum.StateDimension])
        
        # Off-diagonal terms of the Hamiltonian    
        keff= self.Atom.keffsign[self.IntDex]
        SD = self.Atom.Quantum.StateDimension
        if SD % 2 == 0:
            SD1 = np.arange(2, SD, 2)                  
            SD2 = np.arange(2, SD + 1, 2) - 1          
            if keff > 0:
                for Hindex in SD1:
                    n_left  = self.Atom.Quantum.States['Momentum'][Hindex]
                    n_right = self.Atom.Quantum.States['Momentum'][Hindex-1]
                    H[Hindex,Hindex-1] = self.RamanCoupling(p+n_left*hbar*keff, p+n_right*hbar*keff, time, Omega10, Omega20)
            else:
                keff = abs(keff)# There should be no sign in the keff used below
                for Hindex in SD2:
                    n_left  = self.Atom.Quantum.States['Momentum'][Hindex-1]
                    #print self.Atom.Quantum.States['Momentum']
                    #print 'hindex', Hindex
                    n_right = self.Atom.Quantum.States['Momentum'][Hindex]
                    H[Hindex-1,Hindex] = self.RamanCoupling(p+n_left*hbar*keff, p+n_right*hbar*keff, time, Omega10, Omega20)

        else:
            SD1 = np.arange(1, SD, 2)                
            SD2 = np.arange(1, SD, 2) + 1            
            if keff > 0:
                for Hindex in SD1:
                    n_left  = self.Atom.Quantum.States['Momentum'][Hindex]
                    n_right = self.Atom.Quantum.States['Momentum'][Hindex-1]
                    H[Hindex,Hindex-1] = self.RamanCoupling(p+n_left*hbar*keff, p+n_right*hbar*keff, time, Omega10, Omega20)
            else:
                keff = abs(keff)# There should be no sign in the keff used below
                for Hindex in SD2:
                    n_left  = self.Atom.Quantum.States['Momentum'][Hindex-1]
                    #print self.Atom.Quantum.States['Momentum']
                    #print 'hindex', Hindex
                    n_right = self.Atom.Quantum.States['Momentum'][Hindex]
                    H[Hindex-1,Hindex] = self.RamanCoupling(p+n_left*hbar*keff, p+n_right*hbar*keff, time, Omega10, Omega20)
        """        
        if keff > 0:
            for Hindex in np.arange(1, self.Atom.Quantum.StateDimension, 2):
                n_left  = self.Atom.Quantum.States['Momentum'][Hindex]
                n_right = self.Atom.Quantum.States['Momentum'][Hindex-1]
                H[Hindex,Hindex-1] = self.RamanCoupling(p+n_left*hbar*keff, p+n_right*hbar*keff, time, Omega10, Omega20)
        else:
            keff = abs(keff)# There should be no sign in the keff used below
            for Hindex in np.arange(1, self.Atom.Quantum.StateDimension, 2):
                n_left  = self.Atom.Quantum.States['Momentum'][Hindex]
                #print self.Atom.Quantum.States['Momentum']
                #print 'hindex', Hindex
                n_right = self.Atom.Quantum.States['Momentum'][Hindex+1]
                H[Hindex,Hindex+1] = self.RamanCoupling(p+n_left*hbar*keff, p+n_right*hbar*keff, time, Omega10, Omega20)
        """
        
        H = H - transpose(conjugate(H))  # because the imaginary  factor in the coupling equation already included in the H so we must use a negative sign at here. 

        # Diagonal terms of the Hamiltonian       
        for Hindex in np.arange(self.Atom.Quantum.StateDimension):
            H[Hindex,Hindex] = self.ACStarkShift2(self.Omega2(time, Omega20)) if self.Atom.Quantum.States['Einternal'][Hindex] == 1.0  else self.ACStarkShift1(self.Omega1(time, Omega10))
 
        return H

    #RHS defintion for Schrodinger Equation, called by the ODE solver    
    def RHS_SchrodingerEq2(self, StateVector, time, Omega10, Omega20):
        #assuming the atom movement during pulse interaction is neglegible small
        #so Omega10 and Omega20 remain as constant
      
        dStateVector_dt=array(zeros([self.Atom.Quantum.StateDimension*2]))
        #the input argument StateVector must be an array, not a list.
        ComplexProbAmp = StateVector[0:self.Atom.Quantum.StateDimension] + 1.0j*StateVector[self.Atom.Quantum.StateDimension:(self.Atom.Quantum.StateDimension*2)] 
        dComplexProbAmp_dt = dot(self.Hamiltonian12HbarK(time, Omega10, Omega20), ComplexProbAmp)
        dStateVector_dt[0: self.Atom.Quantum.StateDimension]  = dComplexProbAmp_dt.real
        dStateVector_dt[self.Atom.Quantum.StateDimension:(self.Atom.Quantum.StateDimension*2)]  = dComplexProbAmp_dt.imag

        return dStateVector_dt
    
  
    def TimeEvolution(self):
        # assign values for initial states with filtering and create time period for integration,
        StateVector0 = self.Atom.Quantum.StateVec  * self.StateFilter[self.IntDex,:]
        time = np.array(linspace(self.PulseTimeStart, self.PulseTimeEnd, self.N)) 

        # add calculation for Rabi Frequencies before atom-laser-pulse interaction start by odeint
        self.AtomPosition_LocalRabiFrequencyUpdateAfterFlightOf(self.FlightDuration)
        StateVector = odeint(self.RHS_SchrodingerEq2, StateVector0, time, (self.Atom.Omega10, self.Atom.Omega20))
        self.Atom.Quantum.StateVec = StateVector[-1,:]
        #self.Atom.Quantum.StateVec[0:7]
             
        return (StateVector,time, self.Atom)
    

    def AtomPosition_LocalRabiFrequencyUpdateAfterFlightOf(self, FlightDuration):
        self.Atom.Position.UpdateDueTo(self.Atom.Velocity, FlightDuration)
        self.Atom.Time += FlightDuration
        # THSE: change the input arguments for RFspatialprofile
        Omega10 = self.RabiFrequencySpatialProfile(self.Atom.Position) 
        #print 'CCC Omega10, x, y, RabiF:', Omega10, self.Atom.Position.x*self.Atom.LengthUnit[0]*1000, self.Atom.Position.y*self.Atom.LengthUnit[0]*1000, self.Atom.RabiFreqmax
        self.Atom.Omega10 = Omega10
        self.Atom.Omega20 = Omega10
        
    """   
    def RabiFrequencySpatialProfile(self, radius):
        #the profile is determined by the laser beam profile
        return self.Atom.RabiFreqmax*np.exp(-(radius/self.Atom.BeamWaistAt1OverESquared)**2/2.0)
        """
    
    def RabiFrequencySpatialProfile(self, Position):
        #the profile is determined by the laser beam profile
        # THSE: add 2 more beams
        radius = sqrt(Position.x**2 + Position.y**2)
        if self.intrabeamDistance > 0.0:
            radius1 = sqrt(Position.x**2 + (Position.y -     self.intrabeamDistance)**2)
            radius2 = sqrt(Position.x**2 + (Position.y - 2 * self.intrabeamDistance)**2)
            RFS = self.Atom.RabiFreqmax*(np.exp(-(radius/self.Atom.BeamWaistAt1OverESquared)**2/2.0)+ \
                                         np.exp(-(radius1/self.Atom.BeamWaistAt1OverESquared)**2/2.0) + \
                                         np.exp(-(radius2/self.Atom.BeamWaistAt1OverESquared)**2/2.0))
            

        else:
            RFS = self.Atom.RabiFreqmax*(np.exp(-(radius/self.Atom.BeamWaistAt1OverESquared)**2/2.0))

        return RFS
    
        
    
class AtomInterferometer(object):
    
    def __init__(self, Atom = SingleAtom(Velocity=Velocity(vec_v = np.array([0.0, 0.0, 0.0]))) , Laser2PhoDetuningFactor = 1.0):
        self.SingleAtom = Atom
        self.TrajectoryRecorderInitialization()
        # THSE: add this factor parameter
        self.Laser2PhoDetuningFactor = Laser2PhoDetuningFactor
        #self.intraLaserBeamDistance = 0.0
            
    def AtomSourceRandomization(self):
        self.SingleAtom.RadomizePositionAndMomentumAccordingToTempAndCloudSize()
        
        
    def ShowAtomPositionAndMomentum(self):
        print self.SingleAtom.Position.vec#*self.LengthUnit[0]
        print self.SingleAtom.Velocity.vec#*self.LengthUnit[0]/self.TimeUnit[0]
       
    def TrajectoryRecorderInitialization(self):
        self.time = np.zeros(len(self.SingleAtom.Transitions))
        self.v = np.zeros(len(self.SingleAtom.Transitions))
        
    def UpperArmTransition(self): 
        SingleAtom = deepcopy(self.SingleAtom)
        StateStack = np.array([SingleAtom.Quantum.StateVec])
        TimeStack = np.array([[SingleAtom.Time]])
        for InteractionIndex in np.arange(len(SingleAtom.Transitions)):
            # THSE: add factor parameter
            LaserAtomInt = LaserAtomInteraction(InputAtom = SingleAtom, \
                                                InteractionPeriodIndex = InteractionIndex, ArmFilterIndex = 'upper', Laser2PhoDetuningFactor=self.Laser2PhoDetuningFactor)
            StateVector, Time, SingleAtom = LaserAtomInt.TimeEvolution() 
            StateStack = concatenate((StateStack, StateVector), axis=0)
            TimeStack  = concatenate((TimeStack, np.transpose(array([Time]))), axis=0)
        self.UpperArmStateStack = StateStack
        self.UpperArmTimeStack = TimeStack
        self.UpperArmPartialSingleAtom = SingleAtom
        self.intraLaserBeamDistance = LaserAtomInt.intrabeamDistance
        
        """
        print 'line 714'
        print 'check if upperstate stack last state is consistent with SingleAtom state vector'
        print self.UpperArmStateStack[-1,:]
        print 'uppperarmpartialSingleatom'
        print self.UpperArmPartialSingleAtom.Quantum.StateVec
        """        
        return SingleAtom
    
    def LowerArmTransition(self): 
        SingleAtom = deepcopy(self.SingleAtom)
        StateStack = np.array([SingleAtom.Quantum.StateVec])
        TimeStack = np.array([[SingleAtom.Time]])
        for InteractionIndex in np.arange(len(SingleAtom.Transitions)):
            # THSE: add factor parameter
            LaserAtomInt = LaserAtomInteraction(InputAtom = SingleAtom, \
                                                InteractionPeriodIndex = InteractionIndex, ArmFilterIndex = 'lower', Laser2PhoDetuningFactor=self.Laser2PhoDetuningFactor)     
            StateVector, Time, SingleAtom = LaserAtomInt.TimeEvolution() 
            StateStack = concatenate((StateStack, StateVector), axis=0)
            TimeStack  = concatenate((TimeStack, np.transpose(array([Time]))), axis=0)
        self.LowerArmStateStack = StateStack
        self.LowerArmTimeStack = TimeStack
        self.LowerArmPartialSingleAtom = SingleAtom
        self.intraLaserBeamDistance = LaserAtomInt.intrabeamDistance
        """
        print 'line 735'
        print 'check if lowerstate stack last state is consistent with SingleAtom state vector'
        print self.LowerArmStateStack[-1,:]
        print 'lowerarmpartialSingleatom'
        print self.LowerArmPartialSingleAtom.Quantum.StateVec
        """
        return SingleAtom   
        
    def ShowUpperArmTransitionSequence(self):
        StateStack = self.UpperArmStateStack
        TimeStack = self.UpperArmTimeStack
        Probability = np.abs(StateStack[:,0:self.SingleAtom.Quantum.StateDimension])**2 + np.abs(StateStack[:,self.SingleAtom.Quantum.StateDimension:(self.SingleAtom.Quantum.StateDimension*2)])**2
        Plot7Figs(TimeStack, Probability, 'title', 'xlabel', 'ylabel') 
        return self.SingleAtom
    
    def ShowLowerArmTransitionSequence(self):
        StateStack = self.LowerArmStateStack
        TimeStack = self.LowerArmTimeStack
        Probability = np.abs(StateStack[:,0:self.SingleAtom.Quantum.StateDimension])**2 + np.abs(StateStack[:,self.SingleAtom.Quantum.StateDimension:(self.SingleAtom.Quantum.StateDimension*2)])**2
        Plot7Figs(TimeStack, Probability, 'title', 'xlabel', 'ylabel')       
        return self.SingleAtom


    def SeparatedStateInterference(self):
        self.AtomSourceRandomization()
        #print 'line 761, ensable the atomsource randomization() above'
        #print 'atom poistion after randomization'
        #self.ShowAtomPositionAndMomentum()
        self.UpperArmTransition()
        #print 'atom position after upper interference'
        #self.ShowAtomPositionAndMomentum()
        self.LowerArmTransition()
        #print 'atom position after lower intereference'
        #self.ShowAtomPositionAndMomentum()
        """
        print 'atom position and velocity of lower arm'
        print 'positioin',self.LowerArmPartialSingleAtom.Position.vec
        print 'velcity', self.LowerArmPartialSingleAtom.Velocity.vec
        
        print 'atom position and velocity of upper arm'
        print 'positioin',self.UpperArmPartialSingleAtom.Position.vec
        print 'velcity', self.UpperArmPartialSingleAtom.Velocity.vec
        """
        """
        lowervec = self.LowerArmPartialSingleAtom.Quantum.StateVec 
        uppervec = self.UpperArmPartialSingleAtom.Quantum.StateVec
        Clower=self.SingleAtom.Quantum.ComplexVecConvertedFromStateVec(lowervec)
        Cupper=self.SingleAtom.Quantum.ComplexVecConvertedFromStateVec(uppervec)
        print 'line 784, total atom probability lower:', np.sum(np.abs(Clower)**2)
        print 'line 784, total atom probability upper:', np.sum(np.abs(Cupper)**2)
        print '\n'
        """
        return (self.LowerArmPartialSingleAtom.Quantum.StateVec, self.UpperArmPartialSingleAtom.Quantum.StateVec)
    
    def AtomSplitting(self):
        pass
    
    def AtomOutput(self):
        pass

def Plot10Figs(x, yarray, title, xlabel, ylabel):
    #show 7 figures in a row
    #close('all')
    f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(10, sharex=True, sharey=True)
    ax1.plot(x, yarray[:,0], 'k-' )
    ax1.grid(True)
    ax1.set_title(title)
    ax2.plot(x, yarray[:,1], 'g-' )
    ax2.grid(True)
    ax3.plot(x, yarray[:,2], 'b-' )
    ax3.grid(True)
    ax4.plot(x, yarray[:,3], 'r-' )
    ax4.grid(True)
    ax5.plot(x, yarray[:,4], 'm-' )
    ax5.grid(True)
    ax6.plot(x, yarray[:,5], 'y-' )
    ax6.grid(True)
    ax7.plot(x, yarray[:,6], 'c-' )
    ax8.plot(x, yarray[:,7], 'm-' )
    ax9.plot(x, yarray[:,8], 'y-' )
    ax10.plot(x, yarray[:,9], 'c-' )
    
    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    show()

def Plot7Figs(x, yarray, title, xlabel, ylabel):
    #show 7 figures in a row
    #close('all')
    f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, sharex=True, sharey=True)
    ax1.plot(x, yarray[:,0], 'k-' )
    ax1.grid(True)
    ax1.set_title(title)
    ax2.plot(x, yarray[:,1], 'g-' )
    ax2.grid(True)
    ax3.plot(x, yarray[:,2], 'b-' )
    ax3.grid(True)
    ax4.plot(x, yarray[:,3], 'r-' )
    ax4.grid(True)
    ax5.plot(x, yarray[:,4], 'm-' )
    ax5.grid(True)
    ax6.plot(x, yarray[:,5], 'y-' )
    ax6.grid(True)
    ax7.plot(x, yarray[:,6], 'c-' )
    
    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    show()
    
def TestPlot7Figs():
    x=linspace(1,10,100)
    y=np.random.rand(100,7)
    for ii in range(10):
        print y[ii,:]
    Plot7Figs(x,y,'testplot','xlabel','ylabel')
    
    
def PhaseShifting(State, phi):
    StateShifted = State * exp(1.0j*phi)
    return StateShifted
    

def InterferenceFringes(StateProbAmpUpper,StateProbAmpLower, StateIndex, plotswitch):
    Upper10hbarkeff = StateProbAmpUpper[StateIndex] + StateProbAmpUpper[StateIndex+7]*1.0j
    Lower10hbarkeff = StateProbAmpLower[StateIndex] + StateProbAmpLower[StateIndex+7]*1.0j
    # THSE: a phase gradient was assumed. this needs to be changed.
    phi=1.0*arange(101)/100*4*pi
    UpperShifted = PhaseShifting(Upper10hbarkeff, 0.0*phi)
    LowerShifted = PhaseShifting(Lower10hbarkeff, phi)
    Fringes= abs(UpperShifted + LowerShifted)**2
    if plotswitch == True:
        close('all')
        figure()
        plot(phi, Fringes)
        show()
    return Fringes
    
class FringeObservation(object):
    def __init__(self, SingleAtom = SingleAtom(), FilteringFraction=1.0, NumberOfFringesToSee=2, AtomInterferometer = AtomInterferometer()):
        self.SingleAtom = SingleAtom
        self.PhaseGradient = NumberOfFringesToSee*8.0*np.pi/(self.SingleAtom.BeamWaistAt1OverESquared*2)
        self.Phase0 = np.pi/2.0
        self.FilterDiameter = FilteringFraction * self.SingleAtom.BeamWaistAt1OverESquared
        self.intraBeamDistance = AtomInterferometer.intraLaserBeamDistance
        
    def AtomFiltering(self, AtomDataFSet):
        FilteringIndex = (AtomDataFSet['x']**2 + AtomDataFSet['y']**2) < self.FilterDiameter**2 
        self.TotalNumberOfValids = np.sum(FilteringIndex)
        AtomDataSetAfterFiltering = AtomDataFSet.ix[FilteringIndex]
        return AtomDataSetAfterFiltering
    
    def AtomFilteringDisabled(self,AtomDataFSet):
        # THSE: disable the filtering by increasing the filterDiameter to 1000000 larger
        FilteringIndex = (AtomDataFSet['x']**2 + AtomDataFSet['y']**2) < (self.FilterDiameter*1000000)**2 
        self.TotalNumberOfValids = np.sum(FilteringIndex)
        AtomDataSetAfterFiltering = AtomDataFSet.ix[FilteringIndex]
        return AtomDataSetAfterFiltering
        
    
    def AddPhaseShiftToLowerArm(self, AtomDataFSet):
        StateVecSet={}
        
        #print 'line 892\n'
        #print 'input atom data set'
        #print AtomDataFSet
        
        AtomI = 0
        for StateVec, y in zip(AtomDataFSet['StateLower'], AtomDataFSet['y']):
            StateVecSet[AtomI] = pd.Series({'StateLower': self.PhaseShift(StateVec, y)})
            AtomI +=1
        StateDataSet = pd.DataFrame(StateVecSet).T
        StateDataSet.index = AtomDataFSet.index      
        #print '902'
        #print 'StateDataSet', StateDataSet
        AtomDataFSet['StateLowerShifted'] = StateDataSet
        """
        print 'below is data after phase shift, check if probability is conserved'
        print 'state lower'
        print AtomDataFSet['StateLower'].values
        print 'state upper'
        print AtomDataFSet['StateUpper'].values
        print 'state lower shifted'
        print AtomDataFSet['StateLowerShifted'].values
        """

        del AtomDataFSet['StateLower']
        return AtomDataFSet

    def PhaseShift(self, InputStateVector, y):
        #multiply a phase variation along y direction
        CV = self.SingleAtom.Quantum.ComplexVecConvertedFromStateVec(InputStateVector)
        CV_PShifted = CV*np.exp(1.0j*(y*self.PhaseGradient+self.Phase0))
        OutputStateVector = self.SingleAtom.Quantum.StateVecConvertedFromComplexVec(CV_PShifted)
        return OutputStateVector
    
    def ProbabilityAfterSuperpositionUpperAndLower(self, AtomDataSet):
        AtomNewDataSet = {}
        AtomDataSetArray= np.array([[]])
        AtomDataSety = np.array([[]])
        AtomI = 0
        for StateUpperVec, StateLowerVec, y in zip(AtomDataSet['StateUpper'], AtomDataSet['StateLowerShifted'], AtomDataSet['y']):
            StateUpper = self.SingleAtom.Quantum.ComplexVecConvertedFromStateVec(StateUpperVec)
            StateLower = self.SingleAtom.Quantum.ComplexVecConvertedFromStateVec(StateLowerVec)
            StateSuperposed = StateUpper + StateLower
            StateProbability = np.abs(StateSuperposed)**2
            if AtomI ==0:
                AtomDataSetArray = np.array([StateProbability])
                AtomDataSety = np.array([[y]])
            else:
                AtomDataSetArray = np.concatenate((AtomDataSetArray, np.array([StateProbability])), axis=0)
                AtomDataSety = np.concatenate((AtomDataSety, np.array([[y]])), axis=0)
            
            AtomNewDataSet[AtomI] = deepcopy(pd.Series({'StateProbability': StateProbability}))
            AtomI +=1
        #print AtomDataSetArray
        columnnames = ['y'] + list(self.SingleAtom.Quantum.States.index)
        StateDataSetArray = pd.DataFrame(np.concatenate((AtomDataSety, AtomDataSetArray), axis =1), columns = columnnames)
        
        StateDataSet = pd.DataFrame(AtomNewDataSet).T
        StateDataSet.index = AtomDataSet.index
        AtomDataSet['StateProbability'] = StateDataSet
        del AtomDataSet['StateUpper']
        del AtomDataSet['StateLowerShifted']
        del AtomDataSet['x']
        del AtomDataSet['z']
        return StateDataSetArray
 
    
    def FringeBinningAlongYaxis(self, AtomDataSet, AtomStateSelectedForObservation='|1, 0hbarkeff>', NumberOfYAxisBins=200):
        #generate y coordinate for cutting y axis into bins
        # THSE: change the ycut range, note DDDD
        ycut = np.linspace(-self.SingleAtom.BeamWaistAt1OverESquared + 2*self.intraBeamDistance, self.SingleAtom.BeamWaistAt1OverESquared+ 2*self.intraBeamDistance, NumberOfYAxisBins)
        #ycut = np.linspace(-self.SingleAtom.BeamWaistAt1OverESquared, self.SingleAtom.BeamWaistAt1OverESquared, NumberOfYAxisBins)
        #do the cut and generate bins to the data set, question what the data set start with is?
        #print ' from line 921'
        #print 'atom data set to start with\n', AtomDataSet
        yBinLabels = pd.cut(AtomDataSet['y'], ycut)
        yBinLabels.index = AtomDataSet.index
        ##print 'ybinlabels', yBinLabels
        AtomDataSet['yBins'] = yBinLabels
        Prob= AtomDataSet.groupby('yBins')
        ##print 'line 968'
        ##print 'data for probability in bins: \n', Prob
        ProbMean = Prob[AtomStateSelectedForObservation].mean()
        ProbStd = Prob[AtomStateSelectedForObservation].std()
        ProbStd.name = 'ProbStd'
        ##print 'groups of y'
        ##print Prob['y']
        PositionyMean = Prob['y'].mean()
        NumberAtoms = Prob['y'].count()
        ##print 'line 975'
        print 'number of atoms'
        print NumberAtoms
        ##print 'PositionyMean'
        ##print PositionyMean
        NumberAtoms.name = 'AtomNumber'
        
        ProbVsPosy = pd.concat([PositionyMean, ProbMean, ProbStd], axis=1).sort('y')
        #AtomNumberVsPosy = pd.concat([PositionyMean, NumberAtoms], axis=1).sort('y')
        AtomNumberVsFringeVsPosy = pd.concat([PositionyMean, NumberAtoms, ProbMean, ProbStd], axis=1).sort('y')
        ##print 'probvsposiy and number of atoms', ProbVsPosy
        #plot(ProbVsPosy['y'], ProbVsPosy['3'],'o')
        #show()
        return (ProbVsPosy, AtomNumberVsFringeVsPosy)
    
    def FringeBinningAlongYaxis2(self, AtomDataSet, AtomStateSelectedForObservation='|1, 0hbarkeff>', NumberOfYAxisBins=200):
        # THSE: this is the original method for binning. 
        #generate y coordinate for cutting y axis into bins
        ycut = np.linspace(-self.SingleAtom.BeamWaistAt1OverESquared, self.SingleAtom.BeamWaistAt1OverESquared, NumberOfYAxisBins)
        #do the cut and generate bins to the data set, question what the data set start with is?
        #print ' from line 921'
        #print 'atom data set to start with\n', AtomDataSet
        yBinLabels = pd.cut(AtomDataSet['y'], ycut)
        yBinLabels.index = AtomDataSet.index
        ##print 'ybinlabels', yBinLabels
        AtomDataSet['yBins'] = yBinLabels
        Prob= AtomDataSet.groupby('yBins')
        ##print 'line 968'
        ##print 'data for probability in bins: \n', Prob
        ProbMean = Prob[AtomStateSelectedForObservation].mean()
        ProbStd = Prob[AtomStateSelectedForObservation].std()
        ProbStd.name = 'ProbStd'
        ##print 'groups of y'
        ##print Prob['y']
        PositionyMean = Prob['y'].mean()
        NumberAtoms = Prob['y'].count()
        ##print 'line 975'
        print 'number of atoms'
        print NumberAtoms
        ##print 'PositionyMean'
        ##print PositionyMean
        NumberAtoms.name = 'AtomNumber'
        
        ProbVsPosy = pd.concat([PositionyMean, ProbMean, ProbStd], axis=1).sort('y')
        #AtomNumberVsPosy = pd.concat([PositionyMean, NumberAtoms], axis=1).sort('y')
        AtomNumberVsFringeVsPosy = pd.concat([PositionyMean, NumberAtoms, ProbMean, ProbStd], axis=1).sort('y')
        ##print 'probvsposiy and number of atoms', ProbVsPosy
        #plot(ProbVsPosy['y'], ProbVsPosy['3'],'o')
        #show()
        return (ProbVsPosy, AtomNumberVsFringeVsPosy)
    
    def FringeContrast(self, AtomDataSet, AtomStateSelectedForObservation='|1, 0hbarkeff>'):
        FringeContrast = AtomDataSet[AtomStateSelectedForObservation].max() - AtomDataSet[AtomStateSelectedForObservation].min()
        return FringeContrast
    
    def ContrastExtractionUnit(self, AtomDataSet, StateSelectedForObservation='|1, 0hbarkeff>', NumberOfBins=200 ):
        #print 'line 955'
        #print 'input atom data set, check if the lower and upper mistaken as the same data:'
        #print AtomDataSet
        AtomDataFFSet = self.AtomFiltering(AtomDataSet)
        AtomData = self.AddPhaseShiftToLowerArm(AtomDataFFSet)
        AtomProb = self.ProbabilityAfterSuperpositionUpperAndLower(AtomData)
        AtomFringes, AtomNumberVsFringeVsPosy = self.FringeBinningAlongYaxis(AtomProb, AtomStateSelectedForObservation=StateSelectedForObservation, NumberOfYAxisBins=NumberOfBins)
        FringeContrast = self.FringeContrast(AtomFringes, AtomStateSelectedForObservation=StateSelectedForObservation)
        return (FringeContrast, AtomFringes, AtomNumberVsFringeVsPosy)
    
    def ContrastExtractionUnitWithoutFiltering(self, AtomDataSet, StateSelectedForObservation='|1, 0hbarkeff>', NumberOfBins=200 ):
        #print 'line 955'
        #print 'input atom data set, check if the lower and upper mistaken as the same data:'
        #print AtomDataSet
        # THSE: new create, disable the filtering
        AtomDataFFSet = self.AtomFilteringDisabled(AtomDataSet)
        AtomData = self.AddPhaseShiftToLowerArm(AtomDataFFSet)
        AtomProb = self.ProbabilityAfterSuperpositionUpperAndLower(AtomData)
        AtomFringes, AtomNumberVsFringeVsPosy = self.FringeBinningAlongYaxis(AtomProb, AtomStateSelectedForObservation=StateSelectedForObservation, NumberOfYAxisBins=NumberOfBins)
        FringeContrast = self.FringeContrast(AtomFringes, AtomStateSelectedForObservation=StateSelectedForObservation)
        return (FringeContrast, AtomFringes, AtomNumberVsFringeVsPosy)
                   
        
        
class FringeObservationH(object):
    def __init__(self, PhaseShift = np.pi/2.0, SingleAtom = SingleAtom(), FilteringFraction=1.0, NumberOfFringesToSee=2, AtomInterferometer = AtomInterferometer()):
        # THSE: set phasegradient to 0 and add  phase0 input argument
        self.SingleAtom = SingleAtom
        self.PhaseGradient = 0.0
        self.Phase0 = PhaseShift
        self.FilterDiameter = FilteringFraction * self.SingleAtom.BeamWaistAt1OverESquared
        self.intraBeamDistance = AtomInterferometer.intraLaserBeamDistance
        
    def AtomFiltering(self, AtomDataFSet):
        FilteringIndex = (AtomDataFSet['x']**2 + AtomDataFSet['y']**2) < self.FilterDiameter**2 
        self.TotalNumberOfValids = np.sum(FilteringIndex)
        AtomDataSetAfterFiltering = AtomDataFSet.ix[FilteringIndex]
        return AtomDataSetAfterFiltering
    
    def AtomFilteringDisabled(self,AtomDataFSet):
        # THSE: disable the filtering by increasing the filterDiameter to 1000000 larger
        FilteringIndex = (AtomDataFSet['x']**2 + AtomDataFSet['y']**2) < (self.FilterDiameter*1000000)**2 
        self.TotalNumberOfValids = np.sum(FilteringIndex)
        AtomDataSetAfterFiltering = AtomDataFSet.ix[FilteringIndex]
        return AtomDataSetAfterFiltering
        
    
    def AddPhaseShiftToLowerArm(self, AtomDataFSet):
        StateVecSet={}
        
        #print 'line 892\n'
        #print 'input atom data set'
        #print AtomDataFSet
        
        AtomI = 0
        for StateVec, y in zip(AtomDataFSet['StateLower'], AtomDataFSet['y']):
            StateVecSet[AtomI] = pd.Series({'StateLower': self.PhaseShift(StateVec, y)})
            AtomI +=1
        StateDataSet = pd.DataFrame(StateVecSet).T
        StateDataSet.index = AtomDataFSet.index      
        #print '902'
        #print 'StateDataSet', StateDataSet
        AtomDataFSet['StateLowerShifted'] = StateDataSet
        """
        print 'below is data after phase shift, check if probability is conserved'
        print 'state lower'
        print AtomDataFSet['StateLower'].values
        print 'state upper'
        print AtomDataFSet['StateUpper'].values
        print 'state lower shifted'
        print AtomDataFSet['StateLowerShifted'].values
        """

        del AtomDataFSet['StateLower']
        return AtomDataFSet

    def PhaseShift(self, InputStateVector, y):
        #multiply a phase variation along y direction
        CV = self.SingleAtom.Quantum.ComplexVecConvertedFromStateVec(InputStateVector)
        CV_PShifted = CV*np.exp(1.0j*(y*self.PhaseGradient+self.Phase0))
        OutputStateVector = self.SingleAtom.Quantum.StateVecConvertedFromComplexVec(CV_PShifted)
        return OutputStateVector
    
    def ProbabilityAfterSuperpositionUpperAndLower(self, AtomDataSet):
        AtomNewDataSet = {}
        AtomDataSetArray= np.array([[]])
        AtomDataSety = np.array([[]])
        AtomI = 0
        for StateUpperVec, StateLowerVec, y in zip(AtomDataSet['StateUpper'], AtomDataSet['StateLowerShifted'], AtomDataSet['y']):
            StateUpper = self.SingleAtom.Quantum.ComplexVecConvertedFromStateVec(StateUpperVec)
            StateLower = self.SingleAtom.Quantum.ComplexVecConvertedFromStateVec(StateLowerVec)
            StateSuperposed = StateUpper + StateLower
            StateProbability = np.abs(StateSuperposed)**2
            if AtomI ==0:
                AtomDataSetArray = np.array([StateProbability])
                AtomDataSety = np.array([[y]])
            else:
                AtomDataSetArray = np.concatenate((AtomDataSetArray, np.array([StateProbability])), axis=0)
                AtomDataSety = np.concatenate((AtomDataSety, np.array([[y]])), axis=0)
            
            AtomNewDataSet[AtomI] = deepcopy(pd.Series({'StateProbability': StateProbability}))
            AtomI +=1
        #print AtomDataSetArray
        columnnames = ['y'] + list(self.SingleAtom.Quantum.States.index)
        StateDataSetArray = pd.DataFrame(np.concatenate((AtomDataSety, AtomDataSetArray), axis =1), columns = columnnames)
        
        StateDataSet = pd.DataFrame(AtomNewDataSet).T
        StateDataSet.index = AtomDataSet.index
        AtomDataSet['StateProbability'] = StateDataSet
        del AtomDataSet['StateUpper']
        del AtomDataSet['StateLowerShifted']
        del AtomDataSet['x']
        del AtomDataSet['z']
        return StateDataSetArray
 
    
    def FringeBinningAlongYaxis(self, AtomDataSet, AtomStateSelectedForObservation='|1, 0hbarkeff>', NumberOfYAxisBins=1):
        #generate y coordinate for cutting y axis into bins
        # THSE: change the ycut range, note DDDD
        ycut = np.linspace(-self.SingleAtom.BeamWaistAt1OverESquared + 2*self.intraBeamDistance, self.SingleAtom.BeamWaistAt1OverESquared+ 2*self.intraBeamDistance, NumberOfYAxisBins)
        #ycut = np.linspace(-self.SingleAtom.BeamWaistAt1OverESquared, self.SingleAtom.BeamWaistAt1OverESquared, NumberOfYAxisBins)
        #do the cut and generate bins to the data set, question what the data set start with is?
        #print ' from line 921'
        #print 'atom data set to start with\n', AtomDataSet
        yBinLabels = pd.cut(AtomDataSet['y'], ycut)
        yBinLabels.index = AtomDataSet.index
        ##print 'ybinlabels', yBinLabels
        AtomDataSet['yBins'] = yBinLabels
        #print 'Atom Data Set\n', AtomDataSet['yBins'] 
        Prob= AtomDataSet.groupby('yBins')
        ##print 'line 968'
        #print 'data for probability in bins:, groups \n', Prob.groups
        #print 'data for probability in bins for the observing state: \n', Prob[AtomStateSelectedForObservation].groups
        ProbMean = Prob[AtomStateSelectedForObservation].mean()
        ProbStd = Prob[AtomStateSelectedForObservation].std()
        ProbStd.name = 'ProbStd'
        ##print 'groups of y'
        ##print Prob['y']
        PositionyMean = Prob['y'].mean()
        NumberAtoms = Prob['y'].count()
        ##print 'line 975'
        #print 'number of atoms'
        #print NumberAtoms
        ##print 'PositionyMean'
        ##print PositionyMean
        NumberAtoms.name = 'AtomNumber'
        
        ProbVsPosy = pd.concat([PositionyMean, ProbMean, ProbStd], axis=1).sort('y')
        #AtomNumberVsPosy = pd.concat([PositionyMean, NumberAtoms], axis=1).sort('y')
        AtomNumberVsFringeVsPosy = pd.concat([PositionyMean, NumberAtoms, ProbMean, ProbStd], axis=1).sort('y')
        ##print 'probvsposiy and number of atoms', ProbVsPosy
        #plot(ProbVsPosy['y'], ProbVsPosy['3'],'o')
        #show()
        #print 'AtomNumber Vs Fring Vs Posy\n', AtomNumberVsFringeVsPosy 
        return (ProbVsPosy, AtomNumberVsFringeVsPosy)
    
    def FringeBinningAlongYaxis2(self, AtomDataSet, AtomStateSelectedForObservation='|1, 0hbarkeff>', NumberOfYAxisBins=200):
        # THSE: this is the original method for binning. 
        #generate y coordinate for cutting y axis into bins
        ycut = np.linspace(-self.SingleAtom.BeamWaistAt1OverESquared, self.SingleAtom.BeamWaistAt1OverESquared, NumberOfYAxisBins)
        #do the cut and generate bins to the data set, question what the data set start with is?
        #print ' from line 921'
        #print 'atom data set to start with\n', AtomDataSet
        yBinLabels = pd.cut(AtomDataSet['y'], ycut)
        yBinLabels.index = AtomDataSet.index
        ##print 'ybinlabels', yBinLabels
        AtomDataSet['yBins'] = yBinLabels
        Prob= AtomDataSet.groupby('yBins')
        ##print 'line 968'
        print 'data for probability in bins, groups: \n', Prob
        ProbMean = Prob[AtomStateSelectedForObservation].mean()
        ProbStd = Prob[AtomStateSelectedForObservation].std()
        ProbStd.name = 'ProbStd'
        ##print 'groups of y'
        ##print Prob['y']
        PositionyMean = Prob['y'].mean()
        NumberAtoms = Prob['y'].count()
        ##print 'line 975'
        print 'number of atoms'
        print NumberAtoms
        ##print 'PositionyMean'
        ##print PositionyMean
        NumberAtoms.name = 'AtomNumber'
        
        ProbVsPosy = pd.concat([PositionyMean, ProbMean, ProbStd], axis=1).sort('y')
        #AtomNumberVsPosy = pd.concat([PositionyMean, NumberAtoms], axis=1).sort('y')
        AtomNumberVsFringeVsPosy = pd.concat([PositionyMean, NumberAtoms, ProbMean, ProbStd], axis=1).sort('y')
        ##print 'probvsposiy and number of atoms', ProbVsPosy
        #plot(ProbVsPosy['y'], ProbVsPosy['3'],'o')
        #show()
        return (ProbVsPosy, AtomNumberVsFringeVsPosy) # note: these 2 outputs are quite similar
    
    def FringeContrast(self, AtomDataSet, AtomStateSelectedForObservation='|1, 0hbarkeff>'):
        FringeContrast = AtomDataSet[AtomStateSelectedForObservation].max() - AtomDataSet[AtomStateSelectedForObservation].min()
        return FringeContrast
    
    def ContrastExtractionUnit(self, AtomDataSet, StateSelectedForObservation='|1, 0hbarkeff>', NumberOfBins=1 ):
        #print 'line 955'
        #print 'input atom data set, check if the lower and upper mistaken as the same data:'
        #print AtomDataSet
        AtomDataFFSet = self.AtomFiltering(AtomDataSet)
        AtomData = self.AddPhaseShiftToLowerArm(AtomDataFFSet)
        AtomProb = self.ProbabilityAfterSuperpositionUpperAndLower(AtomData)
        AtomFringes, AtomNumberVsFringeVsPosy = self.FringeBinningAlongYaxis(AtomProb, AtomStateSelectedForObservation=StateSelectedForObservation, NumberOfYAxisBins=NumberOfBins)
        FringeContrast = self.FringeContrast(AtomFringes, AtomStateSelectedForObservation=StateSelectedForObservation)
        return (FringeContrast, AtomFringes, AtomNumberVsFringeVsPosy)
    
    def ContrastExtractionUnitWithoutFiltering(self, AtomDataSet, StateSelectedForObservation='|1, 0hbarkeff>', NumberOfBins=1 ):
        #print 'line 955'
        #print 'input atom data set, check if the lower and upper mistaken as the same data:'
        #print AtomDataSet
        # THSE: new create, disable the filtering
        AtomDataFFSet = self.AtomFilteringDisabled(AtomDataSet)
        AtomData = self.AddPhaseShiftToLowerArm(AtomDataFFSet)
        AtomProb = self.ProbabilityAfterSuperpositionUpperAndLower(AtomData)
        AtomFringes, AtomNumberVsFringeVsPosy = self.FringeBinningAlongYaxis(AtomProb, AtomStateSelectedForObservation=StateSelectedForObservation, NumberOfYAxisBins=NumberOfBins)
        FringeContrast = self.FringeContrast(AtomFringes, AtomStateSelectedForObservation=StateSelectedForObservation)
        
        #print 'Atom Fringes are below:\n', AtomFringes 
        return (FringeContrast, AtomFringes, AtomNumberVsFringeVsPosy)
        
        
        
class AtomInterferometerAnalysis(object):
    def __init__():
        self.AtomInterferometerDefinition
        self.AtomInterferometerAnalysisDefinition
        pass
    
    def specificAnalysisDefinitiion():
        pass
    
    


def MonteCarloStatisticsAtomInterferometry(AtomNumber=100, FileName='AIAnalysis0', AtomCloudTemp = (8.0, 'muK'),  PiPulseDuration=(60,'mus'), InterogationTime=(1,'s')):
    
    #Define Number of Quantum Momentum States
    QStates=QuantumAtomicInter_ExternalCoupledStates()
    #QStates.InCaseOf12hbark()
    
    #Define type of Atoms and associated with Quantum States
    SingleRbOf6hbark = SingleAtom(QuatumStates=QStates)
    SingleRbOf6hbark.InCaseOf6hbark()
    
    #Parameter Modification:
    #Define Temperature etc Atom cloud parameters
    SingleRbOf6hbark.kbTk = SingleRbOf6hbark.kbTk3nK /3 * AtomCloudTemp[0]
    print SingleRbOf6hbark.kbTk
    
    print 'pipulse', SingleRbOf6hbark.pipulse
    SingleRbOf6hbark.pipulse = PiPulseDuration[0] *1.0e-6 / SingleRbOf6hbark.TimeUnit[0]
    print 'pipulse after', SingleRbOf6hbark.pipulse
    
    print '2 x interogation time before modification',  SingleRbOf6hbark.TimeUnit[0] *  SingleRbOf6hbark.pulsetiming[-1] 
    SingleRbOf6hbark.interpulse = InterogationTime[0] * 2.0 / SingleRbOf6hbark.TimeUnit[0] / (SingleRbOf6hbark.pulsetiming[-1]/SingleRbOf6hbark.interpulse) 
    print 'interpulse after', SingleRbOf6hbark.interpulse
    #Parameter Update
    print 'Rabifrequency before changes', SingleRbOf6hbark.RabiFreqmax
    SingleRbOf6hbark.InCaseOf6hbark()
    print '2 x interogation time after modification',  SingleRbOf6hbark.TimeUnit[0] *  SingleRbOf6hbark.pulsetiming[-1] 
    print 'Rabifrequency', SingleRbOf6hbark.RabiFreqmax
    SingleRb = SingleRbOf6hbark
   
    
    #Define Atom Interferometer by using the above Atom and Quantum State definitions              
    IF1 = AtomInterferometer( Atom = SingleRb)
    #Define a Set for Atom Collection
    AtomSet = {}
    AtomSet0 =np.array([[]])

    #starting with a randomly generated single atom
    for AtomI in range(AtomNumber):
        #print 'Atom Index:', AtomI
        StateProbAmpLower, StateProbAmpUpper = IF1.SeparatedStateInterference()
        #Accumulate and assemble a Data Frame for every single atom interference information
        AtomSet[AtomI] = deepcopy(pd.Series({'x':IF1.UpperArmPartialSingleAtom.Position.x, 'y':IF1.UpperArmPartialSingleAtom.Position.y, 'z':IF1.UpperArmPartialSingleAtom.Position.z, 'StateLower': StateProbAmpLower,  'StateUpper': StateProbAmpUpper}))       
        AtomSet01=np.atleast_2d([[IF1.UpperArmPartialSingleAtom.Position.x,] + [IF1.UpperArmPartialSingleAtom.Position.y,] + [IF1.UpperArmPartialSingleAtom.Position.z,] + list(StateProbAmpLower) + list(StateProbAmpUpper) ])
        if AtomI ==0:
            AtomSet0 = AtomSet01
        else:       
            AtomSet0 = np.concatenate((AtomSet0, AtomSet01), axis=0)
    AtomDataSet = pd.DataFrame(AtomSet).T
    #print 'line 1065, show atom dataset and check the c labels'
    #print AtomDataSet
    ObFilter1 = FringeObservation(SingleAtom = SingleRb)
    FringeContrast, AtomFringes, AtomNumberVsFringeVsPosy = ObFilter1.ContrastExtractionUnit(AtomDataSet, StateSelectedForObservation='|1, 0hbarkeff>', NumberOfBins=200)
    
    print 'fringe contast\n', FringeContrast
    
    print 'AtomFringes\n', AtomFringes
       
    print 'time complete', tmstamp.time()
    
    close()
    plot(AtomFringes['y']*SingleCs.LengthUnit[0]*1000, AtomFringes['|1, 0hbarkeff>'],'o')
    plot(AtomNumberVsFringeVsPosy['y']*SingleCs.LengthUnit[0]*1000, AtomNumberVsFringeVsPosy['AtomNumber'],'ro')
    xlabel('Position (mm)')

    ylabel('AtomNumber & Fringe (arb.)')
    show()
    

    StateNames = list(SingleRb.Quantum.States.index) * 4 
    columnnames=['x', 'y', 'z'] + StateNames
    StatePAStackDF = pd.DataFrame(AtomSet0, columns=columnnames)
    StatePAStackDF.to_csv(FileName)
    return (StatePAStackDF, AtomFringes, FringeContrast, AtomNumberVsFringeVsPosy, SingleRb)

def MonteCarloStatisticsAtomInterferometry12hbark(AtomNumber=100, FileName='AIAnalysis0', AtomCloudTemp = (8.0, 'muK'),  PiPulseDuration=(60,'mus'), InterogationTime=(1,'s')):
    
    #Define Number of Quantum Momentum States
    QStates=QuantumAtomicInter_ExternalCoupledStates()
    QStates.InCaseOf12hbark()
    
    #Define type of Atoms and associated with Quantum States
    SingleRbOf12hbark = SingleAtom(QuatumStates=QStates)
    SingleRbOf12hbark.InCaseOf12hbark()
    
    #Parameter Modification:
    #Define Temperature etc Atom cloud parameters
    SingleRbOf12hbark.kbTk = SingleRbOf12hbark.kbTk3nK /3 * AtomCloudTemp[0]
    print SingleRbOf12hbark.kbTk
    
    print 'pipulse', SingleRbOf12hbark.pipulse
    SingleRbOf12hbark.pipulse = PiPulseDuration[0] *1.0e-6 / SingleRbOf12hbark.TimeUnit[0]
    print 'pipulse after', SingleRbOf12hbark.pipulse
    
    print '2 x interogation time before modification',  SingleRbOf12hbark.TimeUnit[0] *  SingleRbOf12hbark.pulsetiming[-1] 
    SingleRbOf12hbark.interpulse = InterogationTime[0] * 2.0 / SingleRbOf12hbark.TimeUnit[0] / (SingleRbOf12hbark.pulsetiming[-1]/SingleRbOf12hbark.interpulse) 
    print 'interpulse after', SingleRbOf12hbark.interpulse
    #Parameter Update
    print 'Rabifrequency before changes', SingleRbOf12hbark.RabiFreqmax
    SingleRbOf12hbark.InCaseOf12hbark()
    print '2 x interogation time after modification',  SingleRbOf12hbark.TimeUnit[0] *  SingleRbOf12hbark.pulsetiming[-1] 
    print 'Rabifrequency', SingleRbOf12hbark.RabiFreqmax
    SingleRb = SingleRbOf12hbark
   
    
    #Define Atom Interferometer by using the above Atom and Quantum State definitions              
    IF1 = AtomInterferometer( Atom = SingleRb)
    #Define a Set for Atom Collection
    AtomSet = {}
    AtomSet0 =np.array([[]])

    #starting with a randomly generated single atom
    for AtomI in range(AtomNumber):
        #print 'Atom Index:', AtomI
        StateProbAmpLower, StateProbAmpUpper = IF1.SeparatedStateInterference()
        #Accumulate and assemble a Data Frame for every single atom interference information
        AtomSet[AtomI] = deepcopy(pd.Series({'x':IF1.UpperArmPartialSingleAtom.Position.x, 'y':IF1.UpperArmPartialSingleAtom.Position.y, 'z':IF1.UpperArmPartialSingleAtom.Position.z, 'StateLower': StateProbAmpLower,  'StateUpper': StateProbAmpUpper}))       
        AtomSet01=np.atleast_2d([[IF1.UpperArmPartialSingleAtom.Position.x,] + [IF1.UpperArmPartialSingleAtom.Position.y,] + [IF1.UpperArmPartialSingleAtom.Position.z,] + list(StateProbAmpLower) + list(StateProbAmpUpper) ])
        if AtomI ==0:
            AtomSet0 = AtomSet01
        else:       
            AtomSet0 = np.concatenate((AtomSet0, AtomSet01), axis=0)
    AtomDataSet = pd.DataFrame(AtomSet).T
    #print 'line 1065, show atom dataset and check the c labels'
    #print AtomDataSet
    ObFilter1 = FringeObservation(SingleAtom = SingleRb)
    FringeContrast, AtomFringes, AtomNumberVsFringeVsPosy = ObFilter1.ContrastExtractionUnit(AtomDataSet, StateSelectedForObservation='|1, 0hbarkeff>', NumberOfBins=2000000)
    
    print 'fringe contast\n', FringeContrast
    
    print 'AtomFringes\n', AtomFringes
       
    print 'time complete', tmstamp.time()
    
    close()

    
    
    #plot(AtomFringes['y']*SingleRb.LengthUnit[0]*1000, AtomFringes['|1, 0hbarkeff>'],'o')
    #plot(AtomNumberVsFringeVsPosy['y']*SingleRb.LengthUnit[0]*1000, AtomNumberVsFringeVsPosy['AtomNumber'],'ro')
    plot(AtomFringes['y']*SingleCs.LengthUnit[0]*1000, AtomFringes['|1, 0hbarkeff>'],'o')
    plot(AtomNumberVsFringeVsPosy['y']*SingleCs.LengthUnit[0]*1000, AtomNumberVsFringeVsPosy['AtomNumber'],'ro')
    xlabel('Position (mm)')
    ylabel('AtomNumber & Fringe (arb.)')
    show()
    

    StateNames = list(SingleRb.Quantum.States.index) * 4 
    columnnames=['x', 'y', 'z'] + StateNames
    StatePAStackDF = pd.DataFrame(AtomSet0, columns=columnnames)
    StatePAStackDF.to_csv(FileName)
    return (StatePAStackDF, AtomFringes, FringeContrast, AtomNumberVsFringeVsPosy, SingleRb)

def MonteCarloStatisticsAtomInterferometryCs6hbark(AtomNumber=100, FileName='AIAnalysis0', AtomCloudTemp = (3.0, 'nK'),  PiPulseDuration=(60,'mus'), InterogationTime=(1,'s')):
    
    #Define Number of Quantum Momentum States
    QStates=QuantumAtomicInter_ExternalCoupledStates()
    #QStates.InCaseOf12hbark()
    
    
    #Define type of Atoms and associated with Quantum States
    SingleCsOf6hbark = SingleAtom(QuatumStates=QStates, AtomDefinition=CommonElementList().Cs133)
    SingleCsOf6hbark.InCaseOf6hbark()
    
    #Parameter Modification:
    #Define Temperature etc Atom cloud parameters
    SingleCsOf6hbark.kbTk = SingleCsOf6hbark.kbTk3nK /3 * AtomCloudTemp[0]
    print SingleCsOf6hbark.kbTk
    
    print 'pipulse', SingleCsOf6hbark.pipulse
    SingleCsOf6hbark.pipulse = PiPulseDuration[0] *1.0e-6 / SingleCsOf6hbark.TimeUnit[0]
    print 'pipulse after', SingleCsOf6hbark.pipulse
     
    print '2 x interogation time before modification',  SingleCsOf6hbark.TimeUnit[0] *  SingleCsOf6hbark.pulsetiming[-1] 
    SingleCsOf6hbark.interpulse = InterogationTime[0] * 2.0 / SingleCsOf6hbark.TimeUnit[0] / (SingleCsOf6hbark.pulsetiming[-1]/SingleCsOf6hbark.interpulse) 
    print 'interpulse after', SingleCsOf6hbark.interpulse
    #Parameter Update
    print 'Rabifrequency before changes', SingleCsOf6hbark.RabiFreqmax
    SingleCsOf6hbark.InCaseOf6hbark()
    print '2 x interogation time after modification',  SingleCsOf6hbark.TimeUnit[0] *  SingleCsOf6hbark.pulsetiming[-1] 
    print 'Rabifrequency', SingleCsOf6hbark.RabiFreqmax
    SingleCs = SingleCsOf6hbark
   
   
    #Define Atom Interferometer by using the above Atom and Quantum State definitions              
    IF1 = AtomInterferometer( Atom = SingleCs)
    #Define a Set for Atom Collection
    AtomSet = {}
    AtomSet0 =np.array([[]])

    #starting with a randomly generated single atom
    for AtomI in range(AtomNumber):
        #print 'Atom Index:', AtomI
        StateProbAmpLower, StateProbAmpUpper = IF1.SeparatedStateInterference()
        #Accumulate and assemble a Data Frame for every single atom interference information
        AtomSet[AtomI] = deepcopy(pd.Series({'x':IF1.UpperArmPartialSingleAtom.Position.x, 'y':IF1.UpperArmPartialSingleAtom.Position.y, 'z':IF1.UpperArmPartialSingleAtom.Position.z, 'StateLower': StateProbAmpLower,  'StateUpper': StateProbAmpUpper}))       
        AtomSet01=np.atleast_2d([[IF1.UpperArmPartialSingleAtom.Position.x,] + [IF1.UpperArmPartialSingleAtom.Position.y,] + [IF1.UpperArmPartialSingleAtom.Position.z,] + list(StateProbAmpLower) + list(StateProbAmpUpper) ])
        if AtomI ==0:
            AtomSet0 = AtomSet01
        else:       
            AtomSet0 = np.concatenate((AtomSet0, AtomSet01), axis=0)
    AtomDataSet = pd.DataFrame(AtomSet).T
    #print 'line 1065, show atom dataset and check the c labels'
    #print AtomDataSet
    ObFilter1 = FringeObservation(SingleAtom = SingleCs)
    FringeContrast, AtomFringes, AtomNumberVsFringeVsPosy = ObFilter1.ContrastExtractionUnit(AtomDataSet, StateSelectedForObservation='|1, 0hbarkeff>', NumberOfBins=2000000)
    
    print 'fringe contast\n', FringeContrast
    
    print 'AtomFringes\n', AtomFringes
       
    print 'time complete', tmstamp.time()
    
    close()
    plot(AtomFringes['y']*SingleCs.LengthUnit[0]*1000, AtomFringes['|1, 0hbarkeff>'],'o')
    plot(AtomNumberVsFringeVsPosy['y']*SingleCs.LengthUnit[0]*1000, AtomNumberVsFringeVsPosy['AtomNumber'],'ro')
    xlabel('Position (mm)')
    ylabel('AtomNumber & Fringe (arb.)')
    show()
    

    StateNames = list(SingleCs.Quantum.States.index) * 4 
    columnnames=['x', 'y', 'z'] + StateNames
    StatePAStackDF = pd.DataFrame(AtomSet0, columns=columnnames)
    StatePAStackDF.to_csv(FileName)
    return (StatePAStackDF, AtomFringes, FringeContrast, AtomNumberVsFringeVsPosy, SingleCs)

def MonteCarloStatisticsAtomInterferometryCs12hbark(AtomNumber=100, FileName='AIAnalysis0', AtomCloudTemp = (3.0, 'nK'),  PiPulseDuration=(60,'mus'), InterogationTime=(1,'s')):
    
    #Define Number of Quantum Momentum States
    QStates=QuantumAtomicInter_ExternalCoupledStates()
    QStates.InCaseOf12hbark()
    
    
    #Define type of Atoms and associated with Quantum States
    SingleCsOf12hbark = SingleAtom(QuatumStates=QStates, AtomDefinition=CommonElementList().Cs133)
    SingleCsOf12hbark.InCaseOf12hbark()
    
    #Parameter Modification:
    #Define Temperature etc Atom cloud parameters
    SingleCsOf12hbark.kbTk = SingleCsOf12hbark.kbTk3nK /3 * AtomCloudTemp[0]
    print SingleCsOf12hbark.kbTk
    
    print 'pipulse', SingleCsOf12hbark.pipulse
    SingleCsOf12hbark.pipulse = PiPulseDuration[0] *1.0e-6 / SingleCsOf12hbark.TimeUnit[0]
    print 'pipulse after', SingleCsOf12hbark.pipulse
    
    print '2 x interogation time before modification',  SingleCsOf12hbark.TimeUnit[0] *  SingleCsOf12hbark.pulsetiming[-1] 
    SingleCsOf12hbark.interpulse = InterogationTime[0] * 2.0 / SingleCsOf12hbark.TimeUnit[0] / (SingleCsOf12hbark.pulsetiming[-1]/SingleCsOf12hbark.interpulse) 
    print 'interpulse after', SingleCsOf12hbark.interpulse
    #Parameter Update
    print 'Rabifrequency before changes', SingleCsOf12hbark.RabiFreqmax
    SingleCsOf12hbark.InCaseOf12hbark()
    print '2 x interogation time after modification',  SingleCsOf12hbark.TimeUnit[0] *  SingleCsOf12hbark.pulsetiming[-1] 
    print 'Rabifrequency', SingleCsOf12hbark.RabiFreqmax
    SingleCs = SingleCsOf12hbark
   
    
    #Define Atom Interferometer by using the above Atom and Quantum State definitions              
    IF1 = AtomInterferometer( Atom = SingleCs)
    #Define a Set for Atom Collection
    AtomSet = {}
    AtomSet0 =np.array([[]])

    #starting with a randomly generated single atom
    for AtomI in range(AtomNumber):
        #print 'Atom Index:', AtomI
        StateProbAmpLower, StateProbAmpUpper = IF1.SeparatedStateInterference()
        #Accumulate and assemble a Data Frame for every single atom interference information
        AtomSet[AtomI] = deepcopy(pd.Series({'x':IF1.UpperArmPartialSingleAtom.Position.x, 'y':IF1.UpperArmPartialSingleAtom.Position.y, 'z':IF1.UpperArmPartialSingleAtom.Position.z, 'StateLower': StateProbAmpLower,  'StateUpper': StateProbAmpUpper}))       
        AtomSet01=np.atleast_2d([[IF1.UpperArmPartialSingleAtom.Position.x,] + [IF1.UpperArmPartialSingleAtom.Position.y,] + [IF1.UpperArmPartialSingleAtom.Position.z,] + list(StateProbAmpLower) + list(StateProbAmpUpper) ])
        if AtomI ==0:
            AtomSet0 = AtomSet01
        else:       
            AtomSet0 = np.concatenate((AtomSet0, AtomSet01), axis=0)
    AtomDataSet = pd.DataFrame(AtomSet).T
    #print 'line 1065, show atom dataset and check the c labels'
    #print AtomDataSet
    ObFilter1 = FringeObservation(SingleAtom = SingleCs)
    FringeContrast, AtomFringes, AtomNumberVsFringeVsPosy = ObFilter1.ContrastExtractionUnit(AtomDataSet, StateSelectedForObservation='|1, 0hbarkeff>', NumberOfBins=2000000)
    
    print 'fringe contast\n', FringeContrast
    
    print 'AtomFringes\n', AtomFringes
       
    print 'time complete', tmstamp.time()
    
    close()
    plot(AtomFringes['y']*SingleCs.LengthUnit[0]*1000, AtomFringes['|1, 0hbarkeff>'],'o')
    plot(AtomNumberVsFringeVsPosy['y']*SingleCs.LengthUnit[0]*1000, AtomNumberVsFringeVsPosy['AtomNumber'],'ro')
    xlabel('Position (mm)')
    ylabel('AtomNumber & Fringe (arb.)')
    show()
    

    StateNames = list(SingleCs.Quantum.States.index) * 4 
    columnnames=['x', 'y', 'z'] + StateNames
    StatePAStackDF = pd.DataFrame(AtomSet0, columns=columnnames)
    StatePAStackDF.to_csv(FileName)
    return (StatePAStackDF, AtomFringes, FringeContrast, AtomNumberVsFringeVsPosy, SingleCs)

def MonteCarloStatisticsAtomInterferometryCs8hbark(AtomNumber=100, FileName='AIAnalysis0', AtomCloudTemp = (3.0, 'nK'),  PiPulseDuration=(60,'mus'), InterogationTime=(1,'s')):
    
    #Define Number of Quantum Momentum States
    QStates=QuantumAtomicInter_ExternalCoupledStates()
    QStates.InCaseOf8hbark()
    
    
    #Define type of Atoms and associated with Quantum States
    SingleCsOf8hbark = SingleAtom(QuatumStates=QStates, AtomDefinition=CommonElementList().Cs133)
    SingleCsOf8hbark.InCaseOf8hbark()
    
    #Parameter Modification:
    #Define Temperature etc Atom cloud parameters
    SingleCsOf8hbark.kbTk = SingleCsOf8hbark.kbTk3nK /3 * AtomCloudTemp[0]
    print SingleCsOf8hbark.kbTk
    
    print 'pipulse', SingleCsOf8hbark.pipulse
    SingleCsOf8hbark.pipulse = PiPulseDuration[0] *1.0e-6 / SingleCsOf8hbark.TimeUnit[0]
    print 'pipulse after', SingleCsOf8hbark.pipulse
    
    print '2 x interogation time before modification',  SingleCsOf8hbark.TimeUnit[0] *  SingleCsOf8hbark.pulsetiming[-1] 
    SingleCsOf8hbark.interpulse = InterogationTime[0] * 2.0 / SingleCsOf8hbark.TimeUnit[0] / (SingleCsOf8hbark.pulsetiming[-1]/SingleCsOf8hbark.interpulse) 
    print 'interpulse after', SingleCsOf8hbark.interpulse
    #Parameter Update
    print 'Rabifrequency before changes', SingleCsOf8hbark.RabiFreqmax
    SingleCsOf8hbark.InCaseOf8hbark()
    print '2 x interogation time after modification',  SingleCsOf8hbark.TimeUnit[0] *  SingleCsOf8hbark.pulsetiming[-1] 
    print 'Rabifrequency', SingleCsOf8hbark.RabiFreqmax
    SingleCs = SingleCsOf8hbark
   
    
    #Define Atom Interferometer by using the above Atom and Quantum State definitions              
    IF1 = AtomInterferometer( Atom = SingleCs)
    #Define a Set for Atom Collection
    AtomSet = {}
    AtomSet0 =np.array([[]])

    #starting with a randomly generated single atom
    for AtomI in range(AtomNumber):
        #print 'Atom Index:', AtomI
        StateProbAmpLower, StateProbAmpUpper = IF1.SeparatedStateInterference()
        #Accumulate and assemble a Data Frame for every single atom interference information
        AtomSet[AtomI] = deepcopy(pd.Series({'x':IF1.UpperArmPartialSingleAtom.Position.x, 'y':IF1.UpperArmPartialSingleAtom.Position.y, 'z':IF1.UpperArmPartialSingleAtom.Position.z, 'StateLower': StateProbAmpLower,  'StateUpper': StateProbAmpUpper}))       
        AtomSet01=np.atleast_2d([[IF1.UpperArmPartialSingleAtom.Position.x,] + [IF1.UpperArmPartialSingleAtom.Position.y,] + [IF1.UpperArmPartialSingleAtom.Position.z,] + list(StateProbAmpLower) + list(StateProbAmpUpper) ])
        if AtomI ==0:
            AtomSet0 = AtomSet01
        else:       
            AtomSet0 = np.concatenate((AtomSet0, AtomSet01), axis=0)
    AtomDataSet = pd.DataFrame(AtomSet).T
    #print 'line 1065, show atom dataset and check the c labels'
    #print AtomDataSet
    ObFilter1 = FringeObservation(SingleAtom = SingleCs)
    FringeContrast, AtomFringes, AtomNumberVsFringeVsPosy = ObFilter1.ContrastExtractionUnit(AtomDataSet, StateSelectedForObservation='|1, 0hbarkeff>', NumberOfBins=2000000)
    
    print 'fringe contast\n', FringeContrast
    
    print 'AtomFringes\n', AtomFringes
       
    print 'time complete', tmstamp.time()
    
    close()
    plot(AtomFringes['y']*SingleCs.LengthUnit[0]*1000, AtomFringes['|1, 0hbarkeff>'],'o')
    plot(AtomNumberVsFringeVsPosy['y']*SingleCs.LengthUnit[0]*1000, AtomNumberVsFringeVsPosy['AtomNumber'],'ro')
    xlabel('Position (mm)')
    ylabel('AtomNumber & Fringe (arb.)')
    show()
    

    StateNames = list(SingleCs.Quantum.States.index) * 4 
    columnnames=['x', 'y', 'z'] + StateNames
    StatePAStackDF = pd.DataFrame(AtomSet0, columns=columnnames)
    StatePAStackDF.to_csv(FileName)
    return (StatePAStackDF, AtomFringes, FringeContrast, AtomNumberVsFringeVsPosy, SingleCs)

def MonteCarloStatisticsAtomInterferometryCs2hbark(AtomNumber=100, FileName='AIAnalysis0', AtomCloudTemp = (3.0, 'nK'),  PiPulseDuration=(60,'mus'), InterogationTime=(1,'s')):
    
    #Define Number of Quantum Momentum States
    QStates=QuantumAtomicInter_ExternalCoupledStates()
    QStates.InCaseOf2hbark()
    
    
    #Define type of Atoms and associated with Quantum States
    SingleCsOf2hbark = SingleAtom(QuatumStates=QStates, AtomDefinition=CommonElementList().Cs133)
    SingleCsOf2hbark.InCaseOf2hbark()
    
    #Parameter Modification:
    #Define Temperature etc Atom cloud parameters
    SingleCsOf2hbark.kbTk = SingleCsOf2hbark.kbTk3nK /3 * AtomCloudTemp[0]
    print SingleCsOf2hbark.kbTk
    
    print 'pipulse', SingleCsOf2hbark.pipulse
    SingleCsOf2hbark.pipulse = PiPulseDuration[0] *1.0e-6 / SingleCsOf2hbark.TimeUnit[0]
    print 'pipulse after', SingleCsOf2hbark.pipulse
    
    print '2 x interogation time before modification',  SingleCsOf2hbark.TimeUnit[0] *  SingleCsOf2hbark.pulsetiming[-1] 
    SingleCsOf2hbark.interpulse = InterogationTime[0] * 2.0 / SingleCsOf2hbark.TimeUnit[0] / (SingleCsOf2hbark.pulsetiming[-1]/SingleCsOf2hbark.interpulse) 
    print 'interpulse after', SingleCsOf2hbark.interpulse
    #Parameter Update
    print 'Rabifrequency before changes', SingleCsOf2hbark.RabiFreqmax
    SingleCsOf2hbark.InCaseOf2hbark()
    print '2 x interogation time after modification',  SingleCsOf2hbark.TimeUnit[0] *  SingleCsOf2hbark.pulsetiming[-1] 
    print 'Rabifrequency', SingleCsOf2hbark.RabiFreqmax
    SingleCs = SingleCsOf2hbark
   
    
    #Define Atom Interferometer by using the above Atom and Quantum State definitions              
    IF1 = AtomInterferometer( Atom = SingleCs)
    #Define a Set for Atom Collection
    AtomSet = {}
    AtomSet0 =np.array([[]])

    #starting with a randomly generated single atom
    for AtomI in range(AtomNumber):
        #print 'Atom Index:', AtomI
        StateProbAmpLower, StateProbAmpUpper = IF1.SeparatedStateInterference()
        #Accumulate and assemble a Data Frame for every single atom interference information
        AtomSet[AtomI] = deepcopy(pd.Series({'x':IF1.UpperArmPartialSingleAtom.Position.x, 'y':IF1.UpperArmPartialSingleAtom.Position.y, 'z':IF1.UpperArmPartialSingleAtom.Position.z, 'StateLower': StateProbAmpLower,  'StateUpper': StateProbAmpUpper}))       
        AtomSet01=np.atleast_2d([[IF1.UpperArmPartialSingleAtom.Position.x,] + [IF1.UpperArmPartialSingleAtom.Position.y,] + [IF1.UpperArmPartialSingleAtom.Position.z,] + list(StateProbAmpLower) + list(StateProbAmpUpper) ])
        if AtomI ==0:
            AtomSet0 = AtomSet01
        else:       
            AtomSet0 = np.concatenate((AtomSet0, AtomSet01), axis=0)
    AtomDataSet = pd.DataFrame(AtomSet).T
    #print 'line 1065, show atom dataset and check the c labels'
    #print AtomDataSet
    ObFilter1 = FringeObservation(SingleAtom = SingleCs)
    FringeContrast, AtomFringes, AtomNumberVsFringeVsPosy = ObFilter1.ContrastExtractionUnit(AtomDataSet, StateSelectedForObservation='|1, 0hbarkeff>', NumberOfBins=2000000)
    
    print 'fringe contast\n', FringeContrast
    
    print 'AtomFringes\n', AtomFringes
       
    print 'time complete', tmstamp.time()
    
    close()
    plot(AtomFringes['y']*SingleCs.LengthUnit[0]*1000, AtomFringes['|1, 0hbarkeff>'],'o')
    plot(AtomNumberVsFringeVsPosy['y']*SingleCs.LengthUnit[0]*1000, AtomNumberVsFringeVsPosy['AtomNumber'],'ro')
    xlabel('Position (mm)')
    ylabel('AtomNumber & Fringe (arb.)')
    show()
    

    StateNames = list(SingleCs.Quantum.States.index) * 4 
    columnnames=['x', 'y', 'z'] + StateNames
    StatePAStackDF = pd.DataFrame(AtomSet0, columns=columnnames)
    StatePAStackDF.to_csv(FileName)
    return (StatePAStackDF, AtomFringes, FringeContrast, AtomNumberVsFringeVsPosy, SingleCs)

# THSE:Monte
def MonteCarloStatisticsAtomInterferometryRb2hbark(AtomNumber=100, Laser2PhoDetuningFactor=1.0, FileName='AIAnalysis0', TimeDelay=(2.93,'ms'), R0=(np.array([0, -13.478, 0]),'mm'), V0=(np.array([0,4.6,0]),'m/s'), AtomCloudTemp = (3.0, 'nK'),  PiPulseDuration=(60,'mus'), InterogationTime=(1,'s')):
    #def MonteCarloStatisticsAtomInterferometryRb2hbark(AtomNumber=100, FileName='AIAnalysis0', TimeDelay=(2.93,'ms'), R0=(np.array([0, -0.0013478, 0])*2,'mm'), V0=(np.array([0,0.46,0]),'m/s'), AtomCloudTemp = (3.0, 'nK'),  PiPulseDuration=(60,'mus'), InterogationTime=(1,'s')):

    #Define Number of Quantum Momentum States
    # THSE: new create 
    QStates=QuantumAtomicInter_ExternalCoupledStates()
    QStates.InCaseOf2hbark()
    
   
    #Define type of Atoms and associated with Quantum States
    # THSE: add intial velocity and position
    SingleRbOf2hbark = SingleAtom(QuatumStates=QStates, AtomDefinition=CommonElementList().Rb87)
    R00 = SpatialCoordinates(vec_r=R0[0]*1.0e-3/SingleRbOf2hbark.LengthUnit[0])
    V00 = Velocity(vec_v=V0[0]/(SingleRbOf2hbark.LengthUnit[0]/SingleRbOf2hbark.TimeUnit[0]))
    SingleRbOf2hbark = SingleAtom(QuatumStates=QStates, R0=R00 , V0=V00, AtomDefinition=CommonElementList().Rb87)
    SingleRbOf2hbark.InCaseOf2hbarkWithInitialDelay()
    

    #Parameter Modification:
    #Define Temperature etc Atom cloud parameters
    SingleRbOf2hbark.kbTk = SingleRbOf2hbark.kbTk3nK /3 * AtomCloudTemp[0]
    print SingleRbOf2hbark.kbTk
    
    print 'pipulse', SingleRbOf2hbark.pipulse
    SingleRbOf2hbark.pipulse = PiPulseDuration[0] *1.0e-6 / SingleRbOf2hbark.TimeUnit[0]
    print 'pipulse after', SingleRbOf2hbark.pipulse
    
    print '2 x interogation time before modification',  SingleRbOf2hbark.TimeUnit[0] *  SingleRbOf2hbark.pulsetiming[-1] 
    SingleRbOf2hbark.interpulse = InterogationTime[0] * 2.0 / SingleRbOf2hbark.TimeUnit[0] / \
                                  (SingleRbOf2hbark.pulsetiming[-1]/SingleRbOf2hbark.interpulse) 
    print 'interpulse after', SingleRbOf2hbark.interpulse
    #Parameter Update
    print 'Rabifrequency before changes', SingleRbOf2hbark.RabiFreqmax 
    # THSE: add a note: actually implement the changes to the timing sequence 
    SingleRbOf2hbark.InCaseOf2hbarkWithInitialDelay()
    print '2 x interogation time after modification',  SingleRbOf2hbark.TimeUnit[0] *  SingleRbOf2hbark.pulsetiming[-1] 
    print 'Rabifrequency', SingleRbOf2hbark.RabiFreqmax
    # THSE: add initial time delay
    SingleRbOf2hbark.pulsetiming = SingleRbOf2hbark.pulsetiming + TimeDelay[0]*1.0e-3/SingleRbOf2hbark.TimeUnit[0]
    print 'after modification, pulse timing sequence is:\n', SingleRbOf2hbark.pulsetiming * SingleRbOf2hbark.TimeUnit[0], SingleRbOf2hbark.TimeUnit[1]
    SingleRb = SingleRbOf2hbark
   

    #Define Atom Interferometer by using the above Atom and Quantum State definitions              
    IF1 = AtomInterferometer( Atom = SingleRb, Laser2PhoDetuningFactor=Laser2PhoDetuningFactor)
    #Define a Set for Atom Collection
    AtomSet = {}
    AtomSet0 =np.array([[]])
    
    #starting with a randomly generated single atom
    for AtomI in range(AtomNumber):
        #print 'Atom Index:', AtomI
        StateProbAmpLower, StateProbAmpUpper = IF1.SeparatedStateInterference()
        

        #Accumulate and assemble a Data Frame for every single atom interference information
        AtomSet[AtomI] = deepcopy(pd.Series({'x':IF1.UpperArmPartialSingleAtom.Position.x, 'y':IF1.UpperArmPartialSingleAtom.Position.y, 'z':IF1.UpperArmPartialSingleAtom.Position.z, 'StateLower': StateProbAmpLower,  'StateUpper': StateProbAmpUpper}))       
        AtomSet01=np.atleast_2d([[IF1.UpperArmPartialSingleAtom.Position.x,] + [IF1.UpperArmPartialSingleAtom.Position.y,] + [IF1.UpperArmPartialSingleAtom.Position.z,] + list(StateProbAmpLower) + list(StateProbAmpUpper) ])
        if AtomI ==0:
            AtomSet0 = AtomSet01
        else:       
            AtomSet0 = np.concatenate((AtomSet0, AtomSet01), axis=0)
            
    AtomDataSet = pd.DataFrame(AtomSet).T
    #print 'line 1065, show atom dataset and check the c labels'
    #print AtomDataSet
    
    PhaseShiftList = np.linspace(0.0, 2.0*np.pi, 100)
    FringeList = pd.DataFrame()
    for Indexx in range(len(PhaseShiftList)):
        ObFilter1 = FringeObservationH(PhaseShift = PhaseShiftList[Indexx], SingleAtom = SingleRb, AtomInterferometer=IF1)
        # THSE: change to withoutfiltering
        FringeContrast, AtomFringes, AtomNumberVsFringeVsPosy = ObFilter1.ContrastExtractionUnitWithoutFiltering(AtomDataSet, StateSelectedForObservation='|1, 0hbarkeff>', NumberOfBins=2)
        FringeList = pd.concat([FringeList, AtomFringes], axis=0)
        
    # correct the value of Fringe contrast
    FringeContrast = max(FringeList['|1, 0hbarkeff>']) - min(FringeList['|1, 0hbarkeff>'])
    # correct the value of Atom Fringes
    AtomFringes = FringeList['|1, 0hbarkeff>']
    
    print 'Fringe List : \n', FringeList
    
    print 'fringe contast\n', FringeContrast
    
    print 'AtomFringes\n', AtomFringes
       
    print 'time complete', tmstamp.time()
    """
    close()
    
    plot(AtomFringes,'o')
    
    xlabel('Phase')
    #plot(AtomFringes['y']*SingleRb.LengthUnit[0]*1000, AtomFringes['|1, 0hbarkeff>'],'o')
    #plot(AtomNumberVsFringeVsPosy['y']*SingleRb.LengthUnit[0]*1000, AtomNumberVsFringeVsPosy['AtomNumber'],'ro')
    #xlabel('Position (mm)')
    ylabel('AtomNumber & Fringe (arb.)')
    show()
    """

    StateNames = list(SingleRb.Quantum.States.index) * 4 
    columnnames=['x', 'y', 'z'] + StateNames
    StatePAStackDF = pd.DataFrame(AtomSet0, columns=columnnames)
    StatePAStackDF.to_csv(FileName)
    return (StatePAStackDF, AtomFringes, FringeContrast, AtomNumberVsFringeVsPosy, SingleRb)




def pipulseDurationAnalysis():
    pulseduration = np.arange(8.0, 68.0, 5)
    FContrast_pipulse= np.zeros(len(pulseduration))
    for index_ in arange(len(pulseduration)):
        UpdateGlobalVariables(pipulsenewvalue = pulseduration[index_])
        DataFileName ='InterferometerOutput.csv'
        AtomNumber = 2

        StatePS, StatePAStack, FringeContrast = MonteCarloStatisticsAtomInterferometry(AtomNumber, DataFileName)
        print 'Fringe Contrast is : ', FringeContrast
        FContrast_pipulse[index_]= FringeContrast

    print 'fring contrast', FContrast_pipulse, 'shape', FContrast_pipulse.shape
    print 'pulseduration', pulseduration, 'shape', pulseduration.shape
    
    FContrast_pipulseDF = pds.DataFrame(FContrast_pipulse, index = pulseduration).transpose()
    FContrast_pipulseDF.to_csv('FContrast_pipulseDF8_68us.csv')

    close()
    plot(pulseduration, FContrast_pipulse)
    xlabel('$\pi$-pulse duration ($\mu$s)')
    ylabel('Fringe Contrast (arb.)')
    savefig('FContrast_pipulseDF.ps')
    show()



class AtomInterferometerCollection(object):
    def __init__(self):
        pass
    
    def Rb12hbarkInterferometer(self):
        #Define Number of Quantum Momentum States
        QStates=QuantumAtomicInter_ExternalCoupledStates()
        QStates.InCaseOf12hbark()
        #Define type of Atoms and associated with Quantum States
        SingleRbOf12hbark = SingleAtom(QuatumStates = QStates)
        SingleRbOf12hbark.InCaseOf12hbark()
        SingleRb = SingleRbOf12hbark
        #Define Temperature etc Atom cloud parameters
        SingleRb.kbTk = SingleRb.kbTk3nK /3 * AtomCloudTemp
        print SingleRb.kbTk

        #Define Atom Interferometer by using the above Atom and Quantum State definitions              
        IF1 = AtomInterferometer( Atom = SingleRb)
        #Define a Set for Atom Collection
        AtomSet = {}
        AtomSet0 =np.array([[]])
        
    def Rb8hbarkInterferometer(self):
        #Define Number of Quantum Momentum States
        QStates=QuantumAtomicInter_ExternalCoupledStates()
        QStates.InCaseOf8hbark()
        #Define type of Atoms and associated with Quantum States
        SingleRbOf8hbark = SingleAtom(QuatumStates=QStates)
        SingleRbOf8hbark.InCaseOf8hbark()
        SingleRb = SingleRbOf8hbark
        #Define Temperature etc Atom cloud parameters
        SingleRb.kbTk = SingleRb.kbTk3nK /3 * AtomCloudTemp
        print SingleRb.kbTk

        #Define Atom Interferometer by using the above Atom and Quantum State definitions              
        IF1 = AtomInterferometer( Atom = SingleRb )
        #Define a Set for Atom Collection
        AtomSet = {}
        AtomSet0 =np.array([[]])
        
    def Rb6hbarkInterferometer(self):
        #Define Number of Quantum Momentum States
        QStates=QuantumAtomicInter_ExternalCoupledStates()
        QStates.InCaseOf6hbark()
        #Define type of Atoms and associated with Quantum States
        SingleRbOf6hbark = SingleAtom(QuatumStates=QStates)
        SingleRbOf6hbark.InCaseOf6hbark()
        SingleRb = SingleRbOf6hbark
        #Define Temperature etc Atom cloud parameters
        SingleRb.kbTk = SingleRb.kbTk3nK /3 * AtomCloudTemp
        print SingleRb.kbTk

        #Define Atom Interferometer by using the above Atom and Quantum State definitions              
        IF1 = AtomInterferometer( Atom = SingleRb)
        #Define a Set for Atom Collection
        AtomSet = {}
        AtomSet0 =np.array([[]])

class AIScans(object):
    
    def __init__(self):
        pass
    
    def scanPulseDuration(self):
        # THSE: new create
        pi_PulseDuration = np.linspace(1, 15, 300) # pulse duration can't be zero.
        FringeContrast = np.zeros((300,1))
        for i in range(0, 300):
            print "in the case of pi pulse duration is ", pi_PulseDuration[i]
            StatePAStackDF, AtomFringes, FringeContrast[i], AtomNumberVsFringeVsPosy, SingleRb = MonteCarloStatisticsAtomInterferometryRb2hbark(AtomNumber=100, Laser2PhoDetuningFactor=1.0, PiPulseDuration=(pi_PulseDuration[i],'mus'), AtomCloudTemp = (20000.0, 'nK'), InterogationTime=(0.005,'s'))
            print 'works!\n'
            
        print FringeContrast        
        plot(FringeContrast,'o')

        xlabel('Phase')
        #plot(AtomFringes['y']*SingleRb.LengthUnit[0]*1000, AtomFringes['|1, 0hbarkeff>'],'o')
        #plot(AtomNumberVsFringeVsPosy['y']*SingleRb.LengthUnit[0]*1000, AtomNumberVsFringeVsPosy['AtomNumber'],'ro')
        #xlabel('Position (mm)')
        ylabel('AtomNumber & Fringe (arb.)')
        #show()
        self.SaveFiles(x = pi_PulseDuration, y = FringeContrast, FileIndex = 0)
    
    def scan2PhotonDetuning(self):
        # THSE: new create
        twoPhotonDetuning = np.linspace(-5, 5, 300) # 2 photon detuning.
        FringeContrast = np.zeros((300,1))
        for i in range(0, 300):
            print "in the case of pi pulse duration is ", twoPhotonDetuning[i]
            StatePAStackDF, AtomFringes, FringeContrast[i], AtomNumberVsFringeVsPosy, SingleRb = MonteCarloStatisticsAtomInterferometryRb2hbark(Laser2PhoDetuningFactor=twoPhotonDetuning[i], AtomNumber=100, PiPulseDuration=(6.0,'mus'), AtomCloudTemp = (20000.0, 'nK'), InterogationTime=(0.005,'s'))
            print 'works!\n'
        print FringeContrast        
        plot(FringeContrast,'o')

        xlabel('Phase')
        #plot(AtomFringes['y']*SingleRb.LengthUnit[0]*1000, AtomFringes['|1, 0hbarkeff>'],'o')
        #plot(AtomNumberVsFringeVsPosy['y']*SingleRb.LengthUnit[0]*1000, AtomNumberVsFringeVsPosy['AtomNumber'],'ro')
        #xlabel('Position (mm)')
        ylabel('AtomNumber & Fringe (arb.)')
        #show()    
        self.SaveFiles(x = twoPhotonDetuning, y = FringeContrast, FileIndex = 1)
    
    def SaveFiles(self, x, y, FileIndex):
        # THSE: new create
        XpdForSave = pd.DataFrame(x)
        ypdForSave = pd.DataFrame(y)
        XpdForSave.to_csv('xdata'+str(FileIndex)+'.dat')
        ypdForSave.to_csv('ydata'+str(FileIndex)+'.dat')


def Testing8hbarkstateextension():
    
    QStates=QuantumAtomicInter_ExternalCoupledStates()
    QStates.InCaseOf8hbark()
    SingleRbOf8hbark = SingleAtom(QuatumStates=QStates)   
    SingleRbOf8hbark.InCaseOf8hbark()
    ai = AtomInterferometer(Atom = SingleRbOf8hbark)
    ai.UpperArmTransition()
    ai.ShowUpperArmTransitionSequence()
    ai.LowerArmTransition()
    ai.ShowLowerArmTransitionSequence()
    
def Testing12hbarkstateextension():
    QStates=QuantumAtomicInter_ExternalCoupledStates()
    QStates.InCaseOf12hbark()
    SingleRbOf12hbark = SingleAtom(QuatumStates=QStates)   
    SingleRbOf12hbark.InCaseOf12hbark()
    ai = AtomInterferometer(Atom = SingleRbOf12hbark)
    ai.UpperArmTransition()
    ai.ShowUpperArmTransitionSequence()
    ai.LowerArmTransition()
    ai.ShowLowerArmTransitionSequence()

def TestingFringeObservation():
    QStates=QuantumAtomicInter_ExternalCoupledStates()
    #QStates.InCaseOf12hbark()
    SingleRb = SingleAtom(QuatumStates=QStates)   
    #SingleRbOf12hbark.InCaseOf12hbark() 
        
    CV=SingleRb.Quantum.ComplexVecConvertedFromStateVec(SingleRb.Quantum.StateVec)
        
    SV=SingleRb.Quantum.StateVecConvertedFromComplexVec(CV)
    print SV
    FO=FringeObservation()
    SVsf=FO.PhaseShift(SV, 10)
    print SVsf

    
    
    
def main():

    #obff=AtomInterferometerAA()
    print 'time start', tmstamp.time()
    
    """
    QStates=QuantumAtomicInter_ExternalCoupledStates()
    QStates.InCaseOf2hbark()
    
    
    #Define type of Atoms and associated with Quantum States
    SingleCsOf2hbark = SingleAtom(QuatumStates=QStates, AtomDefinition=CommonElementList().Cs133)
    SingleCsOf2hbark.InCaseOf2hbark()
    SingleCs = SingleCsOf2hbark
    
    IFtest= AtomInterferometer( Atom = SingleCs)
    IFtest.UpperArmTransition()
    IFtest.ShowUpperArmTransitionSequence()
    IFtest.LowerArmTransition()
    IFtest.ShowLowerArmTransitionSequence()
    """
    # THSE: create a calculation instance for Rb:
    AI = AIScans()
    AI.scanPulseDuration()
    AI.scan2PhotonDetuning()
    print 'time complete', tmstamp.time()
    #return (StatePAStackDF, AtomFringes, FringeContrast, AtomNumberVsFringeVsPosy, SingleRb)
    
 
StatePAStackDF1 = 0
if __name__ == '__main__':
    #StatePAStackDF1, AtomFringes1, FringeContrast1, AtomNumberVsFringeVsPosy1, SingleCs1 = main()
    main()
    #StatePAStackDF1, AtomFringes1, FringeContrast1, AtomNumberVsFringeVsPosy1, SingleRb1 = main()
    #, AtomFringes1, FringeContrast1, AtomNumber1, PositionyMean1
    
   
