import torch
from models.DFFSM import DFFSM
import matplotlib.pyplot as plt


# Set PV system configurations
PVS_INFO = {'Module':{ # PV module specification is referred from SAM (by NREL)
                     'Name': 'TSM-240PA05',
                     'Technology': 'multiSi',
                     'Nmc': 60, # number of cells in the PV module
                     'Nmd': 3, # number of bypass diodes in a PV module
                     'NShortSide': 6, # number of PV cells in the short side of the PV module
                     'Pmp_stc': 240, # MPPT power under STC
                     'Vmp_stc': 29.7, # MPPT power voltage under STC
                     'Imp_stc': 8.1,  # MPPT power current under STC
                     'Voc_stc': 37.3,  # Open-circuit voltage under STC
                     'Isc_stc': 8.62,  # short-circuit current under STC
                     'tc_Isc': 0.047/100,  # temperature coefficient of shor-circuit current (a)
                     'tc_Voc': -0.32/100,  # temperature coefficient of open-circuit voltage (b)
                     'ic_Voc': 0.06,       # irradiance coefficient of open-circuit voltage (c)
                     'tc_Pmax': -0.43,
                     'Iph_ref': 8.627,     # Iph under STC condition
                     'I0_ref': 1.363011e-10, # Io under STC condition
                     'Rs_ref': 0.395,  # Rs under STC condition
                     'Rsh_ref': 455.918, # Rsh under STC condition
                     'n_ref': 0.973,   # diode ideality factor under STC (used by pvsyst)
                     'Adjust': 5.877,  # % defined by CEC for adjusting tc_Isc
                     'Re_a': 0.002,    # 雪崩击穿所涉及的欧姆电流分数
                     'Re_Vb': -15,     # -21.29;%%结击穿电压
                     'Re_m': 3 },         # 击穿指数
            'Nm': 6,     # number of PV modules in a string
            'InstallType': 'vertical'} # installation type (vertical installation, horizontal installation)

Nmc = PVS_INFO['Module']['Nmc'] # number of PV cells in one PV module
Nmd = PVS_INFO['Module']['Nmd'] # number of bypass diodes in the PV module
Nm =  PVS_INFO['Nm'] # number of PV modules in a string
Nsubc = int(Nmc/Nmd) # number of series PV cells in a sub-string
Nsub = int(Nm*Nmd)   # number of sub-strings in a PV string

# Define cell temperature (°C) and in-plane irradiance (W/m2)
T_m, S_m = 56.0, 987.0

## Initialize DFFSM
dffsm = DFFSM(pvsinfo=PVS_INFO,
              solver= {'5paras-method': 'cec', # five parameters calculation method, including 'desoto', 'cec'
                        'it_num': 6,  # the number of newton iterations in forward calculation
                      })
dffsm.adjust_bounds(S_m) # adjust the searching boundary based on the irradiance

## Define a current sequence
I = torch.arange(0, 8.7, 0.02)

## I-V curve under no fault condition using DFFSM
fault_vector_0 = [0, 0, 0, 0, 0, 0, 0, 0, 0,]
result_0 = dffsm.forward(I, torch.tensor([T_m]), torch.tensor([S_m]), torch.tensor(fault_vector_0))


## I-V curve under single shading
n_s1 = 8   # numer of affected sub-strings under shadow-1
n_c1 = 10   # number of shaded cells in sub-strings under shadow-1
r_1 = 0.5   # shading ratio of shadow-1 (percent of lost irradiance)
n_s2 = 0    # numer of affected sub-strings under shadow-2
n_c2 = 0   # number of shaded cells in sub-strings under shadow-2
r_2 = 0   # shading ratio of shadow-2 (percent of lost irradiance)
n_sc = 0    # numer of short-circuited substrings
d_oc1 = 0   # existence of open-circuit bypass diode under shadow-1
d_oc2 = 0   # existence of open-circuit bypass diode under shadow-2
fault_vector_1 = [n_s1, n_c1, r_1, n_s2, n_c2, r_2, n_sc, d_oc1, d_oc2]
result_1 = dffsm.forward(I, torch.tensor([T_m]), torch.tensor([S_m]), torch.tensor(fault_vector_1))

## I-V curve under short-circuit fault
n_s1 = 0   # numer of affected sub-strings under shadow-1
n_c1 = 0   # number of shaded cells in sub-strings under shadow-1
r_1 = 0   # shading ratio of shadow-1 (percent of lost irradiance)
n_s2 = 0    # numer of affected sub-strings under shadow-2
n_c2 = 0   # number of shaded cells in sub-strings under shadow-2
r_2 = 0   # shading ratio of shadow-2 (percent of lost irradiance)
n_sc = 2    # numer of short-circuited substrings
d_oc1 = 0   # existence of open-circuit bypass diode under shadow-1
d_oc2 = 0   # existence of open-circuit bypass diode under shadow-2
fault_vector_2 = [n_s1, n_c1, r_1, n_s2, n_c2, r_2, n_sc, d_oc1, d_oc2]
result_2 = dffsm.forward(I, torch.tensor([T_m]), torch.tensor([S_m]), torch.tensor(fault_vector_2))


## I-V curve under concurrent faults (double shading + short-circuit fault)
n_s1 = 7    # numer of affected sub-strings under shadow-1
n_c1 = 10   # number of shaded cells in sub-strings under shadow-1
r_1 = 0.8   # shading ratio of shadow-1 (percent of lost irradiance)
n_s2 = 3    # numer of affected sub-strings under shadow-2
n_c2 = 10   # number of shaded cells in sub-strings under shadow-2
r_2 = 0.2   # shading ratio of shadow-2 (percent of lost irradiance)
n_sc = 2    # numer of short-circuited substrings
d_oc1 = 0   # existence of open-circuit bypass diode under shadow-1
d_oc2 = 0   # existence of open-circuit bypass diode under shadow-2
fault_vector_3 = [n_s1, n_c1, r_1, n_s2, n_c2, r_2, n_sc, d_oc1, d_oc2]
result_3 = dffsm.forward(I, torch.tensor([T_m]), torch.tensor([S_m]), torch.tensor(fault_vector_3))


case_labels = ['no fault', 'single shading', 'short-circuit', 'concurrent faults']


plt.figure(1)
plt.xlabel('Voltage [V]')
plt.ylabel('Current [A]')
plt.plot(result_0['V'], result_0['I'], '*-',linewidth = 1, color='k')
plt.plot(result_1['V'], result_1['I'], '*-',linewidth = 1, color='b')
plt.plot(result_2['V'], result_2['I'], '*-',linewidth = 1, color='g')
plt.plot(result_3['V'], result_3['I'], '*-',linewidth = 1, color='m')
plt.title(PVS_INFO['Module']['Name'] + ' $\\times$ ' + str(Nm))
plt.legend(case_labels)
plt.grid()
plt.show()

