import torch
import numpy as np
from models.DFFSM import DFFSM
from models.CFFSM import CFFSM
import torch_optimizer as optim
import matplotlib.pyplot as plt


# 1. Set PV system configurations
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


# 2. Configure cell temperature, in-plane irradiation, and fault vector
# 2.1 cell temperature (°C) and in-plane irradiance (W/m2)
T_m, S_m = 56.0, 987.0
# 2.2 fault parameters
n_s1 = 7    # numer of affected sub-strings under shadow-1
n_c1 = 10   # number of shaded cells in sub-strings under shadow-1
r_1 = 0.8   # shading ratio of shadow-1 (percent of lost irradiance)
n_s2 = 3    # numer of affected sub-strings under shadow-2
n_c2 = 10   # number of shaded cells in sub-strings under shadow-2
r_2 = 0.2   # shading ratio of shadow-2 (percent of lost irradiance)
n_sc = 2    # numer of short-circuited substrings
d_oc1 = 0   # existence of open-circuit bypass diode under shadow-1
d_oc2 = 0   # existence of open-circuit bypass diode under shadow-2
# construct the fault vector
fault_vector = [n_s1, n_c1, r_1, n_s2, n_c2, r_2, n_sc, d_oc1, d_oc2]

# 3. Simulate an I-V curve using CFFSM (code-based fast fault simulation model)
## 3.1 Initialize the CFFSM model
cffsm = CFFSM(PVS_INFO)
## 3.2 Configure the input of CFFSM
T = np.ones((Nsub, Nsubc))*T_m  # initialize cell-wide temperature
S = np.ones((Nsub, Nsubc))*S_m  # initialize cell-wide irradiance (with shadow)
cffsm.addShadow(locX=0, locY=0, len=int(n_s1*2), width=int(n_c1/2), shade_ratio=r_1)
cffsm.addShadow(locX=n_s1*2, locY=0, len=int(n_s2*2), width=int(n_c2/2), shade_ratio=r_2)
S = cffsm.get_irradiance_after_shadow(S) # get cell-wide irradiance after adding shadows
### 3.3 Configure fault condition of bypass diodes
D = np.zeros(Nsub, dtype=int) # 0: no fault, 1: short-circuit, 2: open-circuit
for i in range(n_sc):
    D[Nsub-(i+1)] = 1
if d_oc1 == 1:
    D[0] = 2
if d_oc2 == 1:
    D[n_s1] = 2

## 3.4 Simulate an I-V curve as a measured I-V curve
IV_cffsm = cffsm.IV_SCAN(T=T, S=S, D=D, Rc=0,
                         solver={'dI': 0.05, # current increment in IV scanning
                                 '5paras-method': 'cec', # five parameters calculation method, including {'desoto', 'cec'}
                                 'voltage-method': 'Newton-RB', # cell voltage calculation method, including
                                                                # {'LambertW', 'Newton', 'Newton-RB'}
                                })
I_m, V_m = torch.tensor(IV_cffsm['I']), torch.tensor(IV_cffsm['V']) # conver I-V curves to tensors





# 4. Gradient-based fault parameters identification (GFPI) based on DFFSM
## 4.1 Initialize DFFSM
dffsm = DFFSM(pvsinfo=PVS_INFO,
              solver= {'5paras-method': 'cec', # five parameters calculation method, including 'desoto', 'cec'
                        'it_num': 6,  # the number of newton iterations in forward calculation
                      })
dffsm.adjust_bounds(S_m) # adjust the searching boundary based on the irradiance

## 4.2 GFPI using Adahessian optimizor
case_name_1 = "Adahessian"
print(f'Run {case_name_1} Optimizer!')
num_it = 1000
MSE = torch.nn.MSELoss()
lr = 1
x_0 = [0.0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0] # initial fault vector
x = torch.tensor(x_0, dtype=torch.float64, requires_grad=True)
optimizer = optim.Adahessian([x], lr=lr, hessian_power=1.0)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
steps = []
grads = []
for i in range(num_it):
    optimizer.zero_grad()
    V = dffsm.forward(I_m, torch.tensor([T_m]), torch.tensor([S_m]), x)['V']
    loss = MSE(V, V_m)
    loss.backward(create_graph = True)
    optimizer.step()
    scheduler.step()
    with torch.no_grad():
        x.data = dffsm.projection(x) # projected gradient decent
        print('Loss =  ', loss.detach().numpy().copy())
        steps.append(loss.detach().numpy().copy())
        grads.append(x.grad.norm().detach().numpy().copy())

x.requires_grad=False
x_adahessian = x.detach().numpy().copy()
V_adahessian = dffsm.forward(I_m, torch.tensor([T_m]), torch.tensor([S_m]), x_adahessian)['V']


## 4.3 GFPI using Adam
case_name_2 = "Adam"
print(f'Run {case_name_2} Optimizer!')
lr = 0.1
x_0 = [0.0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0]
x = torch.tensor(x_0, dtype=torch.float64, requires_grad=True)
optimizer = torch.optim.Adam([x], lr=lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
steps_2 = []
grads_2 = []
for i in range(num_it):
    optimizer.zero_grad()
    V = dffsm.forward(I_m, torch.tensor([T_m]), torch.tensor([S_m]), x)['V']
    loss = MSE(V, V_m)
    loss.backward()
    optimizer.step()
    scheduler.step()
    with torch.no_grad():
        x.data = dffsm.projection(x) # projected gradient decent
        print('Loss =  ', loss.detach().numpy().copy())
        steps_2.append(loss.detach().numpy().copy())
        grads_2.append(x.grad.norm().detach().numpy().copy())

x.requires_grad=False
x_adam = x.detach().numpy().copy()
V_adam = dffsm.forward(I_m, torch.tensor([T_m]), torch.tensor([S_m]), x_adam)['V']


# 5. Print identification results
print("\nGFPI Results:\n")
print(f"Temperature={T_m}, Irradiance={S_m}")
print(f"Actual fault vector:")
print(f"   {fault_vector}")
print(f"Fault vector identified by {case_name_1}: ")
print(f"   {x_adahessian}")
print(f"Fault vector identified by {case_name_2}: ")
print(f"   {x_adam}")


# 6. Plot results (Loss, Gradient norm, DFFSM-estimated I-V curve)
fig1, ax = plt.subplots(1,2, figsize=(8, 5))
ax[0].title.set_text("Loss")
ax[0].plot(steps, linewidth = 1.5, color='r')
ax[0].plot(steps_2, linewidth = 1.5, color='b')
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("Loss")
ax[0].legend([case_name_1, case_name_2])
ax[0].grid()

ax[1].title.set_text("I-V Curve")
ax[1].plot(V_adahessian, I_m, '-',linewidth = 1.5, color='r')
ax[1].plot(V_adam, I_m, '-',linewidth = 1.5, color='b')
ax[1].plot(V_m, I_m, '-',linewidth = 1, color='k')
ax[1].set_xlabel("Voltage [V]")
ax[1].set_ylabel("Current [A]")
ax[1].legend([case_name_1, case_name_2, 'actual'])
ax[1].grid()



plt.show()