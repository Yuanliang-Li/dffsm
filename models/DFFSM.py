"""
Description: A class for differentiable fast fault simulation model for photovoltaic (PV) string (DFFSM).
Version: 1.0
Authors: Yuanliang Li, Zenan Yang
Latest Update: 2025-12-17
"""

from scipy import constants
import torch


# User-defined autograd functions
class Voltage_Reverse_Bias_Agf(torch.autograd.Function):
    """
    Cell voltage forward and backward calculations based on PyTorch's custom autograd Functions.
    """

    @staticmethod
    def forward(ctx, Iph, Rsh, parms):
        """
        forward function for voltage calculation
        """

        I, I0, Rs, nVth = parms['I'], parms['I0'], parms['Rs'], parms['nVth']
        Re_a, Re_Vb, Re_m = parms['Re_a'], parms['Re_Vb'], parms['Re_m']
        it_num = parms['it_num']

        V = DFFSM.voltage_reverse_bias(I, Iph, I0, Rs, Rsh, nVth, Re_a, Re_Vb, Re_m, it_num)

        ctx.save_for_backward(I, V, Iph, I0, Rs, Rsh, nVth)
        ctx.Re_a = Re_a
        ctx.Re_Vb = Re_Vb
        ctx.Re_m = Re_m

        return V

    @staticmethod
    def backward(ctx, grad_output):
        """
        backward function for voltage differentiation
        """
        I, Vc, Iph, I0, Rs, Rsh, nVth = ctx.saved_tensors
        Re_a = ctx.Re_a
        Re_Vb = ctx.Re_Vb
        Re_m = ctx.Re_m

        dVdIph, dVdRsh = DFFSM.dVdIph_dVdRsh(I, Vc, I0, Rs, Rsh, nVth, Re_a, Re_Vb, Re_m)

        return grad_output * dVdIph, grad_output * dVdRsh, None

voltage_reverse_bias_agf = Voltage_Reverse_Bias_Agf.apply





class DFFSM:
    """
    A class for differential fast fault simulation model (DFFSM) for PV string.
    """

    def __init__(self, pvsinfo, solver, iv_table=None):

        self.Nmc = pvsinfo['Module']['Nmc'] # number of PV cells in one PV module
        self.Nmd = pvsinfo['Module']['Nmd'] # number of bypass diodes in the PV module
        self.NShortSide = pvsinfo['Module']['NShortSide'] # number of PV cells in the long side of the PV module
        self.NLongSide = int(self.Nmc/self.NShortSide) # number of PV cells in the short side of the PV module
        self.Nm =  pvsinfo['Nm'] # number of PV modules in a string
        self.Nsubc = int(self.Nmc/self.Nmd) # number of series PV cells in a sub-string
        self.Nsub = int(self.Nm*self.Nmd) # number of sub-strings in a PV string
        self.InstallType = pvsinfo['InstallType'] # installation type (vertical installation, horizontal installation)

        self.technology = pvsinfo['Module']['Technology'] # Crystalline Silicon
        self.Pmp_stc =  pvsinfo['Module']['Pmp_stc']/self.Nmc # MPPT power under STC
        self.Vmp_stc =  pvsinfo['Module']['Vmp_stc']/self.Nmc # MPPT power voltage under STC
        self.Imp_stc =  pvsinfo['Module']['Imp_stc'] # MPPT power current under STC
        self.Voc_stc = pvsinfo['Module']['Voc_stc']/self.Nmc #/self.Nmc  # Open-circuit voltage under STC
        self.Isc_stc = pvsinfo['Module']['Isc_stc']  # short-circuit current under STC
        self.tc_Isc = pvsinfo['Module']['tc_Isc']  # temperature coefficient of shor-circuit current (a)
        self.tc_Voc = pvsinfo['Module']['tc_Voc']  # temperature coefficient of open-circuit voltage (b)
        self.ic_Voc = pvsinfo['Module']['ic_Voc'] # irradiance coefficient of open-circuit voltage (c)
        self.tc_Pmax = pvsinfo['Module']['tc_Pmax'] # temperature coefficient of Pmax

        self.Iph_ref = pvsinfo['Module']['Iph_ref']  # Iph under STC condition
        self.I0_ref = pvsinfo['Module']['I0_ref']  # Io under STC condition
        self.Rs_ref = pvsinfo['Module']['Rs_ref']/self.Nmc   # Rs under STC condition
        self.Rsh_ref = pvsinfo['Module']['Rsh_ref']/self.Nmc  # Rsh under STC condition
        self.n_ref = pvsinfo['Module']['n_ref']  # diode ideality factor under STC (used by pvsyst)
        self.Adjust = pvsinfo['Module']['Adjust']   # % defined by CEC for adjusting tc_Isc
        self.Re_a = pvsinfo['Module']['Re_a']    # 雪崩击穿所涉及的欧姆电流分数
        self.Re_Vb = pvsinfo['Module']['Re_Vb']  # -21.29;%%结击穿电压
        self.Re_m = pvsinfo['Module']['Re_m']    # 击穿指数

        self.Sref = 1000        # irradiance under STC (W/m2)
        self.Tref = 25          # temperature under STC (degC)
        # self.k = constants.value('Boltzmann constant in eV/K') # boltzmann constant in eV/K 8.617332478e-05 (not clear)
        self.k = constants.k     # boltzmann constant in J/K
        self.q = constants.e       # Charge constant
        self.Eg_ref = 1.121     # energy band, 1.12eV for xtal Si, ?1.75 for amorphous Si.
        self.dEgdT = -0.0002677
        self.diode_con_v = 0.6 # Bypass diode conduction voltage 0.6V

        # default fault-related parameters
        self.max_TS_num = 3 # maximum numer of different TS
        ep = 1e-3
        self.bounds = torch.tensor([[1, self.Nsub],  # n_s1
                                    [1, self.Nsubc], # n_c1
                                    [0, 0.9],        # r_1
                                    [1, self.Nsub],  # n_s2
                                    [1, self.Nsubc], # n_c2
                                    [0, 1],          # r_2
                                    [0, self.Nsub],  # n_sc
                                    [0, 1],          # d_oc1
                                    [0, 1]]          # d_oc2]
                                    , dtype=torch.float64)

        self.solver = solver


    def adjust_bounds(self, S):
        """
        The searching bounds will be updated based on the measured irradiance S.
        """

        self.bounds[2][1] = 1 - 50 / S
        self.bounds[5][1] = 1 - 50 / S


    def forward(self, I_m, T_m, S_m, fault_vector):
        """
        forward function of DFFSM

        Inputs
        I_m: [N] measured I-V current sequence (should be tensor)
        T_m: [1] measured temperature  (should be tensor)
        S_m: [1] measured irradiance (should be tensor)
        fault_vector (should be tensor):
            fault_vector = [n_s1, n_c1, r_1, n_s2, n_c2, r_2, n_sc, d_oc1, d_oc2, Rc]
            0   n_s1 (int [0~Nsub]): number of sub-strings affected by shadow-1 (starting from first substring)
            1   n_c1 (int [0~Nsubc]): number of PV cells in each substring affected by shadow-1
            2   r_1 (float [0~1]): shading ratio affected by shadow-1 (percent of lost irradiance)
                                    effective irradiance = S_m * (1-r_1)
            3   n_s2 (int [0~Nsub]): number of sub-strings affected by shadow-2 (starting after the first shadow)
            4   n_c2 (int [0~Nsubc]): number of PV cells in each substring affected by shadow-2
            5   r_2 (float [0~1]): shading ratio affected by shadow-2
                                    effective irradiance = S_m * (1-r_2)
            6   n_sc (int [0~Nsub-n_s1-n_s2]): number of bypass diodes short-circuited
            7   d_oc1 (int {0,1}): existing of bypass diode open-circuit on the first shadow
            8   d_oc2 (int {0,1}): existing of bypass diode open-circuit on the second shadow
            9   Rc (float [0~10]): resistance degradation

        Returns
        V: string voltage tensor
        """

        I_m = I_m.unsqueeze(0).repeat(self.max_TS_num, 1).T

        # fault_vector.requires_grad = True
        n_s1 = fault_vector[0]
        n_c1 = fault_vector[1]
        r_1 = fault_vector[2]
        n_s2 = fault_vector[3]
        n_c2 = fault_vector[4]
        r_2 = fault_vector[5]
        n_sc = fault_vector[6]
        d_oc1 = fault_vector[7]
        d_oc2 = fault_vector[8]
        # Rc = fault_vector[9]
        Rc = 0

        """1. get model parameters (n_ref_, Rs_ref_, Rsh_ref_)"""
        n_ref_, Rs_ref_, Rsh_ref_ = self.get_opt_parms()


        """2. calculate 5 components (Iph, I0, Rs, Rsh, nVth) under different input conditions"""
        S_type = torch.zeros(3, dtype=torch.float64)
        S_type[0] = (1-r_1)*S_m
        S_type[1] = (1-r_2)*S_m
        S_type[2] = S_m
        T_type = torch.tensor([T_m, T_m, T_m])

        if self.solver['5paras-method'] == 'desoto':
            calc_5params = self.calc_5params_desoto
        else:
            calc_5params = self.calc_5params_cec

        Iph, I0, Rs, Rsh, nVth = calc_5params(effective_irradiance=S_type, # Iph and Rsh are irradiance-dependent
                                              temp_cell=T_type,
                                              alpha_sc=self.tc_Isc,
                                              n_ref=n_ref_,
                                              Iph_ref=self.Iph_ref,
                                              I0_ref=self.I0_ref,
                                              Rs_ref=Rs_ref_,
                                              Rsh_ref=Rsh_ref_,
                                              Adjust=self.Adjust)


        """3. calculate cell voltage under different input conditions"""
        v = voltage_reverse_bias_agf(Iph, Rsh, {'I':I_m, 'I0':I0, 'Rs':Rs, 'nVth':nVth,
                                                'Re_a':self.Re_a, 'Re_Vb':self.Re_Vb, 'Re_m':self.Re_m,
                                                'it_num':self.solver['it_num']})

        """4. get cell counts for different substring types under different input conditions (stored in Qcell)"""
        Qcell = torch.zeros(6,3, dtype=torch.float64)
        Qcell[0][0], Qcell[0][2] = n_c1, self.Nsubc-n_c1
        Qcell[1][0], Qcell[1][2] = n_c1, self.Nsubc-n_c1
        Qcell[2][1], Qcell[2][2] = n_c2, self.Nsubc-n_c2
        Qcell[3][1], Qcell[3][2] = n_c2, self.Nsubc-n_c2
        Qcell[4][2] = self.Nsubc
        Qcell[5][2] = self.Nsubc


        """5. calculate substring voltage for different substring types"""
        Vsub = torch.mm(v, Qcell.T)

        b1 = (Vsub[:,[0,2,5]] < -self.diode_con_v).int() # sub-string under normal diodes
        Vsub[:,[0,2,5]] = Vsub[:,[0,2,5]]*(1-b1) - self.diode_con_v*b1
        Vsub[:,4] = 0 # substring under short-circuited diodes

        """6. get substring counts for different substring types (stored in Qsub)"""
        Qsub = torch.zeros(6, dtype=torch.float64)
        Qsub[0] = n_s1 - d_oc1
        Qsub[1] = d_oc1
        Qsub[2] = n_s2 - d_oc2
        Qsub[3] = d_oc2
        Qsub[4] = n_sc
        Qsub[5] = self.Nsub-n_s1-n_s2-n_sc

        """7. calculate the string voltage"""
        Vstr = torch.mv(Vsub, Qsub) - I_m[:,0]*Rc

        result = dict({'I': I_m[:,0],
                       'V': Vstr,
                       'Vc':v})

        return result


    @staticmethod
    def f_rb(I, v, Iph, I0, Rs, Rsh, nVth, Re_a, Re_Vb, Re_m):
        """
        F function for reverse-biased single diode model
        """
        f = Iph-I0*(torch.exp((v+I*Rs)/nVth)-1)-(v+I*Rs)/Rsh-Re_a*(v+I*Rs)/Rsh*(1-(v+I*Rs)/Re_Vb)**(-Re_m) - I
        return f

    @staticmethod
    def df_rb(I, v, I0, Rs, Rsh, nVth, Re_a, Re_Vb, Re_m):
        """
        dF/dV for reverse-biased single diode model
        """
        df = -I0*torch.exp((I*Rs+v)/nVth)/nVth-1/Rsh-Re_a*(1-(I*Rs+v)/Re_Vb)**(-Re_m)/Rsh \
            - Re_a*(I*Rs+v)*(1-(I*Rs+v)/Re_Vb)**(-Re_m)*Re_m/(Rsh*Re_Vb*(1-(I*Rs+v)/Re_Vb))
        return df

    @staticmethod
    def voltage_initial_guess_lambertw(I, Iph, I0, Rs, Rsh, nVth, v_min):
        """
        initial guess of cell voltage based on LambertW function under traditional single diode model
        """

        log_x = torch.log(I0*Rsh/nVth) + Rsh*(I0-I+Iph)/nVth

        threshold = torch.log(torch.tensor([2.26445]))
        c = 1.546865557
        d = 2.250366841
        logterm = torch.where(log_x < threshold, torch.log(c * torch.exp(log_x) + d), log_x)
        a = (log_x < threshold).int()*0.737769969


        loglogterm = torch.log(logterm)
        w = a + logterm - loglogterm + loglogterm / logterm
        z = (w * w + torch.exp(log_x - w)) / (1.0 + w)

        v = -nVth * z + I0*Rsh - I*Rs - I*Rsh + Iph*Rsh
        b = (v < (v_min+3)).int()
        v = b*(v_min + 2) + (1-b)*v

        return v


    @staticmethod
    def voltage_reverse_bias(I, Iph, I0, Rs, Rsh, nVth, Re_a, Re_Vb, Re_m, it_num):
        """
        calculate cell voltage based on the reverse-biased single diode model with Newton non-recursive method.
        The initial guess of the voltage is obtained by solving traditional single diode model using lambert-w function
        """

        v_min = -I*Rs + Re_Vb # find the lower limit of the voltage (asymptote of the everse-biased single diode model)
        v = DFFSM.voltage_initial_guess_lambertw(I, Iph, I0, Rs, Rsh, nVth, v_min) # initial guess of voltage based on lambertW

        for it in range (it_num):
            f = DFFSM.f_rb(I, v, Iph, I0, Rs, Rsh, nVth, Re_a, Re_Vb, Re_m) # Newton f
            df = DFFSM.df_rb(I, v, I0, Rs, Rsh, nVth, Re_a, Re_Vb, Re_m)    # Newton df
            fdf = f/df
            v_next = v - fdf
            b = (v_next < v_min).int() # check if the next voltage is smaller than v_min since it could jump to another solution branch
            v_next = b*(v_min+0.1) + (1-b)*v_next # if is smaller than v_min, make it to be v_min+0.1

            for i in range(2): # quasi-Newton step: the next value should make |f_next)| <= |f|, other wise v_next/=2
                f_next = DFFSM.f_rb(I, v_next, Iph, I0, Rs, Rsh, nVth, Re_a, Re_Vb, Re_m)
                b = (torch.abs(f_next) > torch.abs(f)).int()
                v_next = b*(v_next+v)/2 + (1-b)*v_next
            v = v_next

        return v

    @staticmethod
    def dVdIph_dVdRsh(Ic, Vc, I0, Rs, Rsh, nVth, Re_a, Re_Vb, Re_m):
        """
        Implicit differentiation for cell voltage (Vc)
        dV / dIph = - (dF / dIph) / (dF / dV)
        dV / dRsh = - (dF / dRsh) / (dF / dV)
        """

        dFdIph = 1

        dFdRsh = (Vc + Ic * Rs) / (Rsh * Rsh) + (Re_a * (Ic * Rs + Vc) / (Rsh * Rsh)) * (1 - (Ic * Rs + Vc) / Re_Vb) ** (-Re_m)

        dFdV = - (I0 / nVth) * torch.exp((Ic * Rs + Vc) / nVth) - 1 / Rsh - (Re_a / Rsh) * (1 - (Ic * Rs + Vc) / Re_Vb) ** (-Re_m) - (Re_a * Re_m * (Vc + Ic * Rs) / (Re_Vb * Rsh)) * (1 - (Ic * Rs + Vc) / Re_Vb) ** (- Re_m - 1)

        dVdIph = - dFdIph / dFdV
        dVdRsh = - dFdRsh / dFdV

        return dVdIph, dVdRsh



    def get_opt_parms(self): # TODO
        """
        Model parameters of the reverse-biased single diode model should be pre-determined.
        """

        r_n_ref = 1
        r_Rs_ref = 1
        r_Rsh_ref = 1

        return self.n_ref*r_n_ref, self.Rs_ref*r_Rs_ref, self.Rsh_ref*r_Rsh_ref


    def calc_5params_desoto(self, effective_irradiance, temp_cell,
                          alpha_sc, n_ref, Iph_ref, I0_ref, Rs_ref, Rsh_ref, Adjust):
        '''
        A modified version based on the calcparams_desoto() from pvlib
        https://pvlib-python.readthedocs.io/en/v0.9.0/generated/pvlib.pvsystem.calcparams_desoto.html

        Calculates five parameter values for the single diode equation at
        effective irradiance and cell temperature using the De Soto et al.
        model described in [1]. The five values returned by calcparams_desoto
        can be used by singlediode to calculate an IV curve.

        Parameters
        ----------
        effective_irradiance : numeric
            The irradiance (W/m2) that is converted to photocurrent.

        temp_cell : numeric
            The average cell temperature of cells within a module in C.

        alpha_sc : float
            The short-circuit current temperature coefficient of the
            module in units of %/C.

        n_ref : float
            The diode ideality factor

        Iph_ref : float
            The light-generated current (or photocurrent) at reference conditions,
            in amperes.

        I0_ref : float
            The dark or diode reverse saturation current at reference conditions,
            in amperes.

        Rsh_ref : float
            The shunt resistance at reference conditions, in ohms.

        Rs_ref : float
            The series resistance at reference conditions, in ohms.

        Returns
        -------
        Tuple of the following results:

        photocurrent : numeric
            Light-generated current in amperes

        saturation_current : numeric
            Diode saturation curent in amperes

        resistance_series : numeric
            Series resistance in ohms

        resistance_shunt : numeric
            Shunt resistance in ohms

        nVth : numeric
            The product of the usual diode ideality factor (n, unitless),
            number of cells in series (Ns=1), and cell thermal voltage at
            specified effective irradiance and cell temperature.

        References
        ----------
        [1] W. De Soto et al., "Improvement and validation of a model for photovoltaic array performance",
        Solar Energy, vol 80, pp. 78-88, 2006.

        '''

        # Boltzmann constant in eV/K, 8.617332478e-05 = 1.380649e-23/1.602176634e-19 (k (in J/K) / q)
        k = constants.value('Boltzmann constant in eV/K')

        # reference temperature
        Tref_K = self.Tref + 273.15
        Tcell_K = temp_cell + 273.15

        E_g = self.Eg_ref * (1 + self.dEgdT*(Tcell_K - Tref_K))

        nVth = n_ref * k * Tcell_K

        # In the equation for IL, the single factor effective_irradiance is
        # used, in place of the product S*M in [1]. effective_irradiance is
        # equivalent to the product of S (irradiance reaching a module's cells) *
        # M (spectral adjustment factor) as described in [1].
        Iph = effective_irradiance/self.Sref * Iph_ref*(1 + alpha_sc * (Tcell_K - Tref_K))
        I0 = (I0_ref * ((Tcell_K / Tref_K) ** 3) * (torch.exp(self.Eg_ref / (k*(Tref_K)) - (E_g / (k*(Tcell_K))))))
        # Note that the equation for Rsh differs from [1]. In [1] Rsh is given as
        # Rsh = Rsh_ref * (S_ref / S) where S is broadband irradiance reaching
        # the module's cells. If desired this model behavior can be duplicated
        # by applying reflection and soiling losses to broadband plane of array
        # irradiance and not applying a spectral loss modifier, i.e.,
        # spectral_modifier = 1.0.
        # use errstate to silence divide by warning
        # with torch.errstate(divide='ignore'):
        Rsh = Rsh_ref* (self.Sref / effective_irradiance)
        # Rsh = Rsh_ref*torch.ones_like(Iph) # we currently do not consider the irradiance dependence for Rsh

        Rs = Rs_ref*torch.ones_like(Iph)

        # Voc = self.Voc_stc * (1 + beta_oc* (Tcell_K - Tref_K))



        return Iph, I0, Rs, Rsh, nVth


    def calc_5params_cec(self, effective_irradiance, temp_cell,
                          alpha_sc, n_ref, Iph_ref, I0_ref, Rs_ref, Rsh_ref, Adjust):
        '''
        A modified version based on the calcparams_cec() from pvlib
        https://pvlib-python.readthedocs.io/en/v0.9.0/generated/pvlib.pvsystem.calcparams_cec.html

        Calculates five parameter values for the single diode equation at
        effective irradiance and cell temperature using the CEC model.
        The CEC model [2] differs from the De soto et al. by the parameter Adjust.
        The five values returned by calcparams_cec can be used by singlediode to calculate an IV curve.

        Parameters
        ----------
        effective_irradiance : numeric
            The irradiance (W/m2) that is converted to photocurrent.

        temp_cell : numeric
            The average cell temperature of cells within a module in C.

        alpha_sc : float
            The short-circuit current temperature coefficient of the
            module in units of %/C.

        n_ref : float
            The diode ideality factor

        Iph_ref : float
            The light-generated current (or photocurrent) at reference conditions,
            in amperes.

        I0_ref : float
            The dark or diode reverse saturation current at reference conditions,
            in amperes.

        Rsh_ref : float
            The shunt resistance at reference conditions, in ohms.

        Rs_ref : float
            The series resistance at reference conditions, in ohms.

        Adjust : float
            The adjustment to the temperature coefficient for short circuit
            current, in percent

        Returns
        -------
        Tuple of the following results:
        photocurrent : numeric
        saturation_current : numeric
        resistance_series : numeric
        resistance_shunt : numeric
        nVth : numeric


        References
        ----------
        [2] A. Dobos, "An Improved Coefficient Calculator for the California Energy Commission 6 Parameter Photovoltaic Module Model",
        Journal of Solar Energy Engineering, vol 134, 2012.
        '''

        # def calc_5params_desoto(self, effective_irradiance, temp_cell,
        #                   alpha_sc, n_ref, Iph_ref, I0_ref, Rs_ref, Rsh_ref):

        return self.calc_5params_desoto(effective_irradiance, temp_cell,
                                        alpha_sc*(1.0 - Adjust/100),
                                        n_ref, Iph_ref, I0_ref, Rs_ref, Rsh_ref, Adjust)


    def projection(self, fault_vector):
        """
        constrain the fault vector to the feasible region during the identification process
        """
        # 0   n_s1 (int [0~Nsub])
        # 1   n_c1 (int [0~Nsubc])
        # 2   r_1 (float [0~1])
        # 3   n_s2 (int [0~Nsub])
        # 4   n_c2 (int [0~Nsubc])
        # 5   r_2 (float [0~1])
        # 6   n_sc (int [0~Nsub-n_s1-n_s2])
        # 7   d_oc1 (int): {0,1}
        # 8   d_oc2 (int): {0,1}

        fv = torch.clamp(fault_vector, self.bounds[:,0], self.bounds[:,1])

        # handle the constraint: fault_vector[0] + fault_vector[3] + fault_vector[6] - self.Nsub <= 0 using projection
        if fv[0] + fv[3] + fv[6] - self.Nsub > 0:
            x_p = torch.clamp((2*fv[0] - (fv[3]+fv[6]-self.Nsub))/3, min=self.bounds[0,0], max=self.bounds[0,1])
            y_p = torch.clamp((2*fv[3] - (fv[0]+fv[6]-self.Nsub))/3, min=self.bounds[3,0], max=self.Nsub-x_p)
            # z_p2 = (2*fv[6] - (fv[0]+fv[3]-self.Nsub))/3
            z_p = self.Nsub - x_p - y_p

            fv[0] = x_p
            fv[3] = y_p
            fv[6] = z_p

        # handle constraint for r2 (r2 < r1/2)
        if fv[5] - fv[2]/2 > 0:
            fv[5] = fv[2]/2


        return fv
