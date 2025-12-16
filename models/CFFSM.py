"""
File Name: CFFSM.py
Description: A class for code-based fast fault simulation model for PV systems (CFFSM-v3).
             Compared with CFFSM-v2 (i.e., ICFFSM), CFFSM-v3 has following features:
                - CFFSM-v3 includes different methods for 5-parameter calculations, including
                  desoto, cec
                - CFFSM-v3 includes different methods for calculating cell voltages, including
                  LambertW, Newton, reverse-bias with Newton
                - CFFSM-v3 is fully Python-based, which can be easily integrated into various software applications.
                - CFFSM-v3 is well-documented.
File Author: Yuanliang Li
Reference: CFFSM-v2 paper: "An improved code-based fault simulation model for PV module (ICFFSM)"
Latest Updating Time: 2025-03-11
"""

from scipy import constants
import numpy as np


class CFFSM:
    """
    class for code-based fast fault simulation model CFFSM-v3
    """
    def __init__(self, pvsinfo):

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


        # Fault configuration
        self.shadow_list = []
        self.shadow = None
        self.statistic_meta = None


    def IV_SCAN(self, T, S, D, Rc, solver):
        """
        obtain the I-V curve for a PV string

        Inputs:
        T: [Nsub x Nsubc], cell temperature matrix
        S: [Nsub x Nsubc], cell effective irradiance matrix
        D: [Nsub], fault condition of bypass diode
        Rc: cable abnormal degradation value
        solver: dict for solver information
                {'5paras-method', 'voltage-method', 'dI'}

        Outputs:
        I: string current
        V: string voltage
        """

        # 1. Perform statistic on the input T, S, D
        statistic_meta = self.input_statistic(T, S, D)
        TS_type = statistic_meta['TS_type']  # type matrix for T and S
        TS_type_num = statistic_meta['TS_type_num']   # number of (T,S) type
        sub_type = statistic_meta['sub_type']        # type of sub-string
        sub_type_num = statistic_meta['sub_type_num']  # number of different types of sub-string
        sub_type_counts = statistic_meta['sub_type_counts']  # number of sub-strings for each sub-string type

        # 2. Get reference value of n, Rs, Rsh (n_ref_, Rs_ref_, Rsh_ref_)
        n_ref_, Rs_ref_, Rsh_ref_ = self.get_opt_parms()


        # 3. Calculate Iph, I0, Rs, Rsh, nVth under different (T,S) pairs based on
        #    single-diode model using 'desoto' or 'cec'
        if solver['5paras-method'] == 'desoto': # use desoto model
            calc_5params = self.calc_5params_desoto
        elif solver['5paras-method'] == 'cec': # use cec model
            calc_5params = self.calc_5params_cec

        Iph, I0, Rs, Rsh, nVth = calc_5params(effective_irradiance=TS_type[:,1],
                                              temp_cell=TS_type[:,0],
                                              alpha_sc=self.tc_Isc,
                                              n_ref=n_ref_,
                                              Iph_ref=self.Iph_ref,
                                              I0_ref=self.I0_ref,
                                              Rs_ref=Rs_ref_,
                                              Rsh_ref=Rsh_ref_,
                                              Adjust=self.Adjust)

        # 4. Start IV scanning，where the current increases iteratively from 0 to Isc
        #    while performing voltage superposition at the same time

        ## 4.1 Initialize some variables
        I = 0 # current starts from 0
        dI = solver['dI'] # current increment
        Vasum = float('inf') # initialize string voltage
        Ia, Va, Vc = [], [], []  # output

        ## 4.2 Estimate Voc (when I=0) as the initial voltage value when using Newton-related methods
        Voc = self.Voc_stc*(1+self.tc_Voc*(1+self.Adjust/100)*((TS_type[:,0]+273.15)-(self.Tref+273.15))) * (1 + np.log(1+self.ic_Voc*(TS_type[:,1]/self.Sref-1)))
        v = Voc.copy()

        ## 4.3 Choose a cell voltage calculation method from:
        ## 'LambertW': use analytical model to calculate voltage based on LambertW without using iteration
        ##             and without considering reverse-biased part of I-V curves
        ## 'Newton': use Newton method to calculate voltage without considering reverse-biased part of I-V curves
        ## 'Newton-RB': use Newton method to calculate voltage considering reverse-biased part of I-V curves
        if solver['voltage-method'] == 'LambertW':
            get_cell_voltage = self.get_cell_voltage_LambertW
        elif solver['voltage-method'] == 'Newton':
            get_cell_voltage = self.get_cell_voltage_Newton
        elif solver['voltage-method'] == 'Newton-RB':
            get_cell_voltage = self.get_cell_voltage_Newton_rb
        else:
            raise Exception(f"{solver['voltage-method']} is not defined. Please choose from 'LambertW', 'Newton' and 'Newton-RB'.")

        ## 4.4 Scanning loop
        Imax = self.Isc_stc * 1.5
        while Vasum > 0 and I < Imax: # set Imax to limit the iteration numbers

            # calculate cell voltage under I for different TS
            # v: [TS_type_num]
            v = get_cell_voltage(I, v, Iph, I0, Rs, Rsh, nVth)

            # calculate sub-string voltage for different type
            Vtype = sub_type[:,:-1].dot(v)


            # The voltage (Vtype) of the sub-string needs to be updated based on the state of the bypass diode:
            # - If the diode is operating normally and the negative voltage generated by the sub-string exceeds the diode's conduction voltage, the diode will conduct.
            # - If the diode is short-circuited, the sub-string voltage is 0.
            # - If the diode is open-circuited, the sub-string voltage remains unchanged.
            Vtype = np.where(((sub_type[:, -1] == 0) & (Vtype < -self.diode_con_v)), -self.diode_con_v, Vtype)
            Vtype = np.where(sub_type[:, -1] == 1, 0, Vtype) # 利用np.where的高级写法(来自chatGPT)


            # Calculate string voltage considering Rc
            Vasum = Vtype.dot(sub_type_counts) - I*Rc

            # Record string current and voltage
            Ia.append(I)
            Va.append(Vasum)
            Vc.append(v)

            # Current increment
            I += dI

        result = dict({'I': Ia,
                       'V': Va,
                       'Vc':np.array(Vc),
                       'TS_type': TS_type})

        return result

    def get_cell_voltage_Newton_rb(self, I, v0, Iph, I0, Rs, Rsh, nVth):
        """
        calculate the cell voltage based on reverse-biased single diode model equation, as follows:
        I = I_{L} - I_{0} \left (\exp \frac{V + I R_{s}}{nV_{th}} - 1 \right )
        - \frac{V + I R_{s}}{R_{sh}}
        - a \frac{V + I R_{s}}{R_{sh}} \left (1 - \frac{V + I R_{s}}{V_{br}} \right )^{-m}
        """


        er_max = float('inf')
        v = v0
        it = 0
        while er_max > 1e-6 and it < 100: # Generally, it converges after 2 to 6 iterations. If the iterations exceed 100, it may diverge
            v_ = v #.copy()
            f = Iph-I0*(np.exp((v+I*Rs)/nVth)-1)-(v+I*Rs)/Rsh-self.Re_a*(v+I*Rs)/Rsh*(1-(v+I*Rs)/self.Re_Vb)**(-self.Re_m) - I
            df = -I0*np.exp((I*Rs+v)/nVth)/nVth-1/Rsh-self.Re_a*(1-(I*Rs+v)/self.Re_Vb)**(-self.Re_m)/Rsh \
            - self.Re_a*(I*Rs+v)*(1-(I*Rs+v)/self.Re_Vb)**(-self.Re_m)*self.Re_m/(Rsh*self.Re_Vb*(1-(I*Rs+v)/self.Re_Vb))
            v = v - f/df
            er_max = np.max(np.abs(v-v_)) # infinite norm
            it += 1
        # print(it)

        return v

    def get_cell_voltage_Newton(self, I, v0, Iph, I0, Rs, Rsh, nVth):

        er_max = float('inf')
        v = v0
        it = 0
        while er_max > 1e-6 and it < 100: # Generally, it converges after 2 to 6 iterations. If the iterations exceed 100, it may diverge
            v_ = v #.copy()
            f = Iph-I0*(np.exp((v+I*Rs)/nVth)-1)-(v+I*Rs)/Rsh-I
            df = -I0*np.exp((I*Rs+v)/nVth)/nVth - 1/Rsh
            v = v - f/df
            er_max = np.max(np.abs(v-v_)) # infinite norm
            it += 1
        # print(it)

        return v


    def get_cell_voltage_LambertW(self, I, v0, Iph, I0, Rs, Rsh,nVth):
        """
        calculate cell voltage with respect to the current

        :param I: output current of PV cell
        :param v0: initial voltage for iteration
        :param Iph:
        :param I0:
        :param Rs:
        :param Rsh:
        :param nVth:
        :return: output voltage
        """
        # standard exact function (deduced by Maple):
        # v = -nVth * lambertw(I0*Rsh*np.exp(Rsh*(I0-I+Iph)/nVth)/nVth) + I0*Rsh - I*Rs - I*Rsh + Iph*Rsh
        # However, I0*Rsh*np.exp(Rsh*(I0-I+Iph)/nVth)/nVth usually outputs inf, which makes lambertw outputs inf,
        # then, making v inf. This could be a big trouble.
        # One compromised solution is using approximation on lambertw based on log(x),
        # in which log(I0*Rsh*np.exp(Rsh*(I0-I+Iph)/nVth)/nVth) will avoid outputting inf by handling the exp() function.
        # Nevertheless, the approximated lambertw based on log(x) still has low accuracy when input value < 2.
        # High-accuracy lambertw approximation for CFFSM can be investigated! Hope someone can research on it!

        # log_x = log(I0*Rsh*np.exp(Rsh*(I0-I+Iph)/nVth)/nVth) = np.log(I0*Rsh/nVth) + Rsh*(I0-I+Iph)/nVth

        log_x = np.log(I0*Rsh/nVth) + Rsh*(I0-I+Iph)/nVth

        v = -nVth * self.approximated_lambertw(log_x) + I0*Rsh - I*Rs - I*Rsh + Iph*Rsh


        return v


    def input_statistic(self, T, S, D):
        """
        perform statistics on input parameters for CFFSM

        :param T: [Nsub x Nsubc], cell temperature matrix
        :param S: [Nsub x Nsubc], cell effective irradiance matrix
        :param D: [Nsub], fault condition of bypass diode
        :return: statistic_meta
        """

        # 寻找不同的环境参数种类envtype(envnum x 2), eachcellenvtype:每个电池片环境参数的类别
        TS = np.concatenate((T.reshape(T.size, 1), S.reshape(T.size, 1)), axis=1)
        TS_type, cell_TS_type, TS_type_counts  = np.unique(TS, axis=0, return_inverse=True, return_counts=True)
        TS_type_num = TS_type.shape[0]
        # TS_type matrix is shown as follows:
        #       T   S
        # TS_1
        # TS_2
        # ...
        # TS_{TS_type_num}

        # 统计substring的配置
        cell_env_type = cell_TS_type.reshape(self.Nsub, self.Nsubc)
        sub_TS_config = np.zeros((self.Nsub,TS_type_num))
        for i in range(self.Nsub):
            for j in range(TS_type_num):
                sub_TS_config[i, j] = np.sum(cell_env_type[i, :] == j)  # 找到每一串a4中元素和j相等的个数

        # 将子串二极管状态考虑进来
        sub_TSD_config = np.concatenate((sub_TS_config, D.reshape(self.Nsub, 1)), axis=1)
        sub_TSD_config = sub_TSD_config.astype(int)
        # sub_TSD_config matrix is shown as follows:
        #        TS_1   TS_2  ...   TS_{TS_type_num}   D
        # sub-1  (count)      ...                      {0,1,2}
        # sub-2
        # ...
        # sub_{Nsub}

        sub_type, sub_sub_type, sub_type_counts  = np.unique(sub_TSD_config, axis=0, return_inverse=True, return_counts=True)
        sub_type_num = sub_type.shape[0]
        # sub_type matrix is shown as follows:
        #        TS_1   TS_2  ...   TS_{TS_type_num}   D
        # type_1 (count)      ...                      {0,1,2}
        # type_2
        # ...
        # type_{sub_type_num}

        self.statistic_meta = dict({'TS_type': TS_type,    # type matrix for T and S
                          'TS_type_num': TS_type_num, # number of (T,S) type
                          'sub_type': sub_type,       # type of sub-string
                          'sub_type_num': sub_type_num, # number of different types of sub-string
                          'sub_type_counts': sub_type_counts, # number of sub-strings for each sub-string type
                          'sub_TSD_config': sub_TSD_config # configuration for each sub-string
                          })

        return self.statistic_meta


    def get_opt_parms(self): # TODO

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
        I0 = (I0_ref * ((Tcell_K / Tref_K) ** 3) * (np.exp(self.Eg_ref / (k*(Tref_K)) - (E_g / (k*(Tcell_K))))))
        # Note that the equation for Rsh differs from [1]. In [1] Rsh is given as
        # Rsh = Rsh_ref * (S_ref / S) where S is broadband irradiance reaching
        # the module's cells. If desired this model behavior can be duplicated
        # by applying reflection and soiling losses to broadband plane of array
        # irradiance and not applying a spectral loss modifier, i.e.,
        # spectral_modifier = 1.0.
        # use errstate to silence divide by warning
        with np.errstate(divide='ignore'):
            Rsh = Rsh_ref* (self.Sref / effective_irradiance)
            # Rsh = Rsh_ref

        Rs = Rs_ref*np.ones_like(Iph)

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




    def addShadow(self, locX = 0, locY = 0, len = 0, width = 0, shade_ratio = 1):
        """
        add a rectangle shadow
        :param locX: X-axis of left-down point location
        :param locY: Y-axis of left-down point location
        :param len: length of the rectangle
        :param width: width of the rectangle
        :param shade_ratio: ration of shading (0~1, percent of lost irradiance)
        :return: a matrix of shade_ratio for PV cell
        """

        if self.InstallType == 'vertical':  # 安装形式为竖装
            Ynum = self.NLongSide
            Xnum = self.NShortSide * self.Nm
        else:  # 安装形式为横装
            print("Currently only support vertical installation!")
            # Ynum = self.NShortSide
            # Xnum = self.NLongSide * self.Nm
            return

        x1, y1 = locX, locY
        x2, y2 = locX + len, locY + width

        shadow_matrix = np.ones((Xnum, Ynum))  # 初始化辐照度矩阵
        shadow_matrix[x1:x2, y1:y2] = 1-shade_ratio
        self.shadow_list.append(shadow_matrix)

        # get shadow combination
        self.shadow = np.ones((Xnum, Ynum))
        for sd in self.shadow_list:
            self.shadow = self.shadow * sd

        return self.shadow

    def get_irradiance_after_shadow(self, Irr):
        if self.shadow_list:
            return Irr * self.shadow.reshape(self.Nsub, self.Nsubc)
        else:
            return Irr

    def approximated_lambertw(self,log_x):
        """
        approximated lambertw based on log(x) from reference: http://www.machinedlearnings.com/2011/07/fast-approximate-lambert-w.html
        Nevertheless, it is still not quite accurate.
        Another method can be tried in paper "Computation of the Lambert W function in photovoltaic modeling"
        :return:
        """
        threshold = np.log(2.26445)

        c = 1.546865557
        d = 2.250366841

        logterm = np.where(log_x < threshold, np.log(c * np.exp(log_x) + d), log_x)
        a = np.zeros(log_x.size)
        a = np.where(log_x < threshold, 0.737769969, a)
        # if log_x < threshold:
        #     c = 1.546865557
        #     d = 2.250366841
        #     a = 0.737769969
        #     logterm = np.log(c * np.exp(log_x) + d)
        # else:
        #     a = 0
        #     logterm = log_x

        loglogterm = np.log(logterm)
        w = a + logterm - loglogterm + loglogterm / logterm
        z = (w * w + np.exp(log_x - w)) / (1.0 + w)

        return z

