# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 19:21:52 2023

@author: bgrenko
"""
import numpy as np
import math
from scipy.interpolate import RectBivariateSpline 

import json

import copy
import sys


import global_vars as gv

''' 
==============================================================================================================================
------------------------------------------------------------------------------------------------------------------------------
--------------------- MODEL FUNCTIONS
------------------------------------------------------------------------------------------------------------------------------
==============================================================================================================================
'''


# -------------------------------------------------------------
# --- MODEL FUNCTIONS and SPECIE CHARACTERISTICS
# -------------------------------------------------------------


# Define class for all species and their characteristics
class Specie(object):
    _reg = [] # Registry of created instances
    def __init__(self, name, Vc, Tc, w, mu, OH_groups, mol_mass, La, Lb, Lc, Ld, Le, Cpa, Cpb, Cpc, Cpd, Cpe):
        self._reg.append(self)
        self.name = name
        self.Vc = Vc # critical volume [cm3 mol-1]
        self.Tc = Tc # [K]
        self.w = w # omega - acentric factor
        self.mu = mu # [D] unit: Debye - GCS unit of electric dipole moment. (see J. Zhu model)
        self.OH_groups = OH_groups # Number of OH groups in the molecule
        self.mol_mass = mol_mass # molar mass [g/mol], numerically same as molecular mass [Da]
        
        # Values for caluclating thermal conductivity of a component (Lambda_i)
        self.La = La
        self.Lb = Lb
        self.Lc = Lc
        self.Ld = Ld
        self.Le = Le
        
        # Values for caluclating specific heat capacity (Cp_i)
        self.Cpa = Cpa
        self.Cpb = Cpb
        self.Cpc = Cpc
        self.Cpd = Cpd
        self.Cpe = Cpe
        
        # --- Given and calculated values needed for gas_mixture_viscosity function
        # Parameters for calculation of gas viscosities
        self.A = 1.16145
        self.B = 0.14874
        self.C = 0.52487
        self.D = 0.77320
        self.E = 2.16178
        self.F = 2.43787
        
        # Special correction for highly polar substances such as alcohols and acids
        self.kappa = 0.0682 + 4.704*(self.OH_groups / self.mol_mass)
        # Dimensionless dipole moment
        self.mu_r = 131.3 * ( self.mu / (self.Vc * self.Tc)**(1/2) ) 
        
        self.Fc = 1 - 0.275*w + 0.059035*(self.mu_r**4) + self.kappa
        
    # First, calculate the first order solution for local the pure gas viscosities (mu_i)
    # mu_i is calculated according to:
    # The Properties of Gases and Liquids - Book by Bruce E. Poling, John Paul O'Connell, and John Prausnitz
    # p. 9.7 - method of Chung et.al.    
    def mu_i(self, T): 
        T_star = 1.2593 * (T / self.Tc)
        Omega_v = self.A*(T_star**(-self.B)) + self.C*np.exp(-self.D*T_star) + self.E*np.exp(-self.F*T_star)
        self.result = 40.758*( self.Fc*((self.mol_mass * T)**(1/2)) / (self.Vc**(2/3) * Omega_v)) 
        return self.result
        
# Define specie instances
CH3OH = Specie('CH3OH', 118.0, 512.64, 0.557970015, 1.7, 1, 32.042, \
               # Lambda_i values
           8.0364e-5, 0.013, 1.4250e-4, -2.8336e-8, 1.2646e-9, \
               # Cp_i values
               13.93945, 111.30774, -41.59074, 5.482564, 0.052037)
    
H2O = Specie('H2O', 55.95, 647.14, 0.331157716, 1.84, 0, 18.01528, \
             # Lambda_i values
                 0.4365, 0.0529, 1.0053e-5, 4.8426e-8, 2.3506e-5, \
                 # Cp_i values
                 30.092, 6.832514, 6.7934356, -2.534480, 0.0821398)
    
H2 = Specie('H2', 64.20, 32.98, -0.22013078, 0, 0, 2.01568, \
            # Lambda_i values
                -11.9, 0.8870, -9.2345e-4, 5.6111e-7, -0.0026, \
                    # Cp_i values
                33.066178, -11.363417, 11.432816, -2.772874, -0.158558)  

CO2 = Specie('CO2', 94.07, 304.12, 0.403593352, 0, 0, 44.01, \
             # Lambda_i values
                 0.6395, 0.019, 1.5214e-4, -1.1666e-7, 7.8815e-5, \
                     # Cp_i values
                24.99735, 55.18696, -33.69137, 7.948387, -0.136638)  
    
CO = Specie('CO', 93.10, 132.85, 0.053424305, 0.122, 0, 28.01,\
            # Lambda_i values
                0.0185, 0.0918, -3.1038e-5, 8.1127e-9, 9.7460e-7, \
                    # Cp_i values
                25.56759, 6.096130, 4.054656, -2.671301, 0.131021) 

    
def ergun_pressure(p, z, u_s, rho_fluid, d_particle, epsilon, mu_mix, reactor_l, reference_p = 'inlet'):
    """
    Calculate PRESSURE FIELD IN A PBR according to the SEMI-EMPIRICAL ERGUN EQUATION.
    
    Parameters
    ----------
    p :       
        [Pa] Pressure
    z : array       
        [m] Axial distances 
    u_s :           
        [m s-1] Superficial fluid velocity 
    rho_fluid :     
        [kg m-3] Fluid mixture density 
    d_particle :    
        [m] Catalyst particle diameter 
    epsilon :       
        [-] Reactor porosity 
    mu_mix :        
        [Pa s] Fluid mixture dynamic viscosity 
    reactor_l :
        [m] Reactor length
    reference_p : (inlet / outlet)
        [] Given known pressure position
        

    Returns
    -------
    p_field : array       
        [Pa] Pressure

    """
    
    # Calculate components of ergun quation
    k1 = -u_s / (rho_fluid * d_particle)
    k2 = (1-epsilon)/epsilon**3
    k3 = ( (150 * (1-epsilon) * mu_mix) / d_particle ) + 1.75*u_s 
    
    # Calculate pressure drop
    d_p = k1 * k2 * k3 * z
    
    d_p_in = 0
    d_p_out = k1*k2*k3 * reactor_l
    
    # If outlet is given pressure, then we calculate pressure increase towards the inlet. So then we subtract the outlet pressure from everything
    if reference_p == 'outlet':
        d_p -= d_p_out
        d_p_in -= d_p_out
        d_p_out -= d_p_out
    
    p_field = p + d_p
    p_in = p + d_p_in
    p_out = p + d_p_out    
        
        
    return p_field, p_in, p_out



def get_velocity_field(v_in, v_out, tube_l, z_centers):
    """
    INTERPOLATE VELOCITY FIELD from inlet and outlet values of velocity

    Parameters
    ----------
    v_in : 
        [m s-1] Inlet velocity
    v_out : 
        [m s-1] Outlet velocity
    tube_l : 
        [m] Reactor length
    z_centers : array
        [m] Array of z cell center positions

    Returns
    -------
    v_array : array
        [m s-1] Array of velocities at cell centers
    """
    
    # Linear interpolation for v_array values
    v_array = np.interp(z_centers, np.asarray([0, tube_l]), np.asarray([v_in, v_out]))
    
    return v_array



def concentration_to_mole_fraction(C_i):
    """
    Calculate MOLE FRACTIONS FROM GIVEN CONCENTRATIONS

    Parameters
    ----------
    C_i : array
        [mol m-3] Molar concentrations of species i 

    Returns
    -------
    X_i : array
        [-] Mole fractions of species i

    """
    X_i = C_i / sum(C_i)
    
    # Replace zeroes with very small value to avoid division by zero
    # X_i[X_i<1e5] = 1e5 

    return X_i



def gas_mixture_viscosity(X_i, T_C):
    """
    Calculate the (DYNAMIC) VISCOSITY OF GAS MIXTURE

    Parameters
    ----------
    X_i : Array     
        [-] Mole fraction of specie i in order (CH3OH, H2O, H2, CO2, CO)
    T_C :             
        [C] Fluid temperature 

    Returns
    -------
    mu_f :          
        [Pa s] Gas mixture viscosity 

    """
    # Convert temperature in Celsius to Kelvin
    T = T_C + 273.15 
    
    
    # First, calculate the first order solution for local the pure gas viscosities (mu_i)
    # mu_i is calculated according to:
    # The Properties of Gases and Liquids - Book by Bruce E. Poling, John Paul O'Connell, and John Prausnitz
    # p. 9.7 - method of Chung et.al.
    mu_i =[] # empty list of local pure gas viscosities
    
    for i in range(5):
        mu_i.append( Specie._reg[i].mu_i(T) ) # Calculation of each pure gas viscosity is done within the class Specie
    
    # Convert list to np array
    mu_i = np.array(mu_i)
    
    # Resulting viscosies are in microPoise, convert to Pascal second
    mu_i = mu_i * 1e-7
    # This array will be the numerator in caluclating mixture viscosity
    
    # Then, calculate the viscosity of gas mixture, according to:
    # A Viscosity Equation for Gas Mixtures, C.R. Wilke (1950) https://doi.org/10.1063/1.1747673
    
    # Make empty arrays of denominators
    denoms = [] # array of denominators
    
    # Calculate arrays of numerators and denominators
    for i in range(5): # i loop - numerator
        j_list = list(range(5))
        denom_temp_sum = 0 # Temporary sum of denominator 
        for j in j_list: # j loop - denominator
                # numerator of phi
                k1 = ( 1 + (mu_i[i] / mu_i[j])**(1/2) * (Specie._reg[j].mol_mass / Specie._reg[i].mol_mass)**(1/4) )**2
                # denominator of phi
                k2 = math.sqrt(8) * (1 + (Specie._reg[i].mol_mass / Specie._reg[j].mol_mass))**(1/2)
                # phi
                phi_ij = k1 / k2
                # sum of phi 
                denom_temp_sum += phi_ij * X_i[j]
        denoms.append(denom_temp_sum) # Append to list of denominators
    denoms = np.array(denoms) # Convert to array in the end of loop

    mu_f = sum( X_i*mu_i / denoms )
    
    return mu_f


def Cp_species(T_C):
    """
    Calculate the SPECIFIC HEAT CAPACITY for INDIVIDUAL SPECIES at given temperature

    Parameters
    ----------
    T_C :                 
        [C] Temperature

    Returns
    -------
    C_p : array         
        [J mol-1 K-1]
    """
    
    # Convert temperature to Kelvin
    T = T_C + 273.15
    
    t = T/1000 
    
    # make empty array
    C_p = np.zeros((5, np.shape(T)[0] ,np.shape(T)[1]  ))
    
    for i in range(5):
        C_p[i] = Specie._reg[i].Cpa + Specie._reg[i].Cpb*t + Specie._reg[i].Cpc*t**2 + Specie._reg[i].Cpd*t**3 + Specie._reg[i].Cpe*(1/t)**2
    
    return C_p
    
    

    
def Cm_mixture(X_i, C_p):
    """
    Calculate MOLAR HEAT CAPACITY of a mixture at given temperature

    Parameters
    ----------
    X_i : array         
        [-] Molar fractions of components
    C_p : array         
        [J mol-1 K-1] Specific heat capacities of individual species

    Returns
    -------
    cp_mix : float      
        [J mol-1 K-1] Specific heat capacity of a mixture
    """

    # Calculate mixture Cp
    cp_mix = sum( X_i * C_p )
    
    return cp_mix 


def Cp_mixture(X_i, C_p):
    """
    Calculate SPECIFIC HEAT CAPACITY of a mixture at given temperature

    Parameters
    ----------
    X_i : array         
        [-] Molar fractions of components
    C_p : array         
        [J mol-1 K-1] Specific heat capacities of individual species

    Returns
    -------
    cp_mix_kg : float      
        [J kg-1 K-1] Specific heat capacity of a mixture
    """
    
    # Make a 
    C_p_kg = copy.deepcopy(C_p)
    
    for i in range(5):
        C_p_kg[i] = C_p[i] / (Specie._reg[i].mol_mass /1000)
        
    cp_mix_kg = sum( X_i * C_p_kg)
    
    return cp_mix_kg
    
    

def specie_mass(V, C_i):
    """
    Calculate MASS OF INDIVIDUAL SPECIES in a given volume

    Parameters
    ----------
    V :            
        [m3] Gas volume
    C_i : array     
        [mol m-3] Molar concentration of component i

    Returns
    -------
    m_i : array     
        [kg] Mass of specie i

    """
    # Create an empty array of m_i values
    m_i = np.zeros(5)
    for i in range(5):
        m_i[i] = V * C_i[i] * Specie._reg[i].mol_mass
   
    # Convert grams to kilograms (because molar mass is in g mol-1)
    m_i = m_i / 1000
    
    return m_i 

def density_mixture(C_i, V, P, T_C):
    """
    Calculate the DENSITY OF AN IDEAL GAS MIXTURE

    Parameters
    ----------
    C_i : 1D array     
        [mol m-3] Molar concentration of component i
    V : int
        [m3] Volume (of computational cell)
    P : float       
        [Pa] Pressure
    T_C : float      
        [C] Temperature

    Returns
    -------
    rho_mix : 2D array
        [kg m-3] Density of ideal gas mixture
    """
    
    # Convert temperature to Kelvin
    T = T_C + 273.15
    
    # Universal gas constant [K K-1 mol-1]
    R = 8.31446261815324
    
    # Get number of moles in a volume
    n_i = C_i * V
    
    # --- Component individual masses
    # Create an empty array of m_i values
    m_i = np.zeros(5)
    for i in range(5):
        m_i[i] = n_i[i] * Specie._reg[i].mol_mass
   
    # Convert grams to kilograms (because molar mass is in g mol-1)
    m_i = m_i / 1000
    
    n_mix = sum(n_i) # [mol] Total number of moles in a volume
    m_mix = sum(m_i) # [kg] Total gas weight in a given volume
    
    # Average molecular mass in the mixture
    mol_mass_mix = m_mix / n_mix # [kg mol-1]
    
    # According to ideal gas law,
    # rho = (mol_mass * p_ / (R * T)
    rho_mix = (mol_mass_mix * P) / (R*T)
    
    return rho_mix



def mol_density_mixture(C_i):
    """
    Calculate the MOLAR DENSITY OF AN IDEAL GAS MIXTURE. This is used sometimes because most of the script uses molar amounts

    Parameters
    ----------
    C_i : array     
        [mol m-3] Molar concentration of component i

    Returns
    -------
    rho_mix : 
        [mol m-3] Density of ideal gas mixture
    """
    
    rho_mix = sum(C_i)    
    
    return rho_mix


def superficial_mass_velocity(C_i, u_s):
    """
    Calculate SUPERFICIAL MASS VELOCITY within a computational cell

    Parameters
    ----------
    C_i : 3D array
        [mol m-3] Molar concentration arrays
    u_s : 1D array
        [m s-1] Superficial fluid velocity array

    Returns
    -------
    G : 1D array
        [kg m-2 s-1] Superficial mass velocity

    """
    # Make a deep copy as not to mess up the original
    Ci_copy = copy.deepcopy(C_i)
    
    # Calculate the molar weight 
    for i in range(5):
        Ci_copy[i] *= Specie._reg[i].mol_mass # [g m-3]
    
    rho_mix = sum(Ci_copy)/1000 # total density within the computational cell + convert to [kg m-3]
    
    # Get superficial fl
    G = rho_mix * u_s
    
    return G


def SC_ratio_to_concentration(SC_ratio, p, T_C, Q, A_inlet):
    """
    Calculate INLET MOLAR CONCENTRATIONS from given steam to carbon ratio

    Parameters
    ----------
    SC_ratio : float
        [-] Ratio of steam to carbon (water to methanol) at the inlet
    p : float
        [Pa] Pressure at the inlet
    T_C : float         
        [C] Temperature at the inlet 
    Q : float         
        [m3 s-1] Inlet flow rate

    Returns
    -------
    C_i : array     
        [mol m-3] Molar concentration

    """
    
    # Convert temperature to Kelvin
    T = T_C + 273.15 
    
    # Universal gas constant [J K-1 mol-1]
    R = 8.31446261815324
    
    # Ideal gas law to calculate inlet molar concentration
    # C_mix = (Q * p) / (R * T * A_inlet)
    
    # Use ideal gas law to calculate molar concentration of mixture [mol m-3] at the inlet
    C_mix = (p) / (R * T)
    
    # Make an array of concentrations
    C_i = np.zeros(5)
    
    # Get factors 
    steam_factor = SC_ratio / (SC_ratio + 1) 
    methanol_factor = 1 / (SC_ratio + 1)
    
    # Set concentrations of methanol and steam    
    C_i[0] = (C_mix * methanol_factor)
    C_i[1] = (C_mix * steam_factor)
    
    
    return C_i


def molar_flow_rate(C_i, u_s):
    """
    Calculate MOLAR FLOW RATES of species

    Parameters
    ----------
    C_i : array     
        [mol m-3] Molar concentration 
    u_s :           
        [m3 s-1] Superficial fluid velocity

    Returns
    -------
    F_i : array     
        [mol s-1] 

    """

    F_i = u_s * C_i

    return F_i 



def velocity_and_flowrate(W_cat, W_F_CH3OH, SC_ratio, p_tot, T_C, r_tube):
    """
    Calculate SUPERFICIAL FLOW VELOCITY and VOLUMETRIC FLOW RATE

    Parameters
    ----------
    W_cat : float
        [kg] Weight of catalyst in the reactor tube
    W_F_CH3OH : 
        [kg s mol-1] Catalyst weight per methanol molar flow rate 
    SC_ratio : 
        [-] Inlet steam to carbon ratio
    p_tot : 
        [Pa] Flow pressure
    T : 
        [C] Flow temperature
    r_tube : 
        [m] Reactor tube radius

    Returns
    -------
    Q_mix : 
        [m3 s-1] Flow volumetric flow rate
    u_s : 
        [m s-1] Flow superficial velocity

    """
    
    # Universal gas constant [J K-1 mol-1]
    R = 8.31446261815324
    
    # Convert temperature to Kelvin
    T = T_C + 273.15
    
    # Molar flow rate of methanol
    ndot_CH3OH = W_cat / W_F_CH3OH # [mol s-1]
    
    # Molar flow rate of water (steam)
    ndot_H2O = ndot_CH3OH * SC_ratio # [mol s-1]
    
    # Get total molar flow rate
    ndot_mix = ndot_H2O + ndot_CH3OH # [mol s-1]
    
    # Ideal gas law says => V = nRT / p, so
    # Get volumetric flow rate of mixture
    Q_mix = (ndot_mix* R * T) / p_tot  # [m3 s-1]
    
    # We can get (single reactor tube) cross section surface area from tube radius,
    # and cross section area is used to:
    # Get superficial flow velocity 
    u_s = Q_mix / (np.pi * r_tube**2)
    
    
    return Q_mix, u_s


def catalyst_dimensions(cat_shape, cat_dimensions):
    """
    Calculates (equivalent) volume and diameter of catalyst particle

    Parameters
    ----------
    cat_shape : string
        (sphere / cylinder) - choice of catalyst particle shape
    cat_dimensions : 
        Diameter (sphere) or diameter and height (cylinder) of a catalyst particle

    Returns
    -------
    V_cat_part : float
        [m3] Catalyst particle volume
    d_cat_part : float
        [m] Catalyst particle (equivalent) diameter

    """
    
    
    if cat_shape == 'sphere':
        d_cat_part = cat_dimensions # diameter is given for sphere
        V_cat_part = (4 / 3) * np.pi * (d_cat_part / 2)**3 # volume is calculated
        
    elif cat_shape == 'cylinder':
        # Base diameter and height are given for sphere
        # base diameter -> cat_dimensions[0]
        # height -> cat_dimensions[1]
        V_cat_part = np.pi * (cat_dimensions[0]/2)**2 * cat_dimensions[1] # calculate volume first
        d_cat_part = (6 * V_cat_part / np.pi)**(1/3) # calculate EQUIVALENT SPHERE DIAMETER FOR VOLUME
        
    
    return V_cat_part, d_cat_part



def porosity(d_ti, d_p, shape='sphere'):
    """
    Calculate POROSITY OF A RANDOMLY PACKED BED
    Expressions taken from https://doi.org/10.1080/02726350590922242

    Parameters
    ----------
    d_ti :              
        [m] Reactor tube inner diameter  
    d_p :               
        [m] Catalyst particle diameter (equivalent sphere diameter for cylinders)
    shape : string      
        [sphere / cylinder] Choice of catalyst particle shape

    NOTE:
    Used expressions are valid in a specific range:
        Spherical:      1.5 <= d_t_i/d_p <= 50
        Cylinders:      1.7 <= d_t_i/d_p <= 26.3

    Returns
    -------
    epsilon :           
        [-] Packed bed porosity (0 <= epsilon <= 1)

    """
    
    if shape == 'cylinder':
        ratio = d_ti/d_p
        
        if not 1.7 <= ratio <= 26.3:
            print('WARNING: ratio of diameters of reactor tube to catalyst particle not ideal.')
            print('Recomended: 1.7 <= d_ti/d_p <= 26.3 for CYLINDRICAL catalyst pellets')
            print('Calculated:', ratio)
            print('Porosity evaluation may be incorrect')
            print('\n')
        
        epsilon = 0.373 + 1.703/( (d_ti / d_p) + 0.611)**2
    
    elif shape == 'sphere':
        ratio = d_ti/d_p
        
        if not 1.5 <= ratio <= 50:
            print('WARNING: ratio of diameters of reactor tube to catalyst particle not ideal.')
            print('Recomended: 1.5 <= d_ti/d_p <= 50 for SPHERICAL catalyst pellets')
            print('Calculated:', ratio)
            print('Porosity evaluation may be incorrect')
            print('\n')
        
        epsilon = 0.39 + 1.740/( (d_ti / d_p) + 1.140)**2
    
    else: 
        print('WARNING: Specification of catalyst pellet shape not recognized:', shape)
        print('Proceeding with calculations for SPHERICAL catalyst')
        print('\n')
        
        ratio = d_ti/d_p
        
        if not 1.5 <= ratio <= 50:
            print('WARNING: ratio of diameters of reactor tube to catalyst particle not ideal.')
            print('Recomended: 1.5 <= d_ti/d_p <= 50 for SPHERICAL catalyst pellets')
            print('Calculated:', ratio)
            print('Porosity evaluation may be incorrect')
            print('\n')
        
        epsilon = 0.39 + 1.740/( (d_ti / d_p) + 1.140)**2
    
    return epsilon



def catalyst_densities(rho_cat, rho_cat_bulk, epsilon, known_cat_density):
    """
    Calculate CATALYST MATERIAL DENSITY or CATALYST BULK (SHIPPING) DENSITY through porosity

    Parameters
    ----------
    rho_cat : 
        [kg m-3] Catalyst material density 
    rho_cat_bulk : 
        [kg m-3] Catalyst bulk (shipping) density
    epsilon : 
        [-] Reactor bed porosity
    known_cat_density : string
        ('density' / 'bulk density' / 'both') Selection of which density is known

    Returns
    -------
    rho_cat : 
        [kg m-3] Catalyst material density 
    rho_cat_bulk : 
        [kg m-3] Catalyst bulk (shipping) density

    """
    
    if known_cat_density == 'density': # if we know material density
        rho_cat_bulk = rho_cat * (1 - epsilon) # calculate bulk density through porosity
        
    elif known_cat_density == 'bulk density': # if we know bulk density
        rho_cat = rho_cat_bulk / (1 - epsilon) # calculate material density through porosity
        
    else: # otherwise, assume we know both densities
        pass
    
    return rho_cat, rho_cat_bulk


def catalyst_cp(cat_composition):
    """
    Calculate the SPECIFIC HEAT CAPACITY OF CuO/ZnO/Al2O3 catalyst

    Parameters
    ----------
    cat_composition : list
        [-] List of molar fractions (percentages) of CuO and ZnO in a catalyst

    Returns
    -------
    cat_cp : float
        [J kg-1 K-1] Specific heat capacity of a catalyst

    """
    # Composition fractions
    cu_f = cat_composition[0]
    zn_f = cat_composition[1]
    al_f = 1 - (cu_f + zn_f)
    
    # --- Molecular weights of components
    cu_mw = 79.545 # CuO [g mol-1]
    zn_mw = 81.38 # ZnO [g mol-1]
    al_mw = 101.96 # Al2O3 [g mol-1]
    
    # --- Specific heat capacities of individual components 
    # Found on internet, some are mean values for range  250-300C, some are for STP, based on what i could find
    cu_cp = 50.1 # CuO [J mol-1 K-1]
    zn_cp = 40.3 # ZnO [J mol-1 K-1]
    al_cp = 109.15 # Al2O3 [J mol-1 K-1]
    
    # 'average' molecular weight of the catalyst pellets
    cat_mw = cu_f*cu_mw + zn_f*zn_mw + al_f*al_mw # [g mol-1]
    
    # 'average' heat capacity of catalyst pellets (using the mixing rule)
    cat_cp_molar = cu_f*cu_cp + zn_f*zn_cp + al_f*al_cp # [J mol-1 K-1]
    
    # Convert units [J mol-1 K-1] -> [J kg-1 K-1]
    cat_cp_g =  cat_cp_molar / cat_mw #[J g-1 K-1]
    cat_cp = cat_cp_g * 1000 # [J kg-1 K-1]
    
    return cat_cp


def radial_diffusion_coefficient(field_v, d_particle, d_ti):
    """
    Calculate the EFFECTIVE RADIAL DIFFUSION COEFFICIENT in a packed bed reactor.

    Parameters
    ----------
    field_v : array           
        [m s-1] Superficial fluid velocity at axial positions
    d_particle :    
        [m] Catalyst particle diameter 
    d_ti : array        
        [m] Reactor single tube inner diameter 

    Returns
    -------
    D_er : 2D array         
        [m2 s-1] Effective radial diffusion coefficient 

    """
    
    # Calculate components for convenience
    k1 =  field_v * d_particle 
    k2 = 1 + 19.4 * (d_particle / d_ti)**2
    
    # Calculate eff. rad. diff. coeff.
    D_er = k1 / (11 * k2)
    
    return D_er 



def aspect_ratio(d_ti, d_p):
    """
    Calculate ASPECT RATIO of the reactor - number of catalyst particles across the tube inner diameter

    Parameters
    ----------
    d_ti :              
        [m] Reactor tube inner diameter  
    d_p :               
        [m] Catalyst particle diameter (equivalent sphere diameter for cylinders)

    Returns
    -------
    N :         
        [-] Aspect ratio

    """
    
    N = d_ti / d_p
    
    return N


def radial_thermal_conductivity(u_s, rho_mix_mol, Cm_mix, Cp_mix, d_p, X_i, T_C, epsilon, N, Ci_near_wall, shape='sphere'):
    """
    Calculate effective RADIAL THERMAL CONDUCTIVITY in a packed bed

    Parameters
    ----------
    u_s : 1D array              
        [m s-] Superficial flow velocity
    rho_mix_mol : 2D array         
        [mol m-3] Gas mixture density
    Cp_mix_mol : 2D array         
        [J mol-1 K-1] Specific heat capacity of fluid mixture
    Cp_mix_kg : 2D array         
        [J kg-1 K-1] Specific heat capacity of fluid mixture
    d_p : float          
        [m] (effective )diameter of catalyst particle
    X_i : 3D         
        [-] Molar fraction of components
    T_C : 2D array            
        [C] Temperature
    epsilon : float      
        [-] Packed bed porosity
    N : float            
        [-] Reactor aspect ratio
    Ci_near_wall : 1D array
        [mol m-3] Concentrations near the tube wall
    shape : string      
        [-] catalyst particle shape (cylinder / sphere)

    Returns
    -------
    Lambda_er : 2D array     
        [W m-1 K-1] Effective radial thermal conductivity
    h_t : 1D array
        [W m-2 K-1] Heat transfer coefficient of the tube-side film
    
    """
    
    # Convert temperature to Kelvin 
    T = T_C + 273.15
    
    # Thermal conductivity of catalyst particles from Bert Koning 
    # https://ris.utwente.nl/ws/portalfiles/portal/6073702/t000003e.pdf
    Lambda_p = 0.21 + T*1.5*1e-4 # [W m-1 K-1]
    
    
    # Lambda_i - thermal conductivites of component i
    Lambda_i = np.zeros((5, np.shape(T)[0] ,np.shape(T)[1]))
    
    for i in range(5):
        Lambda_i[i] = Specie._reg[i].La + Specie._reg[i].Lb*T + Specie._reg[i].Lc*T**2 + Specie._reg[i].Ld*T**3 +Specie._reg[i].Le*(1/T)**2
    
    
    # Lambda_i is in unit [mW m-1 K-1], convert to [W m-1 K-1]
    Lambda_i = Lambda_i * 0.001
    
    # NOTE: Array Lambda_i will be the numerators in calculation of Lambda_f
    
    # Calculate denominators - use np.ones to avoid adding one to each denominator sum for when i = j
    Lambda_f_denoms = np.zeros((5, np.shape(T)[0] ,np.shape(T)[1]))
    
    # Calculate arrays of denominators
    for i in range(5): # i loop - numerator
        j_list = list(range(5))
        for j in j_list: # j loop - denominator
            phi_ij = (Specie._reg[j].mol_mass / Specie._reg[i].mol_mass)**(1/2) 
                
            # sum of phi 
            Lambda_f_denoms[i] += phi_ij * X_i[j] 

    # divide and sum up - zeros in numerator cancel out ones in denominator
    Lambda_f = sum(Lambda_i * X_i /Lambda_f_denoms) # [W m-1 K-1]
    
    
    # Ratio of thermal conductivity of solid cat. particle and gas fluid
    kappa = Lambda_p / Lambda_f # [-]
    
    # Calculate B needed in Lambda_r0  - for this we need some factor C_f based on cat. shape
    C_f_library = {
        'sphere': 1.25,
        'cylinder': 2.5}
    C_f = C_f_library[shape.lower()]
    B = C_f * ( (1-epsilon)/epsilon )**(10/9) # [-]
    
    # Calculate components of Lambda_r0
    # These are all dimensionless [-]
    k1 = 1 - math.sqrt(1 - epsilon)
    k2 = (2 * math.sqrt(1-epsilon)) / ( 1 - B*kappa**(-1) )
    k3 = (B * (1 - kappa**(-1))) * (1 - B * kappa**(-1))**2 
    k4 = np.log( (kappa/B) )
    k5 = ((B - 1) / (1 - B*kappa**(-1)) ) - ((B + 1)/2)
    
    # Calculate Lambda_r0 - stagnant thermal conductivity in packed beds
    Lambda_r0 = Lambda_f * ( k1 + k2*(k3*k4 + k5) ) # [W m-1 K-1]
    
    # --- Calculate Peclet numbers 
    # Peclet radial heat transfer for fully developed turbulent flow
    Pe_h_inf = 8 * (2 - (1 - 2/N)**2) # [-]
    
    # Fluid Peclet number for heat transfer
    Pe_h_0 = (u_s * rho_mix_mol * Cm_mix * d_p) / Lambda_f
    
    # Radial thermal conductivity in packed bed [W m-1 K-1]
    Lambda_er = Lambda_r0 + Lambda_f  *( Pe_h_0/Pe_h_inf)
    # [W m-1 K-1]
    
    # --- Heat transfer coefficient
    # Heat transfer resistance between the reactor tube wall and catalyst bed has to be calculated
    # Necessary when the ratio between tube and catalyst particle is small (d_t / d_p =< 100)  
    # According to Mears https://doi.org/10.1016/0021-9517(71)90073-X

     # The calculations below are only done for wall adjacent cell and thus are 1D arrays with number of axial cells

    # Get gas mixture viscosity for only wall adjacent cells
    mu_mix_wall = gas_mixture_viscosity(X_i[:, 0, :], T_C[0,:])
    
    # Superficial mass velocity 
    G = superficial_mass_velocity(Ci_near_wall, u_s)
    
    # Calculate local Reynolds number in a packed bed
    # https://neutrium.net/fluid-flow/packed-bed-reynolds-number/
    Re = (d_p * G) / (mu_mix_wall * (1 - epsilon))

    # Prandtl number
    Pr = (Cp_mix[0,:] * mu_mix_wall) / Lambda_f[0,:]
    
    # Heat transfer coefficient from Mears https://doi.org/10.1016/0021-9517(71)90073-X
    h_t = (0.4 * Re**(1/2) + 0.2*Re**(2/3)) * Pr**0.4 * (1 - epsilon)/epsilon * Lambda_f[0,:]/d_p
    # [W m-2 K-1]
    return Lambda_er, h_t






def heat_transfer_coeff_tube(u_s, d_p, T_C, C_i, epsilon):
    """
    Calculate heat transfer coefficient on the tube side of the reactor

    Parameters
    ----------
    u_s : 1D array              
        [m s-] Superficial flow velocity
    d_p : float          
        [m] (effective) diameter of catalyst particle
    T_C : 1D array            
        [C] Temperature of near wall cells
    C_i : 2D array            
        [mol m-3] Concentration at near wall cells
    epsilon : float      
        [-] Packed bed porosity
        
    Returns
    -------
    h_t : 1D array
        [W m-2 K-1] Heat transfer coefficient of the tube-side film
    
    """
    
    # Convert temperature to Kelvin 
    T = T_C + 273.15
    
    
    X_i_wall = concentration_to_mole_fraction(C_i) 
    # Get Cp of individual species at temperature T
    Cp_i_wall = Cp_species(T_C)
    # Get Cp of mixture
    Cp_wall_mix = Cp_mixture(X_i_wall, Cp_i_wall)
    
    # Lambda_i - thermal conductivites of component i
    Lambda_i = np.zeros((5, np.shape(T)[0] ,np.shape(T)[1]))
    
    for i in range(5):
        Lambda_i[i] = Specie._reg[i].La + Specie._reg[i].Lb*T + Specie._reg[i].Lc*T**2 + Specie._reg[i].Ld*T**3 + Specie._reg[i].Le*(1/T)**2
    
    # Lambda_i is in unit [mW m-1 K-1], convert to [W m-1 K-1]
    Lambda_i = Lambda_i * 0.001 
    
    # NOTE: Array Lambda_i will be the numerators in calculation of Lambda_f
    
    # Calculate denominators - use np.ones to avoid adding one to each denominator sum for when i = j
    Lambda_f_denoms = np.zeros((5, np.shape(T)[0] ,np.shape(T)[1]))
    
    # Calculate arrays of denominators
    for i in range(5): # i loop - numerator
        j_list = list(range(5))
        for j in j_list: # j loop - denominator
            phi_ij = (Specie._reg[j].mol_mass / Specie._reg[i].mol_mass)**(1/2) 
            
            # sum of phi 
            Lambda_f_denoms[i] += phi_ij * X_i_wall[j] 

    # divide and sum up - zeros in numerator cancel out ones in denominator
    Lambda_f = sum(Lambda_i * X_i_wall /Lambda_f_denoms) # [W m-1 K-1]
    
    # --- Heat transfer coefficient
    # Heat transfer resistance between the reactor tube wall and catalyst bed has to be calculated
    # Necessary when the ratio between tube and catalyst particle is small (d_t / d_p =< 100)  
    # According to Mears https://doi.org/10.1016/0021-9517(71)90073-X

     # The calculations below are only done for wall adjacent cell and thus are 1D arrays with number of axial cells

    # Get gas mixture viscosity for only wall adjacent cells
    mu_mix_wall = gas_mixture_viscosity(X_i_wall, T_C)
    
    G = superficial_mass_velocity(C_i, u_s)
    
    # Calculate local Reynolds number in a packed bed
    # https://neutrium.net/fluid-flow/packed-bed-reynolds-number/
    # Re = (d_p * u_s * rho_wall_mix) / (mu_mix_wall * (1 - epsilon))
    Re = (d_p * G) / (mu_mix_wall * (1 - epsilon))


    # Prandtl number
    Pr = (Cp_wall_mix * mu_mix_wall) / Lambda_f

    
    # Heat transfer coefficient from Mears https://doi.org/10.1016/0021-9517(71)90073-X
    h_t = (0.4 * Re**(1/2) + 0.2*Re**(2/3)) * Pr**0.4 * (1 - epsilon)/epsilon * Lambda_f/d_p
    # [W m-2 K-1]
    return h_t








def get_IO_velocity_and_pressure(p_ref_n, p_set_pos, T_in_n, WF_in_n, SC_ratio, W_cat, epsilon, r_tube, l_tube, d_cat_part, cell_z_centers): 
    
    """
    Calculate inlet,outlet and field of velocity and pressure. Also get some other intermediate variables

    Parameters
    ----------
    p_ref_n : 
        [Pa] Reference pressure
    p_set_pos : string
        (inlet / outlet) Position where the pressure is controllet 
    T_in_n : 
        [C] Inlet temperature
    WF_in_n : 
        [kg s mol-1] Inlet reactor mass feed
    SC_ratio : 
        [-] Steam to carbon ratio of inlet
    W_cat : 
        [kg] Catalyst weight in the reactor tube
    epsilon : 
        [-] Porosity
    r_tube : 
        [m] Reactor tube radius
    l_tube : 
        [-] Reactor tube length
    d_cat_part : 
        [m3] Catalyst particle (equivalent) diameter 
    cell_z_centers : array
        [m] Axial positions of cell centers 

    Returns
    -------
    v_in_n : 
        [m s-1] Inlet flow velocity
    v_out_n : 
        [m s-1] Outlet flow velocity
    field_v : array
        [m s-1] 1D field array of flow velocities - variable only in axial direction
    p_in_n : 
        [Pa] Inlet pressure
    p_out_n : 
        [Pa] Outlet pressure
    field_p : array
        [Pa] 1D array of field pressures
    Q_in_n : 
        [m3 s-1] Inlet volumetric flow
    Q_out_n : 
        [m3 s-1] Outlet volumetric flow
    C_in_n : array
        [mol m-3] Inlet concentrations
    C_out_n : array
        [mol m-3] Outlet Concentrations
    rho_in_n : 
        [kg m-3] Inlet density
    rho_out_n : 
        [kg m-3] Outlet density
    X_in_n : 
        [-] Inlet mole fractions
    X_out_n : 
        [-] Outlet mole fractions
    mu_in_n : 
        [Pa s] Inlet gas mixture viscosity
    mu_out_n : 
        [Pa s] Outlet gas mixture viscosity

    """
    
    if p_set_pos == 'outlet': 
        # First get outlet values if we're setting p_out
        # Vol. flow rate and velocity
        Q_out_n, v_out_n = velocity_and_flowrate(W_cat, WF_in_n, SC_ratio, p_ref_n, T_in_n, r_tube)
        # Concentrations at outlet[mol m-3] - we calculate pressure by assuming that there is no conversion at reactor so outlet concentration is methanol and water
        C_out_n = SC_ratio_to_concentration(SC_ratio, p_ref_n, T_in_n, Q_out_n, (np.pi * r_tube**2))
        # Density of outlet mixture - consisting of only MeOH and H2O
        rho_out_n = density_mixture(C_out_n, 1, p_ref_n, T_in_n) 
        # Get outlet mole fractions
        X_out_n = concentration_to_mole_fraction(C_out_n)
        # Get outlet mixture viscosity
        mu_out_n = gas_mixture_viscosity(X_out_n, T_in_n)
        # Get pressure in cells, inlet, and outlet
        field_p, p_in_n, p_out_n = ergun_pressure(p_ref_n, cell_z_centers, v_out_n, rho_out_n, d_cat_part, epsilon, mu_out_n, l_tube, p_set_pos)
        # Calculate inlet volumetric flow rate and flow velocity
        # v_in  [m s-1] inlet flow velocity (superficial fluid velocity)
        # Q_in  [m3 s-1] inlet volumetric flow rate
        Q_in_n, v_in_n = velocity_and_flowrate(W_cat, WF_in_n, SC_ratio, p_in_n, T_in_n, r_tube)
        # Concentrations at inlet [mol m-3]
        C_in_n = SC_ratio_to_concentration(SC_ratio, p_in_n, T_in_n, Q_in_n, (np.pi * r_tube**2))
        # Density of inlet mixture 
        rho_in_n = density_mixture(C_in_n, 1, p_in_n, T_in_n) 
        # Get inlet mole fractions
        X_in_n = concentration_to_mole_fraction(C_in_n)
        # Get inlet mixture viscosity
        mu_in_n = gas_mixture_viscosity(X_in_n, T_in_n)
    
    elif p_set_pos == 'inlet': 
        # Calculate inlet volumetric flow rate and flow velocity
        # v_in  [m s-1] inlet flow velocity (superficial fluid velocity)
        # Q_in  [m3 s-1] inlet volumetric flow rate
        Q_in_n, v_in_n = velocity_and_flowrate(W_cat, WF_in_n, SC_ratio, p_ref_n, T_in_n, r_tube)
        # Concentrations at inlet [mol m-3]
        C_in_n = SC_ratio_to_concentration(SC_ratio, p_ref_n, T_in_n, Q_in_n, (np.pi * r_tube**2))
        # Density of inlet mixture 
        rho_in_n = density_mixture(C_in_n, 1, p_ref_n, T_in_n) 
        # Get inlet mole fractions
        X_in_n = concentration_to_mole_fraction(C_in_n)
        # Get inlet mixture viscosity
        mu_in_n = gas_mixture_viscosity(X_in_n, T_in_n)
        # Calculate pressure field based on inlet values
        field_p, p_in_n, p_out_n = ergun_pressure(p_ref_n, cell_z_centers, v_in_n, rho_in_n, d_cat_part, epsilon, mu_in_n, l_tube, p_set_pos)
        # Vol. flow rate and velocity
        Q_out_n, v_out_n = velocity_and_flowrate(W_cat, WF_in_n, SC_ratio, p_out_n, T_in_n, r_tube)
        # Concentrations at outlet[mol m-3] - we calculate pressure by assuming that there is no conversion at reactor so outlet concentration is methanol and water
        C_out_n = SC_ratio_to_concentration(SC_ratio, p_out_n, T_in_n, Q_out_n, (np.pi * r_tube**2))
        # Density of outlet mixture - consisting of only MeOH and H2O
        rho_out_n = density_mixture(C_out_n, 1, p_out_n, T_in_n) 
        # Get outlet mole fractions
        X_out_n = concentration_to_mole_fraction(C_out_n)
        # Get outlet mixture viscosity
        mu_out_n = gas_mixture_viscosity(X_out_n, T_in_n)
    
    
    field_v = get_velocity_field(v_in_n, v_out_n, l_tube, cell_z_centers)
    
    
    return v_in_n, v_out_n, field_v, p_in_n, p_out_n, field_p, Q_in_n, Q_out_n, C_in_n, C_out_n, rho_in_n, rho_out_n, X_in_n, X_out_n, mu_in_n, mu_out_n












# -------------------------------------------------------------
# --- REACTION RATE FUNCTIONS and PARAMETERS
# -------------------------------------------------------------

'''
This code uses: 
    - Methanol conversion model developed by developed by Peppley et.al. (1999) https://doi.org/10.1016/S0926-860X(98)00299-3    
    - Conversion model parameters provided by Peppley et.al. (1999) https://doi.org/10.1016/S0926-860X(98)00299-3        
'''


def partial_pressure(p_mix, X_i):
    """
    Calculate PARTIAL PRESSURES of species 

    Parameters
    ----------
    p_mix :             
        [Pa] Mixture pressure 
    X_i : array         
        [-] Molar fraction of specie i

    Returns
    -------
    p_i : array         
        [Pa] Partial pressure of specie i 

    """
    
    p_i = p_mix * X_i
        
    return p_i



# Define class for reaction parameters of kinetic model
class KmodelReactions(object):
    _reg = [] # Registry of created instances
    def __init__(self, name, E, k_inf):
        self._reg.append(self)
        self.name = name    
        self.E = E              # [J mol-1] activation energy for rate constant of reaction   
        self.k_inf = k_inf      # [m2 s-1 mol-1] pre-exponential term in Arrhenius expression (inf = infinity)
        
        
# Define instances    
# Values taken from table 2 in Peppley et.al. (1999) https://doi.org/10.1016/S0926-860X(98)00299-3    
kMSR = KmodelReactions('MSR', 102.8e3, 7.4e14) # Methanol Steam Reforming
kMD = KmodelReactions('MD', 170e3, 3.8e20 ) # Methanol Decomposition      
kWGS = KmodelReactions('WGS', 87.6e3, 5.9e13) # Water Gas Shift
        


# Define class for surface species parameters of kinetic model
class KmodelSpecies(object):
    _reg = [] # Registry of created instances
    def __init__(self, name, DeltaS_1, DeltaS_2, DeltaH_1, DeltaH_2):
        self._reg.append(self)
        self.name = name  
        # DeltaS_i_s  = [J mol-1 K-1] Entropy of adsorption for species i on site s
        self.DeltaS_1 = DeltaS_1
        self.DeltaS_2 = DeltaS_2 
        # DeltaH_i_s = [J mol-1] Heat of adsorption for surface species i on site s OR heat of reaction for formation of surface species i on site s 
        self.DeltaH_1 = DeltaH_1
        self.DeltaH_2 = DeltaH_2
        
        
# Define instances
# Values taken from table 2 in Peppley et.al. (1999) https://doi.org/10.1016/S0926-860X(98)00299-3
kCH3O = KmodelSpecies('CH3O', -41.8, 30, -20e3, -20e3)
kHCOO = KmodelSpecies('HCOO', 179.2, 0, 100e3, 0)
kOH = KmodelSpecies('OH', -44.5, 30, -20e3, -20e3)
# kCO2 = KmodelSpecies('CO2', 0, 0, 0, 0)
kH = KmodelSpecies('H', -100.8, -46.2, -50e3, -50e3)



# Define class for surface concentration sites 
class SurfC(object):
    _reg = [] # Registry of created instances
    def __init__(self, name, C_S1, C_S1a, C_S2, C_S2a):
        self._reg.append(self)
        self.name = name 
        # CS_Si = [mol m-2] Total surface concentration of Site i
        self.C_S1 = C_S1    # 1 - MSR and WGS site
        self.C_S1a = C_S1a  # 1a - MSR and WGS - H2 adsorption site 
        self.C_S2 = C_S2    # 2 - MD site
        self.C_S2a = C_S2   # 2a - MD - H2 adsorption site

# Define instances
# Values taken from Peppley et.al. (1999) https://doi.org/10.1016/S0926-860X(98)00299-3
sites = SurfC('Peppley', 7.5e-6, 1.5e-5, 7.5e-6, 1.5e-5) # Surface concentration sites according to Peppley's model

        
def reaction_rates(p_i, T_C, p_limit=1e-3):
    """
    Calculates reaction rates for MSR, MD, and WGS reactions, according to Peppley et.al. (1999) https://doi.org/10.1016/S0926-860X(98)00299-3   

    Parameters
    ----------
    p_i : array     
        [Pa] Partial pressures of species
    T_C :             
        [C] Temperature
    p_limit :       
        [Pa] Lower partial pressure limits of H2 and CO to avoid division by zero, default is 1e-3

    Returns
    -------
    r_R :           
        [mol s-1 m-2] Reaction rate of MSR
    r_D :           
        [mol s-1 m-2] Reaction rate of MD
    r_W :           
        [mol s-1 m-2] Reaction rate of WGS

    """
    # Convert temperature to Kelvin
    T = T_C + 273.15
    
    # Set partial pressure lower limit
    p_i[p_i<p_limit] = p_limit 
    
    
    # Convert Pascal to bar
    p_i = p_i * 1e-5
    
    
    # Universal gas constant [J K-1 mol-1] 
    R = 8.31446261815324
    
    # --- Arrhenius expressions
    # Rate constants for reaction i [m2 s-1 mol-1]
    k_R = kMSR.k_inf * np.exp( - kMSR.E / (R*T) ) # MSR reaction
    k_D = kMD.k_inf * np.exp( - kMD.E / (R*T) ) # MD reaction
    k_W = kWGS.k_inf * np.exp( - kWGS.E / (R*T) ) # WGS reaction
    
    # Van't Hoff constants for reaction i 
    k_R_eq = np.exp( - (50240 - 170.98*T - 2.64e-2 * T**2)/(R*T))
    k_W_eq = np.exp( - (-41735 + 46.66*T - 7.55e-3 * T**2)/(R*T))
    k_D_eq = k_R_eq / k_W_eq
    
    # --- Equilibrium constant for reaction i on site s - K_i_s [bar^-0.5]
    # Active site 1
    K_CH3O_1 = np.exp( kCH3O.DeltaS_1 / R - kCH3O.DeltaH_1 / (R*T) )
    K_HCOO_1 = np.exp( kHCOO.DeltaS_1 / R - kHCOO.DeltaH_1 / (R*T) )
    K_OH_1 = np.exp( kOH.DeltaS_1 / R - kOH.DeltaH_1 / (R*T) )    
    # K_CO2_1 = np.exp( kCO2.DeltaS_1 / R - kCO2.DeltaH_1 / (R*T) )    
    K_H_1a = np.exp( kH.DeltaS_1 / R - kH.DeltaH_1 / (R*T) )    
    
    # Active site 2
    K_CH3O_2 = np.exp( kCH3O.DeltaS_2 / R - kCH3O.DeltaH_2 / (R*T) )
    # K_HCOO_2 = np.exp( kHCOO.DeltaS_2 / R - kHCOO.DeltaH_2 / (R*T) )
    K_OH_2 = np.exp( kOH.DeltaS_2 / R - kOH.DeltaH_2 / (R*T) )    
    # K_CO2_2 = np.exp( kCO2.DeltaS_2 / R - kCO2.DeltaH_2 / (R*T) )    
    K_H_2a = np.exp( kH.DeltaS_2 / R - kH.DeltaH_2 / (R*T) )    
    
    # --- Reaction rate calculation
    # Top numerators are from Zhu, bottom ones from Peppley
    # MSR rate
    numerator =  k_R*K_CH3O_1 * (p_i[0]/p_i[2]**0.5)*(1 - (p_i[2]**3 * p_i[3])/(k_R_eq*p_i[0]*p_i[1])) * sites.C_S1 * sites.C_S1a 
    # numerator =  k_R*K_CH3O_1 * (p_i[0]/p_i[2]**0.5)*(1 - (p_i[2]**3 * p_i[3])/(k_R*p_i[0]*p_i[1])) * sites.C_S1 * sites.C_S1a 
    denominator = (1 + K_CH3O_1*(p_i[0]*p_i[2]**0.5) + K_HCOO_1*p_i[3]*p_i[2]**0.5 + K_OH_1 * (p_i[1]/p_i[2]**0.5)) * (1 + K_H_1a**0.5 * p_i[2]**0.5)
    r_R = numerator/denominator
    
    
    # MD rate
    numerator = k_D * K_CH3O_2 * (p_i[0]/p_i[2]**0.5)*(1 - (p_i[2]**2 * p_i[4]) / (k_D_eq * p_i[0])) * sites.C_S2 * sites.C_S2a 
    # numerator = k_D * K_CH3O_2 * (p_i[0]/p_i[2]**0.5)*(1 - (p_i[2]**2 * p_i[4]) / (k_D * p_i[0])) * sites.C_S2 * sites.C_S2a 
    denominator = (1 + K_CH3O_2 * (p_i[0]/p_i[2]**0.5) + K_OH_2 * (p_i[1]/p_i[2]**0.5) ) * (1 + K_H_2a**0.5 * p_i[2]**0.5)
    r_D = numerator/denominator
    
    
    # WGS rate
    numerator = k_W * K_OH_1 * (p_i[4]*p_i[1]/p_i[2]**0.5) * (1 - (p_i[2]*p_i[3])/(k_W_eq*p_i[4]*p_i[1])) * sites.C_S1**2
    # numerator = k_W * K_OH_1 * (p_i[4]*p_i[1]/p_i[2]**0.5) * (1 - (p_i[2]*p_i[3])/(k_W*p_i[4]*p_i[1])) * sites.C_S1**2
    denominator = (1 + K_CH3O_1*(p_i[0]/p_i[2]**0.5) + K_HCOO_1*p_i[3]*p_i[2]**0.5 + K_OH_1 * (p_i[1]/p_i[2]**0.5) )**2
    r_W = numerator/denominator
    
    
    return r_R, r_D, r_W




def formation_rates(r_R, r_D, r_W, S_a):
    """
    Calculates RATES OF FORMATION/CONSUMPTION for CH3OH, H2O, H2, CO2, and CO per catalyst weight

    Parameters
    ----------
    r_R :           
        [mol s-1 m-2] Reaction rate for methanol steam reforming reaction
    r_D :           
        [mol s-1 m-2] Reaction rate for methanol decomposition reaction
    r_W :           
        [mol s-1 m-2] Reaction rate for water-gas shift reaction
    S_a :           
        [m2 kg-1] Surface area per weight unit of (fresh) catalyst

    Returns
    -------
    r_CH3OH :       
        [mol s-1 kg-1] Formation/consumption rate of CH3OH per catalyst weight
    r_H2O :         
        [mol s-1 kg-1] Formation/consumption rate of H2O per catalyst weight
    r_H2 :          
        [mol s-1 kg-1] Formation/consumption rate of H2 per catalyst weight
    r_CO2 :         
        [mol s-1 kg-1] Formation/consumption rate of CO2 per catalyst weight
    r_CO :          
        [mol s-1 kg-1] Formation/consumption rate of CO per catalyst weight
    """
    
    r_CO2 = (r_R + r_W)*S_a
    r_CO = (r_D - r_W)*S_a
    r_H2 = 3*r_CO2 + 2*r_CO
    r_CH3OH = - (r_CO2 + r_CO)
    r_H2O = - (r_CO2)
    
    return r_CH3OH, r_H2O, r_H2, r_CO2, r_CO 




def enthalpy_R(C_p, T):
    """
    Calculate temperature dependant ENTHALPY OF METHANOL STEAM REFORMING reaction

    Parameters
    ----------
    C_p : array         
        [J mol-1 K-1] Specific heat capacities of individual species
    T :                 
        [C] Temperature

    Returns
    -------
    H_R :               
        [J mol-1] Reaction enthalpy
    """
    
    # Indices
    # CH3OH     0
    # H2O       1
    # H2        2
    # CO2       3
    # CO        4
    
    H_R = 4.95 * 1e4 + (C_p[3] + 3*C_p[2] - C_p[0] - C_p[1]) * (T + 273.15 - 298)
    
    return H_R

def enthalpy_D(C_p, T):
    """
    Calculate temperature dependant ENTHALPY OF METHANOL DECOMPOSITION reaction

    Parameters
    ----------
    C_p : array         
        [J mol-1 K-1] Specific heat capacities of individual species
    T :                 
        [C] Temperature

    Returns
    -------
    H_D :               
        [J mol-1] Reaction enthalpy
    """
    
    # Indices
    # CH3OH     0
    # H2O       1
    # H2        2
    # CO2       3
    # CO        4
    
    H_D = 9.07 * 1e4 + (C_p[4] + 2*C_p[2] - C_p[0]) * (T + 273.15 - 298)
    
    return H_D

def enthalpy_W(C_p, T):
    """
    Calculate temperature dependant ENTHALPY OF WATER GAS SHIFT reaction

    Parameters
    ----------
    C_p : array         
        [J mol-1 K-1] Specific heat capacities of individual species
    T :                 
        [C] Temperature

    Returns
    -------
    H_W :               
        [J mol-1] Reaction enthalpy
    """
    
    # Indices
    # CH3OH     0
    # H2O       1
    # H2        2
    # CO2       3
    # CO        4
    
    H_W = -4.12 * 1e4 + (C_p[3] + C_p[2] - C_p[1] - C_p[4]) * (T + 273.15 - 298)
    
    return H_W





















'''
==============================================================================================================================
------------------------------------------------------------------------------------------------------------------------------
---------------- DISCRETIZATION FUNCTIONS
------------------------------------------------------------------------------------------------------------------------------
==============================================================================================================================
'''

def discretization_schemes_and_ghost_cells(adv_scheme, diff_scheme):
    """
    Sets the number of ghost cells on each domain side

    Parameters
    ----------
    adv_scheme : str
        Advection scheme choice
    diff_scheme : str
        Diffusion scheme choice

    Returns
    -------
    gcells_z_in : 
        [-] Number of ghost cells at inlet
    gcells_z_out : 
        [-] Number of ghost cells at outlet
    gcells_r_wall : 
        [-] Number of ghost cells at reactor wall
    gcells_r_ax : 
        [-] Number of ghost cells at axis of symmetry
    gcells_r : 
        [-] Number of total radial ghost cells
    gcells_z : 
        [-] Number of total axial ghost cells 
    adv_index : int
        Internal index for selecting advection discretization scheme 
    diff_index : int
    Iinternal index for selecting diffusion discretization scheme

    """
    
    # --- Setting the amount of ghost cells 
    # Axial ghost cells
    if adv_scheme == 'upwind_1o': 
        # gcells_z_in = 1     # Inlet cells
        # gcells_z_out = 0    # Outlet cells
        adv_index = 0       # Internal index for discretization scheme
    elif adv_scheme == 'upwind_2o':
        # gcells_z_in = 2     # Inlet cells
        # gcells_z_out = 0    # Outlet cells    
        adv_index = 1       # Internal index for discretization scheme
        
    # Radial ghost cells
    if diff_scheme == 'central_2o': 
    #     gcells_r_wall = 1       # Wall cells
    #     gcells_r_ax = 1         # Axis cells
        diff_index = 0          # Internal index for discretization scheme
    elif diff_scheme == 'central_4o':
    #     gcells_r_wall = 2       # Wall cells
    #     gcells_r_ax = 2         # Axis cells  
        diff_index = 1          # Internal index for discretization scheme


    # Total ghost cells 
    # gcells_r = gcells_r_wall + gcells_r_ax # Radial
    # gcells_z = gcells_z_in + gcells_z_out # Axial
    
    # gcells_z_in, gcells_z_out, gcells_r_wall, gcells_r_ax, gcells_r, gcells_z,
    
    return adv_index, diff_index









# -------------------------------------------------
# ------ INITIALIZATION
# -------------------------------------------------

# --- Domain discretization 

def uniform_mesh(cells_ax, cells_rad, l_tube, r_tube):
    """
    Calculates GEOMETRIC INFORMATION OF UNIFORMLY DISCRETIZED DOMAIN

    Parameters
    ----------
    cells_ax :                      
        [-] Number of cells in axial direction
    cells_rad :                     
        [-] Number of cells in radial direction
    l_tube :                        
        [m] Reactor tube length
    r_tube :                        
        [m] Reactor tube radius
 
    Returns
    -------
    d_z :                           
        [m] Axial computational cell size
    d_r :                           
        [m] Radial computational cell size       
    cell_z_faces_L : array(1d)      
        [m] Positions of AXIAL LEFT CELL FACES                  
    cell_z_faces_R : array(1d)      
        [m] Positions of AXIAL RIGHT CELL FACES
    cell_z_centers : array(1d)      
        [m] Positions of AXIAL CELL CENTERS
    cell_r_faces_IN : array(2d)     
        [m] Positions of RADIAL INTERNAL CELL FACES
    cell_r_faces_EX : array(2d)     
        [m] Positions of RADIAL EXTERNAL CELL FACES
    cell_r_centers : array(2d)      
        [m] Positions of RADIAL CELL CENTERS
    cell_z_A : array(1d)            
        [m2] Areas of AXIAL CELL BASE FACES
    cell_r_A_IN : array(2d)         
        [m2] Areas of RADIAL CELL INTERNAL FACES
    cell_r_A_EX : array(2d)         
        [m2] Areas of RADIAL CELL EXTERNAL FACES
    cell_V : array(2d)              
        [m3] Cell volumes

    """

    # --- Computational cell sizes
    # [m] Delta l - uniform axial cell size
    d_z = np.ones(cells_ax) * (l_tube / cells_ax)  
    
    # [m] Delta r - radial cell size
    d_r = np.ones(cells_rad) * (r_tube / cells_rad)    
    
    
    # --- Axial cell face positions
    cell_z_faces_R = np.zeros(cells_ax) # define array size 
    for i in range(cells_ax):
        # Positions of right cell faces
        cell_z_faces_R[i] = sum(d_z[:i+1]) 
        
    # Positions of left cell faces    
    cell_z_faces_L = cell_z_faces_R - d_z 
        
    # Axial cell center positions 
    cell_z_centers = (cell_z_faces_R + cell_z_faces_L) / 2
    
    # --- Radial cell face positions
    # Internal cell faces
    cell_r_faces_IN = np.linspace(r_tube-(r_tube / cells_rad), 0, cells_rad)
    # External cell faces
    cell_r_faces_EX = np.linspace(r_tube, (r_tube / cells_rad), cells_rad)
    
    # Radial cell center positions
    cell_r_centers = (cell_r_faces_EX + cell_r_faces_IN) / 2
    
    
    # --- Cell surfaces and volumes
    # Axial face surface areas 
    # The base surface area is always the same on both sides (east and west), so I use one array
    cell_z_A = np.pi * (cell_r_faces_EX**2 - cell_r_faces_IN**2)
    
    # Radial cell surfaces 
    cell_r_A_IN = np.ones((cells_rad, cells_ax))
    cell_r_A_EX = np.ones((cells_rad, cells_ax))
    
    for i in range(cells_rad):
        for j in range(cells_ax):
            cell_r_A_IN[i,j] = 2 * np.pi * cell_r_faces_IN[i] * d_z[j]
            cell_r_A_EX[i,j] = 2 * np.pi * cell_r_faces_EX[i] * d_z[j]
            
    
    # --- Volumes of cells 
    cell_V = np.ones((cells_rad, cells_ax))
    
    for i in range(cells_rad):
        for j in range(cells_ax):
            cell_V[i,j] = cell_z_A[i] * d_z[j]
        

    return d_z, d_r, \
            cell_z_faces_L, cell_z_faces_R, cell_z_centers, \
            cell_r_faces_IN, cell_r_faces_EX, cell_r_centers, \
            cell_z_A, cell_r_A_IN, cell_r_A_EX, \
            cell_V



def interpolate_to_new_mesh(l_tube, r_tube, 
                            cells_ax_old, cells_rad_old, cells_ax_new, cells_rad_new,
                            field_Ci_n, field_T_n, field_BET_cat):
    """
    Interpolate 2D fields to newly defined mesh

    Parameters
    ----------
    l_tube : 
         [m] Reactor tube length
    r_tube : 
        [m] Reactor tube radius
    cells_ax_old : 
        [-] Number of axial cells on old mesh
    cells_rad_old : 
        [-] Number of radial cells on old mesh
    cells_ax_new : 
        [-] Number of axial cells on new mesh
    cells_rad_new : 
        [-] Number of radial cells on new mesh
    field_Ci_n : 3D array
        [mol m-3] Concentration fields for all species
    field_T_n : 2D array
        [C] Temperature field
    field_BET_cat : 2D array
        [m2 kg-1] Field of BET surface area

    Returns
    -------
    return_list : list
        Return list

    """
    
    # --- First get old mesh info
    d_z_old, d_r_old, cell_z_faces_L_old, cell_z_faces_R_old, cell_z_centers_old, \
            cell_r_faces_IN_old, cell_r_faces_EX_old, cell_r_centers_old, \
            cell_z_A_old, cell_r_A_IN_old, cell_r_A_EX_old, \
            cell_V_old = uniform_mesh(cells_ax_old, cells_rad_old, l_tube, r_tube)
    
    
    # --- Make interpolation functions
    func_T          = RectBivariateSpline(np.flip(cell_r_centers_old), cell_z_centers_old, field_T_n)
    func_BET        = RectBivariateSpline(np.flip(cell_r_centers_old), cell_z_centers_old, field_BET_cat)
    func_CH3OH      = RectBivariateSpline(np.flip(cell_r_centers_old), cell_z_centers_old, field_Ci_n[0])
    func_H2O        = RectBivariateSpline(np.flip(cell_r_centers_old), cell_z_centers_old, field_Ci_n[1])
    func_H2         = RectBivariateSpline(np.flip(cell_r_centers_old), cell_z_centers_old, field_Ci_n[2])
    func_CO2        = RectBivariateSpline(np.flip(cell_r_centers_old), cell_z_centers_old, field_Ci_n[3])
    func_CO         = RectBivariateSpline(np.flip(cell_r_centers_old), cell_z_centers_old, field_Ci_n[4])
    
    
    # --- Get new mesh
    d_z, d_r, cell_z_faces_L, cell_z_faces_R, cell_z_centers, \
            cell_r_faces_IN, cell_r_faces_EX, cell_r_centers, \
            cell_z_A, cell_r_A_IN, cell_r_A_EX, \
            cell_V = uniform_mesh(cells_ax_new, cells_rad_new, l_tube, r_tube)
    
    
    # --- Interpolation - get new values
    field_T_n       = func_T(np.flip(cell_r_centers), cell_z_centers)
    field_BET_cat   = func_BET(np.flip(cell_r_centers), cell_z_centers)
    field_CH3OH     = func_CH3OH(np.flip(cell_r_centers), cell_z_centers)
    field_H2O       = func_H2O(np.flip(cell_r_centers), cell_z_centers)
    field_H2        = func_H2(np.flip(cell_r_centers), cell_z_centers)
    field_CO2       = func_CO2(np.flip(cell_r_centers), cell_z_centers)
    field_CO        = func_CO(np.flip(cell_r_centers), cell_z_centers)
    
    # For species fields - join species array them in one 3D array
    field_Ci_n = np.dstack((field_CH3OH, field_H2O, field_H2, field_CO2, field_CO))
    # Shuffle around the axes to match wanted output
    field_Ci_n = np.swapaxes(field_Ci_n, 1, 2)
    field_Ci_n = np.swapaxes(field_Ci_n, 0, 1)
    
    # Make a return list and return
    return_list = [d_z, d_r, cell_z_faces_L, cell_z_faces_R, cell_z_centers, \
            cell_r_faces_IN, cell_r_faces_EX, cell_r_centers, \
            cell_z_A, cell_r_A_IN, cell_r_A_EX, \
            cell_V, \
            field_Ci_n, field_T_n, field_BET_cat]
    
    return return_list




# --- Flow field initialization 


def get_rate_fields(C_field, T_field, field_p, field_BET,
                          pi_limit):
    """
    Get some fields that we wanto save to .json and plot
    Currently that is REACTION RATES, RATES OF SPECIE FORMATION, and VISCOSITY

    Parameters
    ----------
    C_field : 2D array
        [J mol-1 K-1] Specie concentration field
    T_field : 2D array
        [C] Temperature field
    field_p : array
        [Pa] Pressure field
    field_BET : 2D array
        [m2 kg-1] BET area field
    pi_limit : 
        [Pa] Numerical lower limit of specie partial pressure 

    Returns
    -------
    MSR_rate : 
        [mol s-1 m-2] Rate of MSR reaction
    MD_rate : 
        [mol s-1 m-2] Rate of MD reaction
    WGS_rate : 
        [mol s-1 m-2] Rate of WGS reaction
    CH3OH_rate : 
        [mol s-1 kg-1] Rate of CH3OH formation/consumption
    H2O_rate : 
        [mol s-1 kg-1] Rate of H2O formation/consumption
    H2_rate : 
        [mol s-1 kg-1] Rate of H2 formation/consumption
    CO2_rate : 
        [mol s-1 kg-1] Rate of CO2 formation/consumption
    CO_rate : 
        [mol s-1 kg-1] Rate of CO formation/consumption

    """
    
    # Mole fraction
    X_i = concentration_to_mole_fraction(C_field)
    # Partial pressures
    p_i = partial_pressure(field_p, X_i)
    
    # - Reaction rates
    MSR_rate, MD_rate, WGS_rate = reaction_rates(p_i, T_field, pi_limit)            
    # - Rates of formation
    CH3OH_rate, H2O_rate, H2_rate, CO2_rate, CO_rate = formation_rates(MSR_rate, MD_rate, WGS_rate, field_BET)
    
    
    
    return MSR_rate, MD_rate, WGS_rate, CH3OH_rate, H2O_rate, H2_rate, CO2_rate, CO_rate



def get_mass_flow(v_inlet, v_outlet, r_tube, C_inlet, C_field):
    """
    Calculate inlet and outlet mass flows

    Parameters
    ----------
    v_inlet : 
        [m s-1] Inlet flow velocity
    v_outlet : 
        [m s-1] Outlet flow velocity
    r_tube : 
        [m] Reactor tube internal radius
    C_inlet : array 
        [mol m-3] Molar concentration of species at inlet
    C_field : 3D array
        [mol m-3] Molar concentration field of species

    Returns
    -------
    m_inlet : 
        [kg s-1] Inlet mass flow 
    m_outlet : 
        [kg s-1] Outlet mass flow

    """

    # Reactor tube area
    A_tube = r_tube**2 * np.pi
    
    # Get volumetric flows 
    Q_in = v_inlet * A_tube 
    Q_out = v_outlet * A_tube 
    
    # Empty arrays
    C_outlet = np.zeros((5)) # Make empty array for outlet concetrations
    molar_masses = np.zeros((5)) # Make empty array for molar masses
    
    # Fill empty arrays
    for specie in range(5):
        molar_masses[specie] = Specie._reg[specie].mol_mass * 0.001 # Add to array of molar masses and convert from g to kg
        C_outlet[specie] = np.average(C_field[specie, :, -1]) # Add to array of average outlet concentrations

    # Get mass flows
    m_inlet = molar_masses * C_inlet * Q_in 
    m_outlet = molar_masses * C_outlet * Q_out
    
    return m_inlet, m_outlet




# -------------------------------------------------
# ------ TEMPORAL AND SPATIAL TRANSPORT
# -------------------------------------------------

def CFL_criterion(CFL, u, dz):
    """
    Calculate MAXIMUM TIMESTEP SIZE according to the CFL criteion.
    Note: needed only in axial (z) direction for the advection equation.

    Parameters
    ----------
    CFL :           
        [-] Targeted Courant-Friedrichs-Lewy number
    u :             
        [m s-1] Propagation velocity - superficial fluid velocity 
    dz :            
        [m] Minimum axial cell size of the mesh

    Returns
    -------
    dt :            
        [s] Calculated timestep size

    """
    
    dt = abs(CFL*dz/u)
    
    return dt



def advection_flux(phi_P, phi_W, phi_WW, dz, u_s, scheme):
    """
    Calculate advection flux in axial (z) direction

    Parameters
    ----------
    phi_P : 2D/3D array
        Value at point P
    phi_W : 2D/3D array
        Value at point W.
    phi_WW : 2D/3D array
        Value at point WW.
    dz : 2D/3D array 
        [m] Cell axial size
    u_s : 1D array
        [m s-1] Superficial flow velocity (propagation speed)
    scheme : int
        Advection scheme choice

    Returns
    -------
    phi_flux : 2D/3D array
        Advection flux at point P

    """

    # Coefficients for 1st order derivative via upwind scheme
    # [phi_P, phi_W, phi_WW, dz]
    coefficients = [[1, -1, 0, 1], # Upwind first order
                    [3, -4, 1, 2]] # Upwind second order
    
    f = coefficients[scheme] # Multiplication factors

    phi_flux = -u_s * ( (phi_P*f[0] + phi_W*f[1] + phi_WW*f[2]) / (dz*f[3]) )

    return phi_flux


def diffusion_flux(phi_P, phi_EX, phi_EXX, phi_IN, phi_INN, dr, r_P, p_s, scheme):
    """
    Calculate diffusion flux in radial (r) direction

    Parameters
    ----------
    phi_P : 2D/3D array
        Value at point P.
    phi_EX : 2D/3D array
        Value at point EX.
    phi_EXX : 2D/3D array
        Value at point EXX.
    phi_IN : 2D/3D array
        Value at point IN.
    phi_INN : 2D/3D array
        Value at point INN.
    dr : 2D/3D array
        [m] Cell radial size.
    r_P : 
        [m] Cell radial distance from axis of symmetry
    p_s : 1D array
        [] Propagation speed (Lambda_er or D_er)
    scheme : int
        Diffusion scheme choice

    Returns
    -------
    phi_flux : 3D array
        Diffusion flux at point P

    """

    # Coefficients for 1st derivative via central differencing scheme
    # [EXX, EX, IN, INN, dr]
    coeffs_1d = [[0, 1, -1, 0, 2],     # 2nd order CD
                  [-1, 8, -8, 1, 12]]   # 4th order CD
    
    # Coefficients for 2nd derivative via central differencing scheme
    # [EXX, EX, P, IN, INN, dr]
    coeffs_2d = [[0, 1, -2, 1, 0, 1],           # 2nd order
                 [-1, 16, -30, 16, -1, 12]]     # 4th order

    f1 = coeffs_1d[scheme] # Multiplication factors for first derivative
    f2 = coeffs_2d[scheme] # Multiplication factors for second derivative
    
    # first derivative evaluation
    d_phi = (phi_EXX*f1[0] + phi_EX*f1[1] + phi_IN*f1[2] + phi_INN*f1[3]) / (dr*f1[4])
    # Second derivative evaluation
    d2_phi = (phi_EXX*f2[0] + phi_EX*f2[1] + phi_P*f2[2] + phi_IN*f2[3] + phi_INN*f2[4]) / (f2[5]*dr**2)
    
    # Total flux     
    phi_flux = p_s * (d2_phi + (1/r_P)*d_phi)
    
    return phi_flux



def heat_diffusion_flux(phi_P, phi_EX, phi_EXX, phi_IN, phi_INN, dr, r_P, p_s, h_t, scheme):
    """
    Calculate heat diffusion flux in radial (r) direction
    This function has some special treatment for wall cells that are not necessary in mass diffusion

    Parameters
    ----------
    phi_P : 2D/3D array
        Value at point P.
    phi_EX : 2D/3D array
        Value at point EX.
    phi_EXX : 2D/3D array
        Value at point EXX.
    phi_IN : 2D/3D array
        Value at point IN.
    phi_INN : 2D/3D array
        Value at point INN.
    dr : 2D/3D array
        [m] Cell radial size.
    r_P : 
        [m] Cell radial distance from axis of symmetry
    p_s : 1D array
        [] Propagation speed (Lambda_er or D_er)
    h_t : 1D array
        [W m-2 K-1] Heat transfer coefficient of tube-side film
    scheme : int
        Diffusion scheme choice

    Returns
    -------
    phi_flux : 2D array
        Diffusion flux at point P

    """

    # Coefficients for 1st derivative via central differencing scheme
    # [EXX, EX, IN, INN, dr]
    coeffs_1d = [[0, 1, -1, 0, 2],     # 2nd order CD
                  [-1, 8, -8, 1, 12]]   # 4th order CD
    
    # Coefficients for 2nd derivative via central differencing scheme
    # [EXX, EX, P, IN, INN, dr]
    coeffs_2d = [[0, 1, -2, 1, 0, 1],           # 2nd order
                 [-1, 16, -30, 16, -1, 12]]     # 4th order

    f1 = coeffs_1d[scheme] # Multiplication factors for first derivative
    f2 = coeffs_2d[scheme] # Multiplication factors for second derivative
    
    # Multiplication factors for second near wall cell (gets special treatment and is always evaluated w/ 2nd order)
    f1_nw = [0, 1, -1, 0, 2]
    f2_nw = [0, 1, -2, 1, 0, 1]
    
    
    # # first derivative evaluation
    # d_phi = (phi_EXX[1:]*f1[0] + phi_EX[1:]*f1[1] + phi_IN[1:]*f1[2] + phi_INN[1:]*f1[3]) / (dr[1:]*f1[4])
    # # Second derivative evaluation
    # d2_phi = (phi_EXX[1:]*f2[0] + phi_EX[1:]*f2[1] + phi_P[1:]*f2[2] + phi_IN[1:]*f2[3] + phi_INN[1:]*f2[4]) / (f2[5]*dr[1:]**2)
    
    
    ## - Second to near near wall cell
    # first derivative evaluation
    d_phi_nnw = (phi_EXX[1]*f1_nw[0] + phi_EX[1]*f1_nw[1] + phi_IN[1]*f1_nw[2] + phi_INN[1]*f1_nw[3]) / (dr[1]*f1_nw[4])
    # Second derivative evaluation
    d2_phi_nnw = (phi_EXX[1]*f2_nw[0] + phi_EX[1]*f2_nw[1] + phi_P[1]*f2_nw[2] + phi_IN[1]*f2_nw[3] + phi_INN[1]*f2_nw[4]) / (f2_nw[5]*dr[1]**2)
    ## - field
    # first derivative evaluation
    d_phi = (phi_EXX[2:]*f1[0] + phi_EX[2:]*f1[1] + phi_IN[2:]*f1[2] + phi_INN[2:]*f1[3]) / (dr[2:]*f1[4])
    # Second derivative evaluation
    d2_phi = (phi_EXX[2:]*f2[0] + phi_EX[2:]*f2[1] + phi_P[2:]*f2[2] + phi_IN[2:]*f2[3] + phi_INN[2:]*f2[4]) / (f2[5]*dr[2:]**2)
    # Stack these two arrays
    d_phi = np.vstack([d_phi_nnw, d_phi])
    d2_phi = np.vstack([d2_phi_nnw, d2_phi])
    
    
    # --- Near wall cells
    # Coefficients for 1st derivative via central differencing scheme - Wall adjacent cells
    #               [P, IN, INN, dr]
    coeffs_wall_1d = [[1, -1, 0, 1],     # 2nd order CD
                      [3, -4, 1, 2]]   # 4th order CD
    
    # Coefficients for 2nd derivative via central differencing scheme - Wall adjacent cells
    #               [P, IN, INN, INNN, dr]
    coeffs_wall_2d = [[1, -2, 1, 0, 1],           # 2nd order
                      [2, -5, 4, -1, 1]]     # 4th order
    
    f1 = coeffs_wall_1d[scheme] # Multiplication factors for first derivative
    f2 = coeffs_wall_2d[scheme] # Multiplication factors for second derivative
    
    # Fluxes at the wall (r=R)
    phi_wall = - (h_t / p_s[0,:]) * (phi_P[0,:] - phi_EX[0,:])
    # Second derivative at point EX (first cell in the wall)
    phi_EX = -phi_wall/2 
    # it's (0-phi_wall)/2 which is average flux between wall interface and face between EX and EXX
    # the flux between two wall cells (EX and EXX) is assumed zero - uniform temperature in the wall
    
    # first derivative evaluation
    wall_d_phi = (( (phi_P[0]*f1[0] + phi_IN[0]*f1[1] + phi_INN[0]*f1[2]) / (dr[0]*f1[3])) + phi_wall) /2
    # Second derivative evaluation
    wall_d2_phi = (( (phi_P[0]*f2[0] + phi_IN[0]*f2[1] + phi_INN[0]*f2[2] + phi_INN[1]*f2[3]) / (f2[4]*dr[0]**2)) + phi_EX) /2
    
    # Stack the wall derivative array on top of field array
    d_phi = np.vstack([wall_d_phi, d_phi])
    d2_phi = np.vstack([wall_d2_phi, d2_phi])
    
    # Total flux     
    phi_flux = p_s * (d2_phi + (1/r_P)*d_phi)
    
    return phi_flux





def get_neighbour_fields(field_Ci_n, field_T_n, cells_rad, C_in_n, T_in_n, T_wall_n):
    """
    Retrieves fields of neighbouring cells for Ci and T fields, simultaneously setting ghost cells 

    Parameters
    ----------
    field_Ci_n : 3D array
        [J mol-1 K-1] Specie concentration fields
    field_T_n : 2D array
        [C] Temperature field
    cells_rad : int
        [-] Number of cells in radial direction
    C_in_n : 1D array
        [J mol-1 K-1] Specie concentrations at inlet
    T_in_n : float
        [C] Inlet gas temperature
    T_wall_n : 1D array
        [C] Wall temperature array

    Returns
    -------
    field_C_W : 3D array
        	[J mol-1 K-1] Specie concentrations at neighbouring point W (west)
    field_C_WW : 3D array
        	[J mol-1 K-1] Specie concentrations at neighbouring point WW (west west)
    field_C_E : 3D array
        	[J mol-1 K-1] Specie concentrations at neighbouring point E (east)
    field_C_EX : 3D array
        	[J mol-1 K-1] Specie concentrations at neighbouring point EX (external)
    field_C_EXX : 3D array
        	[J mol-1 K-1] Specie concentrations at neighbouring point EXX (external external)
    field_C_IN : 3D array
        	[J mol-1 K-1] Specie concentrations at neighbouring point IN (internal)
    field_C_INN : 3D array
        	[J mol-1 K-1] Specie concentrations at neighbouring point INN (internal internal)
    field_T_W : 2D array
        	[C] Temperature at neighbouring point W (west)
    field_T_WW : 2D array
        	[C] Temperature at neighbouring point WW (west west)
    field_T_E : 2D array
        	[C] Temperature at neighbouring point E (east)
    field_T_EX : 2D array
        	[C] Temperature at neighbouring point EX (external)
    field_T_EXX : 2D array
        	[C] Temperature at neighbouring point EXX (external external)
    field_T_IN : 2D array
        	[C] Temperature at neighbouring point IN (internal)
    field_T_INN : 2D array
        	[C] Temperature at neighbouring point INN (internal internal)

    """
    
    
    # Get fields of neighbouring cells
    field_C_W = np.roll(field_Ci_n, 1, 2) # WEST
    field_C_WW = np.roll(field_Ci_n, 2, 2) # WEST WEST
    field_C_E = np.roll(field_Ci_n, -1, 2) # EAST
    
    field_C_EX = np.roll(field_Ci_n, 1, 1) # EXTERNAL
    field_C_EXX = np.roll(field_Ci_n, 2, 1) # EXTERNAL EXTERNAL
    field_C_IN = np.roll(field_Ci_n, -1, 1) # INTERNAL
    field_C_INN = np.roll(field_Ci_n, -2, 1) # INTERNAL INTERNAL
    
    field_T_W = np.roll(field_T_n, 1, 1)
    field_T_WW = np.roll(field_T_n, 2, 1)
    field_T_E = np.roll(field_T_n, -1, 1)
    
    field_T_EX = np.roll(field_T_n, 1, 0)
    field_T_EXX = np.roll(field_T_n, 2, 0)
    field_T_IN = np.roll(field_T_n, -1, 0)
    field_T_INN = np.roll(field_T_n, -2, 0)
    
    
    # --- Set ghost cells
    # WEST - inlet
    field_C_W[:, :, 0] = np.rot90( np.ones((cells_rad, 5)) * np.asarray(C_in_n), 3 ) # Dirichliet BC
    field_T_W[:,0] = T_in_n # Dirichliet BC
    # WEST WEST - inlet
    field_C_WW[:, :, 0] = np.rot90( np.ones((cells_rad, 5)) * np.asarray(C_in_n), 3 ) # Dirichliet BC
    field_C_WW[:, :, 1] = np.rot90( np.ones((cells_rad, 5)) * np.asarray(C_in_n), 3 ) # Dirichliet BC
    field_T_WW[:,0] = T_in_n # Dirichliet BC
    field_T_WW[:,1] = T_in_n # Dirichliet BC
    # EAST - outlet
    field_C_E[:, :, -1] = field_C_W[:, :, -2] # Neumann BC
    field_T_E[:,-1] = field_T_E[:,-2] # Neumann BC
    # EXTERNAL - wall
    field_C_EX[:, 0, :] =  field_C_EX[:, 1, :] # Neumann BC
    field_T_EX[0, :] = T_wall_n # Dirichliet BC
    # EXTERNAL EXTERNAL - wall
    field_C_EXX[:, 0, :] =  field_C_EXX[:, 2, :] # Neumann BC
    field_C_EXX[:, 1, :] =  field_C_EXX[:, 2, :] # Neumann BC
    field_T_EXX[0, :] = T_wall_n # Dirichliet BC
    field_T_EXX[1, :] = T_wall_n # Dirichliet BC
    # INTERNAL - symmetry axis
    field_C_IN[:, -1, :] =  field_C_IN[:, -2, :] # Neumann BC
    field_T_IN[-1, :] = field_T_IN[-2, :] # Neumann BC
    # INTERNAL INTERNAL 
    field_C_INN[:, -1, :] =  field_C_INN[:, -4, :] # Neumann BC - symmetric
    field_C_INN[:, -2, :] =  field_C_INN[:, -3, :] # Neumann BC
    field_T_INN[-1, :] = field_T_INN[-4, :] # Neumann BC - symmetryc
    field_T_INN[-2, :] = field_T_INN[-3, :] # Neumann BC
    
    
    
    return field_C_W, field_C_WW, field_C_E, field_C_EX, field_C_EXX, field_C_IN, field_C_INN, \
        field_T_W, field_T_WW, field_T_E, field_T_EX, field_T_EXX, field_T_IN, field_T_INN



def RK4_fluxes(dt_RK4, times_RK4, cells_rad, cells_ax, cell_V, cell_r_centers, cell_z_centers,
        dz_mgrid, dr_mgrid, z_centers_mgrid, r_centers_mgrid,
        W_cat, SC_ratio, r_tube, 
        T_wall_RK4, T_in_RK4, WF_in_RK4, p_in_RK4, p_set_pos,
        field_Ci, field_T,
        bulk_rho_c, BET_cat_P, d_cat_part, cat_cp, cat_shape,
        d_tube_in, l_tube,
        epsilon, N, pi_limit, nu_i, nu_j, adv_scheme, diff_scheme):
    
    """
    Calculates RUNGE KUTTA 4TH ORDER FLUXES for concentration Ci and temperature T fields

    Parameters
    ----------
    dt_RK4 : 1D array
        [s] Array of RK4 timestep sizes
    times_RK4 : 1D array
        [s] Array of absolute times in the simulation
    cells_rad : int
        [-] Number of cells in radial direction
    cells_ax : int
        [-] Number of cells in axial direction
    cell_V : array
        [m3] Cell volumes
    cell_r_centers : array
        [m] Positions of cell radial centers
    cell_z_centers : array
        [m] Positions of cell axial centers
    dz_mgrid : 2D array
        [m] Meshgrid of dz sizes
    dr_mgrid : 2D array
        [m] Meshgrid of  dr sizes
    z_centers_mgrid : 2D array
        [m] Meshgrid of z cell centers
    r_centers_mgrid: 2D array   
        [m] Meshgrid of r cell centers
    W_cat : float
        [kg] Weight of catalyst in one reactor tube
    SC_ratio : float
        [-] Steam to carbon ration
    r_tube : float 
        [m] Reactor tube radius
    calculate_T_wall_dynamic_RK4 : function 
        Function to calculate wall temperature profile inside RK4
    T_wall_RK4 : 1D array
        [C] Wall temperature profile
    T_in_RK4 : 1D array
        [C] Inlet fluid temperature at RK4 times
    WF_in_RK : 1D array
        [kg s mol-1] Catalyst weight per methanol molar flow rate at RK4 times
    p_in_RK4 : 1D array
       [Pa] Inlet fluid velocity at RK4 times
    p_set_pos : str
       (inlet / outlet) : Position at which pressure is given
    field_Ci : 3D array
        [J mol-1 K-1] Specie concentration field at time t 
    field_T : 2D array
        [T] Specie temperature field at time t 
    bulk_rho_c : float
        [kg m-3] Bulk catalyst density
    BET_cat_P : 2D array
        [m2 kg-1] Surface area of fresh catalyst per mass
    d_cat_part : float
        [m] Catalyst particle diameter
    cat_cp : float
        [J kg-1 K-1] Catalyst specific heat capacity
    cat_shape : string
        [sphere / cylinder] Shape of catalyst particles
    d_tube_in : float
        [m] Reactor tube inner diameter
    l_tube : float
        [m] Reactor tube length
    epsilon : float
        [-] Packed bed porosity
    N : float
        [-] Aspect ratio
    pi_limit : float
        [bar] Partial pressure lower limit for reaction rate 
    nu_i : array
        [-] Effectivness factor for production / consumption of species i
    nu_j : array
        [-] Effectiveness factor for reaction j
    adv_scheme : int
        Choice of advection scheme
    diff_scheme : int
        Choice of diffusion scheme

    Returns
    -------
    Ci_fluxes : 3D array
        [J mol-1 K-1 s-1] Flux fields of specie concentrations
    T_fluxes : 2D array
        [C s-1] Flux field of temperature/heat
    """
    
    # Make empty flux fields. remember - no ghost cells here 
    C_flux_arrays = np.zeros((4, 5, cells_rad, cells_ax))
    T_flux_arrays = np.zeros((4, cells_rad, cells_ax))
  
    # Make temporary value array
    # Need to do a deep copy, otherwise the original gets changed as well
    C_array = copy.deepcopy(field_Ci)
    T_array = copy.deepcopy(field_T)
        
    # --- Calculate flux F0 first 
    # Some things are already set/calculated for the first flux
    # They are: pressure field, D_er, inlet velocity, ghost cell values
    for RKt in range(1): # do this just to keep syntax the same
        p_set_pos = 'outlet'
        # Get inlet/outlet/field velocities and pressures, and some other variables 
        v_in_RK, v_out_RK, field_v_RK, \
            p_in_RK, p_out_RK, field_p, \
            Q_in_RK, Q_out_RK, C_in_RK, C_out_RK, rho_in_RK, rho_out_RK, \
            X_in_RK, X_out_RK, mu_in_RK, mu_out_RK = get_IO_velocity_and_pressure(p_in_RK4[RKt], p_set_pos, T_in_RK4[RKt], WF_in_RK4[RKt], SC_ratio, W_cat, epsilon, r_tube, l_tube, d_cat_part, cell_z_centers)
        # Radial diffusion coefficient
        D_er_RK = radial_diffusion_coefficient(field_v_RK, d_cat_part, d_tube_in)
        
        # Calculate wall temperature profile
        T_wall_RK4 = gv.Twall_func.dynamic(times_RK4[RKt], dt_RK4[RKt], T_wall_RK4, C_array[:,0,:], T_array[0,:], field_v_RK)
        
        # Get neighbouring cell fields                
        field_C_W, field_C_WW, field_C_E, \
            field_C_EX, field_C_EXX, field_C_IN, field_C_INN, \
            field_T_W, field_T_WW, field_T_E, \
                field_T_EX, field_T_EXX, field_T_IN, field_T_INN = get_neighbour_fields(C_array, T_array, cells_rad, C_in_RK, T_in_RK4[RKt], T_wall_RK4)
                
        # Calculate flux
        C_flux_arrays[RKt], T_flux_arrays[RKt] = Euler_fluxes(D_er_RK, field_v_RK, field_p,
            dz_mgrid, dr_mgrid, cell_V, r_centers_mgrid,
            C_array, field_C_W, field_C_WW, field_C_IN,field_C_INN, field_C_EX, field_C_EXX,
            T_array, field_T_W, field_T_WW, field_T_IN, field_T_INN, field_T_EX, field_T_EXX,
            bulk_rho_c, BET_cat_P, d_cat_part, cat_cp, cat_shape,
            epsilon, N, adv_scheme, diff_scheme, pi_limit, nu_i, nu_j)
            
    # --- Calculate RK4 fluxes 2,3,4
    # First flux already calculated
    for RKt in range(1,4): # RKt = runge kutta time 
        # First, evolve values with previously calculated slopes/fluxes
        C_array = field_Ci + (dt_RK4[RKt] * C_flux_arrays[RKt-1])
        # Temperature / heat           
        T_array = field_T + (dt_RK4[RKt] * T_flux_arrays[RKt-1])
        # !!! CHECK WHETHER nu_is and nu_j change
        
        # Get inlet/outlet/field velocities and pressures, and some other variables 
        v_in_RK, v_out_RK, field_v_RK, \
            p_in_RK, p_out_RK, field_p, \
            Q_in_RK, Q_out_RK, C_in_RK, C_out_n, rho_in_RK, rho_out_RK, \
            X_in_RK, X_out_RK, mu_in_RK, mu_out_RK = get_IO_velocity_and_pressure(p_in_RK4[RKt], p_set_pos, T_in_RK4[RKt], WF_in_RK4[RKt], SC_ratio, W_cat, epsilon, r_tube, l_tube, d_cat_part, cell_z_centers)
        # Calculate new radial diffusion coefficient
        D_er_RK = radial_diffusion_coefficient(field_v_RK, d_cat_part, d_tube_in)

        # Calculate wall temperature profile
        T_wall_RK4 = gv.Twall_func.dynamic(times_RK4[RKt], dt_RK4[RKt], T_wall_RK4, C_array[:,0,:], T_array[0,:], field_v_RK)
        
        # Get fields of neighbouring cells                 
        field_C_W, field_C_WW, field_C_E, \
            field_C_EX, field_C_EXX, field_C_IN, field_C_INN, \
            field_T_W, field_T_WW, field_T_E, \
                field_T_EX, field_T_EXX, field_T_IN, field_T_INN = get_neighbour_fields(C_array, T_array, cells_rad, C_in_RK, T_in_RK4[RKt], T_wall_RK4)
    
        # Calculate fluxes
        C_flux_arrays[RKt], T_flux_arrays[RKt] = Euler_fluxes(D_er_RK, field_v_RK, field_p,
            dz_mgrid, dr_mgrid, cell_V, r_centers_mgrid,
            C_array, field_C_W, field_C_WW, field_C_IN,field_C_INN, field_C_EX, field_C_EXX,
            T_array, field_T_W, field_T_WW, field_T_IN, field_T_INN, field_T_EX, field_T_EXX,
            bulk_rho_c, BET_cat_P, d_cat_part, cat_cp, cat_shape,
            epsilon, N, adv_scheme, diff_scheme, pi_limit, nu_i, nu_j)
                
               
    # --- Finally, make arrays of RK4 fluxes
    Ci_fluxes = (C_flux_arrays[0] + 2*C_flux_arrays[1] + 2*C_flux_arrays[2] + C_flux_arrays[3])/6
    T_fluxes = (T_flux_arrays[0] + 2*T_flux_arrays[1] + 2*T_flux_arrays[2] + T_flux_arrays[3])/6
    

    return Ci_fluxes, T_fluxes



def Euler_fluxes(D_er, v_n, p_P,
        cell_dz, cell_dr, cell_V, cell_r_center,
        Ci_P, Ci_W, Ci_WW, Ci_IN, Ci_INN, Ci_EX, Ci_EXX,
        T_P, T_W, T_WW, T_IN, T_INN, T_EX, T_EXX,
        bulk_rho_c, BET_cat_P, d_cat_part, cat_cp, cat_shape,
        epsilon, N, adv_scheme, diff_scheme, pi_limit, nu_i, nu_j):
    """
    Calculates the EULER FLUXES OF CONCENTRATION AND TEMPERATURE FIELDS for time stepping (dCi, dT for dCi/dt, dT/dt)

    Parameters
    ----------
    D_er : 1D array 
        [m2 s-1] Effective radial diffusion coefficient in packed bed
    v_n : 1D array
        [m s-] Superficial flow velocity
    p_P : 1D array
        [Pa] Pressure at cell P
    cell_dz : 2D array
        [m] Cell axial size
    cell_dr : 2D array
        [m] Cell radial size
    cell_V : 2D array
        [m3] Cell volume
    cell_r_center : 2D array
        [m] Cell center radial position
    Ci_P : 3D array
        [J mol-1 K-1] Specie concentrations at point P
    Ci_W : 3D array
        [J mol-1 K-1] Specie concentrations at neighbouring point W (west)
    Ci_WW : 3D array
        [J mol-1 K-1] Specie concentrations at neighbouring point WW (west west)
    Ci_IN : 3D array
        [J mol-1 K-1] Specie concentrations at neighbouring point IN (internal)
    Ci_INN : 3D array
        [J mol-1 K-1] Specie concentrations at neighbouring point INN (internal internal)
    Ci_EX : 3D array
        [J mol-1 K-1] Specie concentrations at neighbouring point EX (external)
    Ci_EXX : 3D array
        [J mol-1 K-1] Specie concentrations at neighbouring point EXX (external external)
    T_P : 2D array
        [C] Temperature at point P
    T_W : 2D array
        [C] Temperature at neighbouring point W (west)
    T_WW : 2D array
        [C] Temperature at neighbouring point WW (west west)
    T_IN : 2D array
        [C] Temperature at neighbouring point IN (internal)
    T_INN : 2D array
        [C] Temperature at neighbouring point INN (internal internal)
    T_EX : 2D array
        [C] Temperature at neighbouring point EX
    T_EXX : 2D array
        [C] Temperature at neighbouring point EXX
    bulk_rho_c : float
        [kg m-3] Bulk catalyst density
    BET_cat_P : 2D array
        [m2 kg-1] Surface area of fresh catalyst per mass
    d_cat_part : float
        [m] Catalyst particle diameter
    cat_cp : float
        [J kg-1 K-1] Catalyst specific heat capacity        
    cat_shape : string
        [sphere / cylinder] Shape of catalyst particles
    epsilon : float 
        [-] Packed bed porosity
    N : float
        [-] Aspect ratio
    adv_scheme : int
        Choice of advection scheme
    diff_scheme : int
        Choice of diffusion scheme
    pi_limit : float
        [bar] Partial pressure lower limit for reaction rate 
    nu_i : array
        [-] Effectivness factor for production / consumption of species i
    nu_j : array
        [-] Effectiveness factor for reaction j
        

    Returns
    -------
    mass_flux : 3D array 
        [-] Fluxes (dCi / dt) in specie concentration fields
    heat_flux : 2D array
        [-] Flux (dT / dt) in temperature (heat) field 

    """
    # --- Mass transport 
    # Mass advection term
    mass_adv_i = advection_flux(Ci_P, Ci_W, Ci_WW, cell_dz, v_n, adv_scheme)
    
    # Mass diffusion term
    mass_diff_i = diffusion_flux(Ci_P, Ci_EX, Ci_EXX, Ci_IN, Ci_INN, cell_dr, cell_r_center, D_er, diff_scheme)
    
    
    # Mass source term 
    # Mole fractions of species
    X_i = concentration_to_mole_fraction(Ci_P) 
    # partial pressure
    p_i = partial_pressure(p_P, X_i)
    # Reaction rates
    r_R, r_D, r_W = reaction_rates(p_i, T_P, pi_limit)
    # Formation rates
    r_i = formation_rates(r_R, r_D, r_W, BET_cat_P)
    r_i = np.asarray(r_i) # Convert this to np array
    # Mass source / sink
    mass_source_i = (nu_i * bulk_rho_c * r_i) 
    
    # Mass flux for time stepping (dCi / dt)
    mass_flux = (mass_adv_i + mass_diff_i + mass_source_i)/epsilon
    
    # --- Heat transport 
    # Specific heat capacity of every specie
    Cp_i = Cp_species(T_P)
    # Mixture heat capacity - specific and molar
    Cm_mix = Cm_mixture(X_i, Cp_i) # [J mol-1 K-1]
    Cp_mix = Cp_mixture(X_i, Cp_i) # [J kg-1 K-1]
    # Density of a mixture 
    rho_mix_mol = mol_density_mixture(Ci_P) 
    # Effective radial thermal conductivity
    Lambda_er, h_t = radial_thermal_conductivity(v_n, rho_mix_mol, Cm_mix, Cp_mix, d_cat_part, X_i, T_P, epsilon, N, Ci_P[:,0,:], cat_shape)    
    
    # Heat advection term
    heat_coeff = v_n  * Cm_mix * rho_mix_mol
    heat_adv_i = advection_flux(T_P, T_W, T_WW, cell_dz, heat_coeff, adv_scheme)

    # Heat diffusion term
    heat_diff_i = heat_diffusion_flux(T_P, T_EX, T_EXX, T_IN, T_INN, cell_dr, cell_r_center, Lambda_er, h_t, diff_scheme)
    
    # Heat source / sink from individual reactions
    source_R = nu_j * r_R * (- enthalpy_R(Cp_i, T_P) )
    source_D = nu_j * r_D * (- enthalpy_D(Cp_i, T_P) )
    source_W = nu_j * r_W * (- enthalpy_W(Cp_i, T_P) )

    # Total heat source/sink   
    heat_source_i = (source_R + source_D + source_W) * bulk_rho_c *  BET_cat_P
    
    # Heat flux for time stepping (dT / dt)
    heat_flux = (heat_adv_i + heat_diff_i + heat_source_i) / (epsilon * rho_mix_mol * Cm_mix + bulk_rho_c * cat_cp)

    return mass_flux, heat_flux






def steady_crank_nicholson(field_Ci_n, field_T_n, cells_rad, C_in_n, T_in_n, T_wall_n,\
                           field_D_er, field_v, field_p,\
                           dz_mgrid, dr_mgrid, cell_V, r_centers_mgrid,
                           rho_cat_bulk, field_BET_cat, d_cat_part, cat_cp, cat_shape,
                           epsilon, N, adv_index, diff_index, pi_limit, nu_i, nu_j,\
                           relax):
    """
    Calculates CRANK NICHOLSON fluxes for concentration Ci and temperature T fields 

    Parameters
    ----------
    field_Ci_n : 3D array
        [J mol-1 K-1] Specie concentration field
    field_T_n : 2D array
        [C] Specie temperature field
    cells_rad : int
        [-] Number of cells in radial direction
    C_in_n : 1D array
        [J mol-1 K-1] Specie concentration field at inlet
    T_in_n : float
        [C] Inlet fluid temperature 
    T_wall_n : 1D array
        [C] Array of Wall temperature
    field_D_er : 1D array
        [m2 s-1] Effective radial diffusion coefficient in packed bed
    field_v : 1D array
        [m s-1] Array of field velocities
    field_p : 1D array
        [Pa] Pressure at cell P
    dz_mgrid : 2D array
        [m] Meshgrid of dz sizes
    dr_mgrid : 2D array
        [m] Meshgrid of  dr sizes
    cell_V : 2D array
        [m3] Cell volumes
    r_centers_mgrid: 2D array   
        [m] Meshgrid of r cell centers
    rho_cat_bulk : float
        [kg m-3] Bulk catalyst density
    field_BET_cat : 2D array
        [m2 kg-1] Surface area of catalyst per mass
    d_cat_part : float
        [m] Catalyst particle diameter
    cat_cp : float
        [J kg-1 K-1] Catalyst specific heat capacity
    cat_shape : string
        [sphere / cylinder] Shape of catalyst particles
    epsilon : float
        [-] Packed bed porosity
    N : float
        [-] Aspect ratio
    adv_index : int
        Choice of advection scheme
    diff_index : int
        Choice of diffusion scheme
    pi_limit : float
        [bar] Partial pressure lower limit for reaction rate 
    nu_i : array
        [-] Effectivness factor for production / consumption of species i
    nu_j : array
        [-] Effectiveness factor for reaction j
    relax : float
        [-] Under-relaxation factor

    Returns
    -------
    C_fluxes_CN : 3D array
        [J mol-1 K-1 s-1] Flux fields of specie concentrations
    T_fluxes_CN : 2D array
        [C s-1] Flux field of temperature/heat

    """
    
     # Get fields of neighbouring cells
    field_C_W, field_C_WW, field_C_E, \
        field_C_EX, field_C_EXX, field_C_IN, field_C_INN, \
        field_T_W, field_T_WW, field_T_E, \
            field_T_EX, field_T_EXX, field_T_IN, field_T_INN = get_neighbour_fields(field_Ci_n, field_T_n, cells_rad, C_in_n, T_in_n, T_wall_n)
    
    # Get first fluxes
    C_fluxes, T_fluxes = steady_Euler_fluxes(field_D_er, field_v, field_p,
        dz_mgrid, dr_mgrid, cell_V, r_centers_mgrid,
        field_Ci_n, field_C_W, field_C_WW, field_C_IN, field_C_INN, field_C_EX, field_C_EXX, 
        field_T_n, field_T_W, field_T_WW, field_T_IN, field_T_INN, field_T_EX, field_T_EXX,
        rho_cat_bulk, field_BET_cat, d_cat_part, cat_cp, cat_shape,
        epsilon, N, adv_index, diff_index, pi_limit, nu_i, nu_j)

    # From fluxes at n, get timestep n+1
    field_Ci_n1 = field_Ci_n + C_fluxes * relax
    field_T_n1 = field_T_n + T_fluxes * relax
    
    T_wall_n1 = gv.Twall_func.steady(T_wall_n, field_Ci_n1[:,0,:], field_T_n1[0,:], field_v)
    
    # Get fields of neighbouring cells for n1
    field_C_W, field_C_WW, field_C_E, \
        field_C_EX, field_C_EXX, field_C_IN, field_C_INN, \
        field_T_W, field_T_WW, field_T_E, \
            field_T_EX, field_T_EXX, field_T_IN, field_T_INN = get_neighbour_fields(field_Ci_n1, field_T_n1, cells_rad, C_in_n, T_in_n, T_wall_n1)

    
    # Get another set of fluxes at n+1
    C_fluxes_n1, T_fluxes_n1 = steady_Euler_fluxes(field_D_er, field_v, field_p,
        dz_mgrid, dr_mgrid, cell_V, r_centers_mgrid,
        field_Ci_n1, field_C_W, field_C_WW, field_C_IN, field_C_INN, field_C_EX, field_C_EXX, 
        field_T_n1, field_T_W, field_T_WW, field_T_IN, field_T_INN, field_T_EX, field_T_EXX,
        rho_cat_bulk, field_BET_cat, d_cat_part, cat_cp, cat_shape,
        epsilon, N, adv_index, diff_index, pi_limit, nu_i, nu_j)
    
    
    C_fluxes_CN = (1/2)*(C_fluxes + C_fluxes_n1)
    T_fluxes_CN = (1/2)*(T_fluxes + T_fluxes_n1)
    
    return C_fluxes_CN, T_fluxes_CN





def steady_Euler_fluxes(D_er, v_n, p_P,
        cell_dz, cell_dr, cell_V, cell_r_center,
        Ci_P, Ci_W, Ci_WW, Ci_IN, Ci_INN, Ci_EX, Ci_EXX,
        T_P, T_W, T_WW, T_IN, T_INN, T_EX, T_EXX,
        bulk_rho_c, BET_cat_P, d_cat_part, cat_cp, cat_shape,
        epsilon, N, adv_scheme, diff_scheme, pi_limit, nu_i, nu_j):
    """
    Calculates the EULER FLUXES OF CONCENTRATION AND TEMPERATURE FIELDS (dCi, dT for dCi/dt, dT/dt)
    This function differs from unsteady Euler fluxed because it doesn't account for thermal innertia of catalyst mass. 

    Parameters
    ----------
    D_er : 1D array
        [m2 s-1] Effective radial diffusion coefficient in packed bed
    v_n : 1D array
        [m s-] Superficial flow velocity
    p_P : 1D array
        [Pa] Pressure at cell P
    cell_dz : 2D array
        [m] Cell axial size
    cell_dr : 2D array
        [m] Cell radial size
    cell_V : 2D array
        [m3] Cell volume
    cell_r_center : 2D array
        [m] Cell center radial position
    Ci_P : 3D array
        [J mol-1 K-1] Specie concentrations at point P
    Ci_W : 3D array
        [J mol-1 K-1] Specie concentrations at neighbouring point W (west)
    Ci_WW : 3D array
        [J mol-1 K-1] Specie concentrations at neighbouring point WW (west west)
    Ci_IN : 3D array
        [J mol-1 K-1] Specie concentrations at neighbouring point IN (internal)
    Ci_INN : 3D array
        [J mol-1 K-1] Specie concentrations at neighbouring point INN (internal internal)
    Ci_EX : 3D array
        [J mol-1 K-1] Specie concentrations at neighbouring point EX (external)
    Ci_EXX : 3D array
        [J mol-1 K-1] Specie concentrations at neighbouring point EXX (external external)
    T_P : 2D array
        [C] Temperature at point P
    T_W : 2D array
        [C] Temperature at neighbouring point W (west)
    T_WW : 2D array
        [C] Temperature at neighbouring point WW (west west)
    T_IN : 2D array
        [C] Temperature at neighbouring point IN (internal)
    T_INN : 2D array
        [C] Temperature at neighbouring point INN (internal internal)
    T_EX : 2D array
        [C] Temperature at neighbouring point EX
    T_EXX : 2D array
        [C] Temperature at neighbouring point EXX
    bulk_rho_c : float
        [kg m-3] Bulk catalyst density
    BET_cat_P : 2D array
        [m2 kg-1] Surface area of fresh catalyst per mass
    d_cat_part : float
        [m] Catalyst particle diameter
    cat_cp : float
        [J kg-1 K-1] Catalyst specific heat capacity        
    cat_shape : string
        [sphere / cylinder] Shape of catalyst particles
    epsilon : float
        [-] Packed bed porosity
    N : float
        [-] Aspect ratio
    adv_scheme : int
        Choice of advection scheme
    diff_scheme : int
        Choice of diffusion scheme
    pi_limit : float
        [bar] Partial pressure lower limit for reaction rate 
    nu_i : array
        [-] Effectivness factor for production / consumption of species i
    nu_j : array
        [-] Effectiveness factor for reaction j
        

    Returns
    -------
    mass_flux : 3D array 
        [-] Fluxes (dCi / dt) in specie concentration fields
    heat_flux : 2D array
        [-] Flux (dT / dt) in temperature (heat) field 

    """
    
    # --- Mass transport 
    # Mass advection term
    mass_adv_i = advection_flux(Ci_P, Ci_W, Ci_WW, cell_dz, v_n, adv_scheme)
    
    # Mass diffusion term
    mass_diff_i = diffusion_flux(Ci_P, Ci_EX, Ci_EXX, Ci_IN, Ci_INN, cell_dr, cell_r_center, D_er, diff_scheme)

    # Mass source term 
    # Mole fractions of species
    X_i = concentration_to_mole_fraction(Ci_P) 
    # partial pressure
    p_i = partial_pressure(p_P, X_i)
    # Reaction rates
    r_R, r_D, r_W = reaction_rates(p_i, T_P, pi_limit)
    # Clip MSR rate because it can get very high from the initial guess in steady state
    r_R = np.clip(r_R, None, 1e-3)
    
    # Formation rates
    r_i = formation_rates(r_R, r_D, r_W, BET_cat_P)
    r_i = np.asarray(r_i) # Convert this to np array
    # Mass source / sink
    mass_source_i = (nu_i * bulk_rho_c * r_i)
    
    mass_flux = (mass_adv_i + mass_diff_i + mass_source_i) /epsilon
    
    # --- Heat transport 
    # Specific heat capacity of every specie
    Cp_i = Cp_species(T_P) 
    # Mixture specific heat capacity
    Cm_mix = Cm_mixture(X_i, Cp_i) # [J mol-1 K-1]
    Cp_mix = Cp_mixture(X_i, Cp_i) # [J kg-1 K-1]
    # Density of a mixture 
    rho_mix_mol = mol_density_mixture(Ci_P) 
    # Effective radial thermal conductivity
    Lambda_er, h_t = radial_thermal_conductivity(v_n, rho_mix_mol, Cm_mix, Cp_mix, d_cat_part, X_i, T_P, epsilon, N, Ci_P[:,0,:], cat_shape)    
    
    # Heat advection term
    heat_coeff = v_n  * Cm_mix * rho_mix_mol
    heat_adv_i = advection_flux(T_P, T_W, T_WW, cell_dz, heat_coeff, adv_scheme)

    # Heat diffusion term
    heat_diff_i = heat_diffusion_flux(T_P, T_EX, T_EXX, T_IN, T_INN, cell_dr, cell_r_center, Lambda_er, h_t, diff_scheme)
    
    # Heat source / sink from individual reactions
    source_R = nu_j * r_R * (- enthalpy_R(Cp_i, T_P) )
    source_D = nu_j * r_D * (- enthalpy_D(Cp_i, T_P) )
    source_W = nu_j * r_W * (- enthalpy_W(Cp_i, T_P) )

    # Total heat source/sink   
    heat_source_i = (source_R + source_D + source_W) * bulk_rho_c * BET_cat_P  
    
    heat_source_i = heat_source_i 
    # Heat flux for time stepping (dT / dt)
    heat_flux = (heat_adv_i + heat_diff_i + heat_source_i) / (Cm_mix * rho_mix_mol) #/ (epsilon * rho_mix_mol * Cm_mix + bulk_rho_c * cat_cp)
    # heat_flux = (heat_adv_i + heat_diff_i + heat_source_i) / (epsilon * rho_mix_mol * Cm_mix + bulk_rho_c * cat_cp)
    # heat_flux = (heat_adv_i + heat_diff_i + heat_source_i)
    # print(np.max(r_R))
    # print(np.min(r_D))
    # print(np.min(r_W))
    # print(heat_flux)


    return mass_flux, heat_flux



 # relax_max = 1e-5 # Max relax factor 
 # residuals_high_limit = 1e3 #When does relax recalculation kick in
 # residuals_low_limit = 1e-2 # What is the threshold for max relax 
 
def new_relax_factor(relax_min, relax_max, residuals, residuals_high_limit, residuals_low_limit):
    """
    Recalculate underrelaxation factor based on residuals using logarithmic interpolation

    Parameters
    ----------
    relax_min : float
        [-] Smallest underrelaxation factor
    relax_max : float
        [-] Largest underrelaxation factor
    residuals : float
        [-] Current residuals
    residuals_high_limit : float
        [-] Residuals high threshold where smallest relax factor is
    residuals_low_limit : float
        [-] Residuals low threshold where highest relax factor is

    Returns
    -------
    new_relax : float
        [-] New underrelaxation factor

    """
    # Make lists
    xx = [residuals_low_limit, residuals_high_limit]
    yy = [relax_max, relax_min]
   
    # calculate natural logarithms of all elements 
    logz = np.log10([residuals])
    logx = np.log10(xx)
    logy = np.log10(yy)
    
    # Interpolate logarithms and revert to original number
    new_relax = np.power(10.0, np.interp(logz, logx, logy))[0]
    
    return new_relax







# -------------------------------------------------
# ------ WALL TEMPERATURE and REACTOR HEATING FUNCTIONS
# -------------------------------------------------


def T_wall_second_derivative(T_P, dz):
    """
    Calculates second derivatives of wall temperature array with 4th order central bounded difference

    Parameters
    ----------
    T_P : 1D array
        [K] Wall temperatures
    dz : float
        [m] Cell spacing array

    Returns
    -------
    d2_T_wall : 1D array
        [K m-1] Second derivative of wall temperature  

    """
    
    # Make arrays for EE, E, W, and WW cells 
    T_E = np.roll(T_P, -1)
    T_E[0,-1] = T_E[0,-2] # Eastmost boundary
    T_EE = np.roll(T_E,-1)
    T_EE[0,-1] = T_EE[0,-2] # Eastmost boundary
    
    T_W = np.roll(T_P, 1)
    T_W[0,0] = T_W[0,1] # Westmost boundary
    T_WW = np.roll(T_W, 1)
    T_WW[0,0] = T_WW[0,1] # Westmost boundary
    
    # Second derivative evaluation 2nd order
    # d2_T_wall = (T_E - 2*T_P + T_W) / (dz**2)
    
    # Second derivative evaluation 4th order
    d2_T_wall = (-T_EE + 16*T_E - 30*T_P + 16*T_W - T_WW) / (12*dz**2)
    
    # # Special treatment for near wall cells
    # # Near inlet cells
    # # d2_T_wall[0,0] = (2*T_P[0,0] - 5*T_P[0,1] + 4*T_P[0,2] - T_P[0,3]) / (dz[0]**2) # 2o right sided FD 
    # d2_T_wall[0,0] = (T_P[0,0] - 2*T_P[0,1] + T_P[0,2]) / (dz[0]**2) # 1o right sided FD 
    # d2_T_wall[0,1] = (T_P[0,0] - 2*T_P[0,1] + T_P[0,2]) / (dz[1]**2) # 2o central difference for second cell
    # # Near outlet cells 
    # # d2_T_wall[0,-1] = (2*T_P[0,-1] - 5*T_P[0,-2] + 4*T_P[0,-3] - T_P[0,-4]) / (dz[-1]**2) # 2o left sided FD 
    # d2_T_wall[0,-1] = (T_P[0,-1] - 2*T_P[0,-2] + T_P[0,-3]) / (dz[-1]**2) # 1o left sided FD 
    # d2_T_wall[0,-2] = (T_P[0,-3] - 2*T_P[0,-2] + T_P[0,-1]) / (dz[-2]**2) # 2o central difference for second to last cell
    
    
    return d2_T_wall




def T_wall_first_derivative(T_P, dz):
    """
    Calculates first derivatives of wall temperature array with 4th order central bounded difference

    Parameters
    ----------
    T_P : 1D array
        [K] Wall temperatures
    dz : 1D array
        [m] Cell spacing array

    Returns
    -------
    d_T_wall : 1D array
        [K m-1] Second derivative of wall temperature  

    """
    
    # Make arrays for EE, E, W, and WW cells 
    T_E = np.roll(T_P, -1)
    T_E[0,-1] = T_E[0,-2] # Eastmost boundary
    T_EE = np.roll(T_E,-1)
    T_EE[0,-1] = T_EE[0,-2] # Eastmost boundary
    
    T_W = np.roll(T_P, 1)
    T_W[0,0] = T_W[0,1] # Westmost boundary
    T_WW = np.roll(T_W, 1)
    T_WW[0,0] = T_WW[0,1] # Westmost boundary
    
    
    # First derivative evaluation 2o
    # d_T_wall = (T_E - T_W) / (2*dz)
    
    # First derivative evaluation 4o
    d_T_wall = (-T_EE + 8*T_E - 8*T_W + T_WW) / (12*dz)
    
    # Special treatment for near wall cells
    # Near inlet cells
    d_T_wall[0,0] = (-3*T_P[0,0] + 4*T_P[0,1] - T_P[0,2]) / (2*dz[0]) # 2o right sided FD 
    # d_T_wall[0,0] = (-T_P[0,0] + T_P[0,1]) / (dz[0]) # 1o right sided FD 
    
    d_T_wall[0,1] = (- T_P[0,0] + T_P[0,2]) / (2*dz[1]) # 2o central difference for second cell
    
    # Near outlet cells 
    d_T_wall[0,-1] = (3*T_P[0,-1] - 4*T_P[0,-2] + T_P[0,-3]) / (2*dz[-1]) # 2o left sided FD 
    # d_T_wall[0,-1] = (T_P[0,-1] - T_P[0,-2]) / (dz[-1]) # 1o left sided FD 
    
    d_T_wall[0,-2] = (- T_P[0,-3] + T_P[0,-1]) / (2*dz[-2]) # 2o central difference for second to last cell
    
    return d_T_wall



def read_and_set_T_wall_BC(input_json_fname, dyn_bc, z_cell_centers, l_tube):
    """
    Read boundary conditions and define time dependant functions of wall temperature boundary condition

    Parameters
    ----------
    input_json_fname : str
        Input .json file name
    dyn_bc : string
        (yes / no) Use dynamic boundary conditions or not 
    z_cell_centers : array
        [m] Distances of axial 
    l_tube : float
        [m] Reactor tube length

    Raises
    ------
    ValueError
        

    Returns
    -------
    heating_choice : string
        Chosen wall heating 
    T_wall_func_steady : function
        Function for steady wall temperature profile
    T_wall_func_dynamic : function
        Function for dynamic wall temperature profile

    """
    
    # Define dictionary of keys for heating options
    heating_dict = {'temperature-profile':'temperature profile parameters',
                   'flue-gas':'flue gas parameters',
                   'joule':'joule heating parameters'}
    
    
    # Retreive the heating choice
    heating_choice = json.load(open(input_json_fname))['reactor heating']['heating type (temperature-profile / flue-gas / joule)']
    # Ready heating parameters for chosen 
    steady_heating_params = json.load(open(input_json_fname))['reactor heating'][heating_dict[heating_choice.lower()]]
    dynamic_heating_params = json.load(open(input_json_fname))['dynamic boundary conditions']['dynamic heating parameters'][heating_dict[heating_choice.lower()]]

    # Define relative cell centers (from 0 to 1) that will be used to interpolate 
    rel_zcell_centers = z_cell_centers / l_tube
    
    # See whether we use dynamic boundary conditions for t wall
    dyn_BC = json.load(open(input_json_fname))['dynamic boundary conditions']
    
    # See whether we're using dynamic boundary conditions
    condition = dyn_BC['use dynamic wall heating (yes / no)'].lower()
    if condition not in ['yes', 'no']: 
        raise NameError('Dynamic boundary conditions: dynamic wall heating choice not recognized')
    
    if heating_choice == 'temperature-profile':

        # for steady we just need to return the pre defined profile 
        wall_T_rel_profile = np.asarray(steady_heating_params['reactor wall temperature profile [C]'])
        wall_T_rel_positions = np.asarray(steady_heating_params['wall temperature relative axial positions [-]'])

        ax_cells = len(z_cell_centers) # Number of axial cells 

        # Calculate wall temperature distribution 
        rel_dz_half = (1 / ax_cells) / 2 # Get relative axial cell size (uniform mesh)
        # Mesh in relative coordinates (from 0 to 1)
        rel_mesh = np.linspace(rel_dz_half,1.-rel_dz_half,int(ax_cells)) 
        # Get initial wall T profile
        wall_T_profile = [np.interp(x, wall_T_rel_positions, wall_T_rel_profile) for x in rel_mesh]

        # Make a function that always returns the same wall T profile        
        def T_wall_func_steady(self, *args, **kwargs):
            return wall_T_profile
        
        
        # -- Dynamic boundary condition function 
        if dyn_bc == 'no' or condition == 'no': # If we're not using dynamic boundary conditions entirely or if we're not using them for wall temp
            # If dynamic 
            def T_wall_func_dynamic(self, *args, **kwargs):
                return wall_T_profile
                
        else: # else if both conditions are true
            # Get axial relative profile
            Tw_ax_profile = np.asarray(dynamic_heating_params['reactor wall temperature profile z relative positions'])
            # Tuple containing times(floats) and Twall profiles at times (tuples)
            Twall_tuple = dynamic_heating_params['reactor wall temperature profile in time (t[s], T_profile[C])']
            # Extract np arrays from this tuple
            Tw = np.asarray([row[1] for row in Twall_tuple])
            time_Tw = np.asarray([row[0] for row in Twall_tuple])
            
            
            if len(Tw) == 0 or len(time_Tw) == 0:
                raise ValueError('In "dynamic boundary conditions", there are empty lists in wall temperature profiles')
            
            for row in Tw: 
                if len(row) != len(Tw[0]): # Every given temp. profile should be of same length
                    raise ValueError('In "dynamic boundary conditions", not all wall profiles are of same size!')
            if len(Tw[0]) != len(Tw_ax_profile): # Amount of relative axial points should match amount of temp. points given in every T_wall_profile
                raise ValueError('In "dynamic boundary conditions", length of relative axial points and T_profile do not match!')
            if not (sorted(Tw_ax_profile) == Tw_ax_profile).all(): # Check if relative profile is given in ascending order
                raise ValueError('In "dynamic boundary conditions", wall temperature axial relative points should be given in ascending order')
            if not ( sorted(time_Tw) == time_Tw).all(): # Time should be given in ascending order
                raise ValueError('In "dynamic boundary conditions", time for wall temperature should be given in ascending order')

            if len(Tw) == 1: # If we defined only one profile point
                # We don't have to do temporal interpolation here, just spatial once 
                Tw_profile = [] # Empty array 
                
                for point in rel_zcell_centers:
                    # Interpolate user given relative axial positions to mesh axial cell distribution
                    Tw_profile.append( np.interp(point, Tw_ax_profile, Tw[0]) ) 
                # Convert to np.array    
                Tw_profile = np.asarray(Tw_profile)
                # Define a function that always returns this profile 
                def Twall_func_dynamic(self, t, *args, **kwargs):
                    return Tw_profile
            
            else: # If array has more than 1 temporal point, we have to 
                # Reshape Tw - "rotate" matrix 90 degrees 
                # In reshaped matrix each row is evolution of temperature in time for a single point
                Tw = np.ravel(Tw, order='F').reshape(len(Tw[0]), len(Tw)) 
                
                def T_wall_func_dynamic(self, t, *args, **kwargs):
                    """
                    Interpolates wall temperature profile in time
        
                    Parameters
                    ----------
                    t : float
                        [s] time
        
                    Returns
                    -------
                    Tw_profile : array
                        Wall temperature profile 
        
                    """
                    
                    # Interpolate temperature profile for time t and given relative z coordinates
                    interp_Tw = [] # Empty list for time interpolated wall temperature profile
                    for point in Tw:
                        # Interpolate in time for each relative axial position
                        interp_Tw.append( np.interp(t, time_Tw, point) )
                     # interp_Tw profile does not match axial z cell distribution
                    
                    Tw_profile = [] # Empty list for Temperature profile that matches axial z cell distribution
                    for point in rel_zcell_centers:
                        # Interpolate user given relative axial positions to mesh axial cell distribution
                        Tw_profile.append( np.interp(point, Tw_ax_profile, interp_Tw) ) 
                    
                    # Convert to numpy array
                    Tw_profile = np.asarray(Tw_profile)
                    
                    return Tw_profile
        
        
        
    elif heating_choice == 'joule':
        # For joule we need to iterate depending on h_wall T_wall, and T_near_wall
        # In dynamic there is also a t variable which changes I
        
        # Get material properties
        material_props = json.load(open(input_json_fname))['reactor parameters']
        d_tube_in_jh = material_props['single tube inner diameter [m]']
        s_tube_jh = material_props['single tube wall thickness [m]']
        rho_tube_jh = material_props['material density [kg m-3]']
        k_tube_jh = material_props['material thermal conductivity [W m-1 K-1]']
        cp_tube_jh = material_props['material specific heat capacity [J kg-1 K-1]']
        
        # Joule heating specifics
        I_tube_jh = steady_heating_params['current through single tube [A]']
        rho_e_tube_jh = steady_heating_params['tube material electrical resistivity [ohm m]']
        
        # The current formulas are derived for total wall thickness across diameter so below we use this s2
        s2_tube_jh = s_tube_jh*2
        
        # underrelaxation factor specifically for reactor tube wall
        wall_relax = json.load(open(input_json_fname))['simulation parameters']['steady simulation parameters']['wall underrelaxation factor']
        
        # Axial cell spacing array
        n_cells = len(z_cell_centers) # Number of cells
        dz = (l_tube / n_cells) 
        
        # Calculate tube wall cell volume
        # r_tube_in = d_tube_in_jh/2 # Inner radius
        # r_tube_out = r_tube_in + s_tube_jh # Outer radius
        
        # Cross section area of tube
        # A_cs_tube = np.pi * (r_tube_out**2 - r_tube_in**2)
        # dV_tube_cell = A_cs_tube * dz
        
        
        def T_wall_func_steady(self, T_wall, C_near_wall, T_near_wall, u_s, *args, **kwargs): 
            """
            Calculates wall temperature from Joule heating

            Parameters
            ----------
            relax : float
                [-] Underrelaxation factor
            T_wall : 1D array
                [C] Current wall temperature profile
            C_near_wall : 2D array
                [mol m-3] Current specie sconcentration array of internal cells adjacent to the wall
            T_near_wall : 1D array
                [C] Current temperature array of internal cells adjacent to the wall
            u_s : 1D array
                [m s-1] Superficial velocity

            Returns
            -------
            wall_T_profile : 1D array
                [C] Newly calculated wall temperature profile

            """
            C_near_wall = np.rot90(np.reshape(C_near_wall, C_near_wall.shape + (1,)), axes=(1,2))
            T_wall = np.rot90(np.reshape(T_wall, T_wall.shape+(1,)))
            T_near_wall = np.rot90(np.reshape(T_near_wall, T_near_wall.shape+(1,)))

            # First get film transfer coefficient
            ht = heat_transfer_coeff_tube(u_s, self.d_p, T_near_wall, C_near_wall, self.epsilon)
            
            d2_T_wall = T_wall_second_derivative(T_wall, dz) # Second derivative along z [K m-2]
            Q_ax = d2_T_wall * k_tube_jh
            
            # Heat generated by Joule effect [kg s-3 m-1]
            Q_gen = (16 * rho_e_tube_jh * I_tube_jh**2) / ( np.pi**2 * s2_tube_jh**2 * (2*d_tube_in_jh + s2_tube_jh)**2)
            
            # Heat radially transferred to the reactor [kg s-3 m-1]
            # Q rad is the formula for radial heat transfer but we modify it in steady state to extract T_wall
            # Q_rad = ( (4 * d_tube_in_jh * ht) / (s2_tube_jh*(2*d_tube_in_jh + s2_tube_jh)) ) * (T_wall - T_near_wall)
            Q_rad_factor = ( (4 * d_tube_in_jh * ht) / (s2_tube_jh*(2*d_tube_in_jh + s2_tube_jh)) ) #* (T_wall - T_near_wall)
            
            # Total delta T 
            dT = (Q_ax + Q_gen)/Q_rad_factor  #/ (rho_tube_jh*cp_tube_jh)
            # Clip to avoid massive fluxes while the simulation stabilizes
            # dT = np.clip(dT, -30, 30) 
            
            # Calculate new wall T profile
            wall_T_profile = (T_near_wall + dT).flatten()
            
            new_T_wall = (wall_T_profile*wall_relax + T_wall*(1-wall_relax)).flatten()
            return new_T_wall
        
        
        
        # -- Dynamic boundary condition function 
        if dyn_bc == 'no' or condition == 'no': # If at least one is false, we have a steady current
            # Make a lambda function for I that always gives the same value
            I_in_func = lambda t : I_tube_jh

        else: # Otherwise make an interpolation function
            # - Read values from dictionary
            # Matrix containing times and currents
            I_matrix = np.asarray(dynamic_heating_params['current through single tube in time (t[s], I[A])'])
            # Extract respective arrays from the matrix
            I_in = I_matrix[:,1]
            time_I = I_matrix[:,0]
            
            # Do some checks
            if len(I_in) == 0 or len(time_I) == 0: # If nothing in list(s) 
                raise ValueError('In "dynamic boundary conditions", there are empty lists in joule heating profiles')
            if len(time_I) != len(I_in): # Amount of inlet feeds points should match number of time points
                raise ValueError('In "dynamic boundary conditions", length of I_in and T_time does not match!')
            if not ( sorted(time_I) == time_I).all(): # Time should be given in ascending order
                raise ValueError('In "dynamic boundary conditions", time_I should be given in ascending order')
            
            if len(I_in) == 1: # If one element in list 
                I_in_func = lambda t : I_in[0] # Dont bother with interpolation, just return that value
                
            else: # If multiple values in list
                    # - First check some things 
                    if len(time_I) != len(I_in): # Amount of inlet feeds points should match number of time points
                        raise ValueError('In "dynamic boundary conditions", length of t_WF_in and WF_in does not match!')
                        
                    if not ( sorted(time_I) == time_I).all(): # Time should be given in ascending order
                        raise ValueError('In "dynamic boundary conditions", t_WF_in should be given in ascending order')
                        
                    def I_in_func(t): 
                        """
                        Interpolates current through reactor tube in time
            
                        Parameters
                        ----------
                        t : float
                            [s] time
            
                        Returns
                        -------
                        I_in : float
                            [A] Electrical current through one tube 
                        """
                        
                        I_tube = np.interp(t, time_I, I_in)
                        
                        return I_tube
            
            
        def T_wall_func_dynamic(self, t, dt, T_wall, C_near_wall, T_near_wall, u_s, *args, **kwargs):
            """
            Calculates wall temperature from Joule heating

            Parameters
            ----------
            t : float
                [s] Time
            dt : float
                [s] Timestep size
            T_wall : 1D array
                [C] Current wall temperature profile
            C_near_wall : 2D array
                [mol m-3] Current specie sconcentration array of internal cells adjacent to the wall
            T_near_wall : 1D array
                [C] Current temperature array of internal cells adjacent to the wall
            u_s : 1D array
                [m s-1] Superficial velocity

            Returns
            -------
            wall_T_profile : 1D array
                [C] Newly calculated wall temperature profile

            """
            C_near_wall = np.rot90(np.reshape(C_near_wall, C_near_wall.shape + (1,)), axes=(1,2))
            T_wall = np.rot90(np.reshape(T_wall, T_wall.shape+(1,)))
            T_near_wall = np.rot90(np.reshape(T_near_wall, T_near_wall.shape+(1,)))
            
            # # Get current at time t
            I_tube_t = I_in_func(t)
            
            # # First get film transfer coefficient
            ht = heat_transfer_coeff_tube(u_s, self.d_p, T_near_wall, C_near_wall, self.epsilon)
            
            # Get heat transfer due to axial diffusion along the reactor 
            d2_T_wall = T_wall_second_derivative(T_wall, dz) # Second derivative along z [K m-2]
            Q_ax =  d2_T_wall * k_tube_jh # 
            
            # Heat radially transferred to the reactor [kg s-3 m-1]
            Q_rad = ( (4 * d_tube_in_jh * ht) / (s2_tube_jh*(2*d_tube_in_jh + s2_tube_jh)) ) * (T_wall - T_near_wall)
            
            # Heat generated by Joule effect [kg s-3 m-1]
            Q_gen = (16 * rho_e_tube_jh * I_tube_t**2) / ( np.pi**2 * s2_tube_jh**2 * (2*d_tube_in_jh + s2_tube_jh)**2)
            
            # Sum up heat fluxes and divide by tube density and cp to get dT_wall / dt
            dT = (Q_ax - Q_rad + Q_gen) / (rho_tube_jh * cp_tube_jh)
            
            # Calculate new wall T profile
            wall_T_profile = (T_wall + dT*dt).flatten()
            
            return wall_T_profile
        
        
    elif heating_choice == 'flue-gas': 
         pass # !!!
         
    class Twall(): # Create an empty class that will contain methods for calculating Twall
        pass 
    
    # Add defined steady and dynamic functions as methods to Twall class
    Twall.steady = T_wall_func_steady
    Twall.dynamic = T_wall_func_dynamic
    
    # Define a class instance within global variables module
    gv.Twall_func = Twall()
    
    
    # Define additional functions in class for output file writing
    if heating_choice == 'joule':
        def I_func(self, t):
            I = I_in_func(t)
            return I
        Twall.I_func = I_func
    else: 
        Twall.I_func = lambda self, t : 'n/a'
    
    
    
    return heating_choice






























