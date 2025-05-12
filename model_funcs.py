# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 19:21:52 2023

@author: bgrenko
"""
import numpy as np
import math
from scipy.interpolate import RectBivariateSpline 
from scipy.interpolate import interp1d

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
    def __init__(self, name, Vc, Tc, w, mu, OH_groups, mol_mass, La, Lb, Lc, Ld, Le, Cma, Cmb, Cmc, Cmd, Cme):
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
        
        # Values for caluclating specific heat capacity (Cp_i) using the Shomate equation
        self.Cma = Cma
        self.Cmb = Cmb
        self.Cmc = Cmc
        self.Cmd = Cmd
        self.Cme = Cme
        
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
    def mu_i(self, T): # Temperature input in Kelvin
        T_star = 1.2593 * (T / self.Tc)
        Omega_v = self.A*(T_star**(-self.B)) + self.C*np.exp(-self.D*T_star) + self.E*np.exp(-self.F*T_star)
        self.result = 40.758*( self.Fc*((self.mol_mass * T)**(1/2)) / (self.Vc**(2/3) * Omega_v)) 
        return self.result # Result in microPoise
    
    # Shomate equation for gas phase molar heat capacity
    def Cm_i(self, T):
        T_K = T+273.15 # Temperature in kelvin
        t = T_K/1000
        self.Cm_result = self.Cma + self.Cmb*t +self.Cmc*t**2 + self.Cmd*t**3 + self.Cme*(1/t)**2
        return self.Cm_result
        
# Define specie instances
CH3OH = Specie('CH3OH', 118.0, 512.64, 0.557970015, 1.7, 1, 32.042, \
               # Lambda_i values
           8.0364e-5, 0.013, 1.4250e-4, -2.8336e-8, 1.2646e-9, \
               # Cm_i values
               13.93945, 111.30774, -41.59074, 5.482564, 0.052037)
    
H2O = Specie('H2O', 55.95, 647.14, 0.331157716, 1.84, 0, 18.01528, \
             # Lambda_i values
                 0.4365, 0.0529, 1.0053e-5, 4.8426e-8, 2.3506e-5, \
                 # Cm_i values
                 30.092, 6.832514, 6.7934356, -2.534480, 0.0821398)
    
H2 = Specie('H2', 64.20, 32.98, -0.22013078, 0, 0, 2.01568, \
            # Lambda_i values
                -11.9, 0.8870, -9.2345e-4, 5.6111e-7, -0.0026, \
                    # Cm_i values
                33.066178, -11.363417, 11.432816, -2.772874, -0.158558)  

CO2 = Specie('CO2', 94.07, 304.12, 0.228, 0, 0, 44.01, \
             # Lambda_i values
                 0.6395, 0.019, 1.5214e-4, -1.1666e-7, 7.8815e-5, \
                     # Cm_i values
                24.99735, 55.18696, -33.69137, 7.948387, -0.136638)  
    
CO = Specie('CO', 93.10, 132.85, 0.053424305, 0.122, 0, 28.01,\
            # Lambda_i values
                0.0185, 0.0918, -3.1038e-5, 8.1127e-9, 9.7460e-7, \
                    # Cm_i values
                25.56759, 6.096130, 4.054656, -2.671301, 0.131021) 
    
N2 = Specie('N2', 89.8, 126.2, 0.039, 0, 0, 28.013, \
            # Lambda_i values
                -0.6586197, 0.1079207, -7.37406e-5, 3.355197e-8, -0.00086439, \
                # Cm_i values
                28.98641, 1.853978, -9.647459, 16.63537, 0.000117)  
    
    
    
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
    # k1 = -u_s / (rho_fluid * d_particle)
    # k2 = (1-epsilon)/epsilon**3
    # k3 = ( (150 * (1-epsilon) * mu_mix) / d_particle ) + 1.75*u_s 
    
    
    k1 = - u_s**2 * rho_fluid / d_particle
    k2 = (1-epsilon)/epsilon**3
    k3 = ( (150 * (1-epsilon) * mu_mix) / (d_particle*rho_fluid* u_s) ) + 1.75
    
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
    
    for i in range(6):
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
    for i in range(6): # i loop - numerator
        j_list = list(range(6))
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


def Cm_species(T_C):
    """
    Calculate the SPECIFIC MOLAR HEAT CAPACITY for INDIVIDUAL SPECIES at given temperature

    Parameters
    ----------
    T_C :                 
        [C] Temperature

    Returns
    -------
    C_p : array         
        [J mol-1 K-1]
    """
    
    # make empty array
    C_m = np.zeros((6, np.shape(T_C)[0] ,np.shape(T_C)[1]))
    
    for i in range(6):
        C_m[i] = Specie._reg[i].Cm_i(T_C)
    
    return C_m
    
    

    
def Cm_mixture(X_i, C_p):
    """
    Calculate MOLAR HEAT CAPACITY of a mixture

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


def Cp_mixture(X_i, Cm_mix):
    """
    Calculate SPECIFIC HEAT CAPACITY of a mixture

    Parameters
    ----------
    X_i : array         
        [-] Molar fractions of components
    Cm_mix : array         
        [J mol-1 K-1] Specific heat capacities of mixture

    Returns
    -------
    cp_mix_kg : float      
        [J kg-1 K-1] Specific heat capacity of a mixture
    """
    
    # Make a deep copy
    m_mix = copy.deepcopy(X_i) # molecular mass of mixture
    
    
    for i in range(6):
        m_mix[i] = X_i[i] * (Specie._reg[i].mol_mass /1000) 
        # [kg mol-1]
    
    cp_mix_kg = Cm_mix / sum(m_mix)
    
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
    m_i = np.zeros(6)
    for i in range(6):
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
    
    # Universal gas constant [J K-1 mol-1]
    R = 8.31446261815324
    
    # Get number of moles in a volume
    n_i = C_i * V
    
    # --- Component individual masses
    # Create an empty array of m_i values
    m_i = np.zeros(6)
    for i in range(6):
        m_i[i] = n_i[i] * Specie._reg[i].mol_mass
   
    
    n_mix = sum(n_i) # [mol] Total number of moles in a volume
    m_mix = sum(m_i) /1000 # [kg] Total gas weight in a given volume + convert to kilograms
    
    # Average molecular mass in the mixture
    mol_mass_mix = m_mix / n_mix # [kg mol-1]
    
    # According to ideal gas law,
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
    for i in range(6):
        Ci_copy[i] = Ci_copy[i] * Specie._reg[i].mol_mass # [g m-3]
        
    rho_mix = np.sum(Ci_copy, axis=0)/1000 # total density within the computational cell + convert to [kg m-3]
    
    # Get superficial mass velocity
    G = rho_mix * u_s 
    
    return G


def X_to_concentration(X_i, p, T_C, Q, A_inlet):
    """
    Calculate MOLAR CONCENTRATIONS from given steam to carbon ratio

    Parameters
    ----------
    X_i : 1D array
        [-] Molar fractions 
    p : float
        [Pa] Pressure
    T_C : float         
        [C] Temperature
    Q : float         
        [m3 s-1] Volumetric flow rate

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
    
    # Multiply mixture concentration with molar fractions to get an array of concentrations for each specie
    C_i = C_mix * X_i # [mol m-3]
    
    return C_i





def velocity_and_flowrate(n_i_in, p_tot, T_C, r_tube):
    """
    Calculate SUPERFICIAL FLOW VELOCITY and VOLUMETRIC FLOW RATE

    Parameters
    ----------
    n_in : 1D array
        [mol s-1] Inlet molar flows 
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
    
    # Get total molar flow rate
    ndot_mix = sum(n_i_in) # [mol s-1]
    
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
    Calculates equivalent volume and surface diameters of catalyst particle

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
    d_cat_part_s : float
        [m] Catalyst particle surface equivalent diameter
    d_cat_part_v : float
        [m] Catalyst particle volume equivalent diameter

    """
    
    
    if cat_shape == 'sphere':
        d_cat_part_v = cat_dimensions # diameter is given for sphere
        d_cat_part_s = cat_dimensions # 
        V_cat_part = (4 / 3) * np.pi * (cat_dimensions / 2)**3 # volume is calculated
        
    elif cat_shape == 'cylinder':
        # Base diameter and height are given for sphere
        cat_d = cat_dimensions[0] # catalyst base diameter
        cat_h = cat_dimensions[1] # Cylinder height
        V_cat_part = np.pi * (cat_dimensions[0]/2)**2 * cat_dimensions[1] # calculate volume first
        d_cat_part_v = cat_d*(3/2 * cat_h/cat_d)**(1/3) # calculate EQUIVALENT SPHERE DIAMETER FOR VOLUME
        
        radius = cat_d/2
        surface = 2*np.pi*radius**2 + 2*radius*np.pi*cat_h
        d_cat_part_s = (surface/np.pi)**(1/2)
        
    return V_cat_part, d_cat_part_s, d_cat_part_v



def porosity(d_ti, dimensions, shape='sphere'):
    """
    Calculate POROSITY OF A RANDOMLY PACKED BED
    Expressions taken from https://doi.org/10.1080/02726350590922242

    Parameters
    ----------
    d_ti :              
        [m] Reactor tube inner diameter  
    dimensions : float or list            
        dimensions for catalyst particle - if sphere then its d, if cylinder then its [h,d]
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
        d_p = dimensions[1]
        ratio = d_ti/d_p
        
        if not 1.7 <= ratio <= 26.3:
            print('WARNING: ratio of diameters of reactor tube to catalyst particle not ideal.')
            print('Recomended: 1.7 <= d_ti/d_p <= 26.3 for CYLINDRICAL catalyst pellets')
            print('Calculated:', ratio)
            print('Porosity evaluation may be incorrect')
            print('\n')
        
        epsilon = 0.373 + 1.703/( (d_ti / d_p) + 0.611)**2
    
    elif shape == 'sphere':
        d_p = dimensions
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


def radial_thermal_conductivity(u_s, rho_mix_mol, Cm_mix, d_p, X_i, T_C, epsilon, N, shape='sphere'):
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
    Lambda_p = 0.21 + T*1.5e-4 # [W m-1 K-1]
    
    # Lambda_i - thermal conductivites of component i
    Lambda_i = np.zeros((6, np.shape(T)[0] ,np.shape(T)[1]))
    
    for i in range(6):
        Lambda_i[i] = Specie._reg[i].La + Specie._reg[i].Lb*T + Specie._reg[i].Lc*T**2 + Specie._reg[i].Ld*T**3 +Specie._reg[i].Le*(1/T)**2
    
    # Lambda_i is in unit [mW m-1 K-1], convert to [W m-1 K-1]
    Lambda_i *= 1e-3
    # NOTE: Array Lambda_i will be the numerators in calculation of Lambda_f

    # Calculate denominators - use np.ones to avoid adding one to each denominator sum for when i = j
    Lambda_f_denoms = np.zeros((6, np.shape(T)[0] ,np.shape(T)[1]))
    
    # Calculate arrays of denominators
    for i in range(6): # i loop - numerator
        for j in range(6): # j loop - denominator
            phi_ij = (Specie._reg[j].mol_mass / Specie._reg[i].mol_mass)**(1/2) 
            # sum of phi 
            Lambda_f_denoms[i] += phi_ij * X_i[j] 

    # divide and sum up
    Lambda_f = sum(Lambda_i * X_i /Lambda_f_denoms) # [W m-1 K-1] # Thermal conductivity of gas mixture
    
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
    
    k1 = 1 - np.sqrt(1 - epsilon)
    k2 = (2 * np.sqrt(1-epsilon)) / ( 1-B*kappa**-1 )
    k3 = (B * (1 - kappa**(-1))) / (1-B*kappa**-1)**2 
    k4 = np.log( (kappa/B) )
    k5 = -((B - 1) / (1-B*kappa**-1) ) - ((B + 1)/2)
    
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
    
    Lambda_er = np.clip(Lambda_er, 0, 0.8) 
    # With cylindrical pellets, the term 1-B/kappa can locally be close to zero during steady state initialization
    # The clip keeps overall coefficient in physical boundaries
    
    return Lambda_er



def heat_transfer_coeff_tube(u_s, d_p, T_C, C_i, epsilon):
    """
    Calculate heat transfer coefficient on the tube side of the reactor

    Parameters
    ----------
    u_s : 1D array              
        [m s-1] Superficial flow velocity
    d_p : float          
        [m] (effective) diameter of catalyst particle
    T_C : 2D array            
        [C] Temperature field
    C_i : 3D array            
        [mol m-3] Concentration field
    epsilon : float      
        [-] Packed bed porosity
        
    Returns
    -------
    h_t : 1D array
        [W m-2 K-1] Heat transfer coefficient of the tube-side film
    
    """
    # Get axial average values of concentration and temperatures 
    T_C_ravg = np.sum(T_C * gv.mesh_volumes, axis=0, keepdims=True) / sum(gv.mesh_volumes)
    Ci_ravg = np.sum(C_i * gv.mesh_volumes, axis=1, keepdims=True) / sum(gv.mesh_volumes)
    
    # Convert temperature to Kelvin 
    T = T_C_ravg + 273.15
    
    X_i_ravg = concentration_to_mole_fraction(Ci_ravg) 
    # Get Cp of individual species at temperature T
    Cm_i_ravg = Cm_species(T_C_ravg)
    # Get mixture Cm
    Cm_ravg_mix = Cm_mixture(X_i_ravg, Cm_i_ravg)
    # Get Cp of mixture
    Cp_ravg_mix = Cp_mixture(X_i_ravg, Cm_ravg_mix)
    
    
    # Lambda_i - thermal conductivites of component i
    Lambda_i = np.zeros((6, np.shape(T)[0] ,np.shape(T)[1]))
    
    for i in range(6):
        Lambda_i[i] = Specie._reg[i].La + Specie._reg[i].Lb*T + Specie._reg[i].Lc*T**2 + Specie._reg[i].Ld*T**3 + Specie._reg[i].Le*(1/T)**2
    
    # Lambda_i is in unit [mW m-1 K-1], convert to [W m-1 K-1]
    Lambda_i *= 1e-3
    
    # NOTE: Array Lambda_i will be the numerators in calculation of Lambda_f
    
    # Calculate denominators - use np.ones to avoid adding one to each denominator sum for when i = j
    Lambda_f_denoms = np.zeros((6, np.shape(T)[0] ,np.shape(T)[1]))
    
    # Calculate arrays of denominators
    for i in range(6): # i loop - numerator
        for j in range(6): # j loop - denominator
            phi_ij = (Specie._reg[j].mol_mass / Specie._reg[i].mol_mass)**(1/2) 
            # sum of phi 
            Lambda_f_denoms[i] += phi_ij * X_i_ravg[j] 
            
    # divide and sum up - zeros in numerator cancel out ones in denominator
    Lambda_f = sum(Lambda_i * X_i_ravg /Lambda_f_denoms) # [W m-1 K-1]
    # Lambda_f = sum(Lambda_i * X_i_ravg)
    
    # Get gas mixture viscosity 
    mu_mix_ravg = gas_mixture_viscosity(X_i_ravg, T_C_ravg)
    
    # Get mass velocity
    G = superficial_mass_velocity(Ci_ravg, u_s)
    
    # Calculate local Reynolds number in a packed bed
    Re = (d_p * G) / (mu_mix_ravg) /(1-epsilon)
    # Re = (d_p * G) / (mu_mix_ravg) 
    
    # Prandtl number
    Pr = (Cp_ravg_mix * mu_mix_ravg) / Lambda_f
    
    
    
    # Heat transfer coefficient from Mears https://doi.org/10.1016/0021-9517(71)90073-X
    h_t = (0.4 * Re**(1/2) + 0.2*Re**(2/3)) * Pr**0.4 * (1 - epsilon)/epsilon * Lambda_f/d_p
    # [W m-2 K-1]
    
    # h_t = (0.4 * Re**(1/2) + 0.2*Re**(2/3))* Pr**0.4 * Lambda_f[0,:]/d_p
    
    return h_t.flatten()






def get_IO_velocity_and_pressure(p_ref_n, p_set_pos, T_in_n, n_in_n, X_in_n, epsilon, r_tube, l_tube, d_cat_part, cell_z_centers): 
    
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
    X_in_n : 1D array
        [-] Inlet molar fractions
    n_in_n : array
        [mol s-1] Inlet molar flow of each specie
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
    mu_in_n : 
        [Pa s] Inlet gas mixture viscosity
    mu_out_n : 
        [Pa s] Outlet gas mixture viscosity

    """
    
    if p_set_pos == 'outlet': 
        # First get outlet values if we're setting p_out
        # Vol. flow rate and velocity
        Q_out_n, v_out_n = velocity_and_flowrate(n_in_n, p_ref_n, T_in_n, r_tube)
        # Concentrations at outlet[mol m-3] - we calculate pressure by assuming that there is no conversion at reactor so outlet concentration is methanol and water
        C_out_n = X_to_concentration(X_in_n, p_ref_n, T_in_n, Q_out_n, (np.pi * r_tube**2))
        # Density of outlet mixture - consisting of only MeOH and H2O
        rho_out_n = density_mixture(C_out_n, 1, p_ref_n, T_in_n) 
        # Get outlet mixture viscosity
        mu_out_n = gas_mixture_viscosity(X_in_n, T_in_n)
        # Get pressure in cells, inlet, and outlet
        field_p, p_in_n, p_out_n = ergun_pressure(p_ref_n, cell_z_centers, v_out_n, rho_out_n, d_cat_part, epsilon, mu_out_n, l_tube, p_set_pos)
        # Calculate inlet volumetric flow rate and flow velocity
        # v_in  [m s-1] inlet flow velocity (superficial fluid velocity)
        # Q_in  [m3 s-1] inlet volumetric flow rate
        Q_in_n, v_in_n = velocity_and_flowrate(n_in_n, p_in_n, T_in_n, r_tube)
        # Concentrations at inlet [mol m-3]
        C_in_n = X_to_concentration(X_in_n, p_in_n, T_in_n, Q_in_n, (np.pi * r_tube**2))
        # Density of inlet mixture 
        rho_in_n = density_mixture(C_in_n, 1, p_in_n, T_in_n) 
        
    elif p_set_pos == 'inlet': 
        # Calculate inlet volumetric flow rate and flow velocity
        # v_in  [m s-1] inlet flow velocity (superficial fluid velocity)
        # Q_in  [m3 s-1] inlet volumetric flow rate
        Q_in_n, v_in_n = velocity_and_flowrate(n_in_n, p_ref_n, T_in_n, r_tube)
        # Concentrations at inlet [mol m-3]
        C_in_n = X_to_concentration(X_in_n, p_ref_n, T_in_n, Q_in_n, (np.pi * r_tube**2))
        # Density of inlet mixture 
        rho_in_n = density_mixture(C_in_n, 1, p_ref_n, T_in_n) 
        # Get inlet mixture viscosity
        mu_in_n = gas_mixture_viscosity(X_in_n, T_in_n)
        # Calculate pressure field based on inlet values
        field_p, p_in_n, p_out_n = ergun_pressure(p_ref_n, cell_z_centers, v_in_n, rho_in_n, d_cat_part, epsilon, mu_in_n, l_tube, p_set_pos)
        # Vol. flow rate and velocity
        Q_out_n, v_out_n = velocity_and_flowrate(n_in_n, p_out_n, T_in_n, r_tube)
        # Concentrations at outlet[mol m-3] - we calculate pressure by assuming that there is no conversion at reactor so outlet concentration is methanol and water
        C_out_n = X_to_concentration(X_in_n, p_out_n, T_in_n, Q_out_n, (np.pi * r_tube**2))
        # Density of outlet mixture - consisting of only MeOH and H2O
        rho_out_n = density_mixture(C_out_n, 1, p_out_n, T_in_n) 

    
    field_v = get_velocity_field(v_in_n, v_out_n, l_tube, cell_z_centers)
    
    return v_in_n, v_out_n, field_v, p_in_n, p_out_n, field_p, Q_in_n, Q_out_n, C_in_n, rho_in_n, rho_out_n












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
        self.C_S2a = C_S2a   # 2a - MD - H2 adsorption site

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
    
    # Make a copy of original partial pressure fields, need this for later
    pi_copy = copy.deepcopy(p_i)
    pi_copy[pi_copy<p_limit] = 0 # Set every partial pressure below pi_limit to zero
    
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
    denominator = (1 + K_CH3O_1*(p_i[0]/p_i[2]**0.5) + K_HCOO_1*p_i[3]*p_i[2]**0.5 + K_OH_1 * (p_i[1]/p_i[2]**0.5)) * (1 + K_H_1a**0.5 * p_i[2]**0.5)
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
    
    
    # -- Reaction rate switches 
    reactant_presence = pi_copy != 0
    
    # Check where in the field we have reactants necessary for positive reaction rate
    MSR_pos_switch = np.logical_and(reactant_presence[0], reactant_presence[1]) # CH3OH and H2O
    MD_pos_switch = reactant_presence[0] # CH3OH 
    WGS_pos_switch = np.logical_and(reactant_presence[1], reactant_presence[4]) # H2O and CO
    # Check where in the field we have reactants necessary for negative reaction rate
    MSR_neg_switch = np.logical_and(reactant_presence[2], reactant_presence[3]) # CO2 and H2 
    MD_neg_switch = np.logical_and(reactant_presence[2], reactant_presence[4]) # CO and H2
    WGS_neg_switch = np.logical_and(reactant_presence[2], reactant_presence[3]) # CO2 and H2
    
    # Multiply positive 'check switches' with positive reaction rates and negative 'switches' with negative reaction rates
    # In this way we ensure limitations to reaction rates when there are no necessary reactants, and avoid negative concentrations in the field
    # This also enforces conservation of mass
    r_R_capped = np.clip(r_R, 0, None)*MSR_pos_switch + np.clip(r_R, None, 0)*MSR_neg_switch
    r_D_capped = np.clip(r_D, 0, None)*MD_pos_switch + np.clip(r_D, None, 0)*MD_neg_switch
    r_W_capped = np.clip(r_W, 0, None)*WGS_pos_switch + np.clip(r_W, None, 0)*WGS_neg_switch
    
    
    # return r_R_capped, r_D_capped*0, r_W_capped*0
    return r_R_capped, r_D_capped, r_W_capped




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
    r_H2 = (3*r_R + 2*r_D + r_W)*S_a
    r_CH3OH = - (r_R + r_D)*S_a
    r_H2O = - (r_R + r_W)*S_a
    
    # Nitrogen does not react
    r_N2 = r_CO2*0
    
    # Make an array
    rates = np.array((r_CH3OH, r_H2O, r_H2, r_CO2, r_CO, r_N2))
    
    
    return rates




def enthalpy_R(C_m, T):
    """
    Calculate temperature dependant ENTHALPY OF METHANOL STEAM REFORMING reaction

    Parameters
    ----------
    C_m : array         
        [J mol-1 K-1] Specific molar heat capacities of individual species
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
    
    H_R = 4.97e4 + (C_m[3] + 3*C_m[2] - C_m[0] - C_m[1]) * (T + 273.15 - 298)
    
    return H_R

def enthalpy_D(C_m, T):
    """
    Calculate temperature dependant ENTHALPY OF METHANOL DECOMPOSITION reaction

    Parameters
    ----------
    C_m : array         
        [J mol-1 K-1] Specific molar heat capacities of individual species
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
    
    H_D = 9.02e4 + (C_m[4] + 2*C_m[2] - C_m[0]) * (T + 273.15 - 298)
    
    return H_D

def enthalpy_W(C_m, T):
    """
    Calculate temperature dependant ENTHALPY OF WATER GAS SHIFT reaction

    Parameters
    ----------
    C_p : array         
        [J mol-1 K-1] Specific molar heat capacities of individual species
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
    
    H_W = -4.12e4 + (C_m[3] + C_m[2] - C_m[1] - C_m[4]) * (T + 273.15 - 298)
    
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
        adv_index = 0       # Internal index for discretization scheme
    elif adv_scheme == 'upwind_2o':
        adv_index = 1       # Internal index for discretization scheme
    elif adv_scheme.lower() == 'laxwendroff':
        adv_index = 2
    elif adv_scheme.lower() == 'beamwarming':
        adv_index = 3
    
        
    # Radial ghost cells
    if diff_scheme == 'central_2o': 
        diff_index = 0          # Internal index for discretization scheme
    elif diff_scheme == 'central_4o':
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
                            field_Ci_n, field_T_n, field_BET_cat, T_wall, T_hfluid):
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
    T_wall : 1D array
        [C] Wall temperature distribution array
    T_hfluid : 1D array
        [C] Heating fluid temperature distribution array

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
            
            
    # --- Get new mesh info
    d_z, d_r, cell_z_faces_L, cell_z_faces_R, cell_z_centers, \
            cell_r_faces_IN, cell_r_faces_EX, cell_r_centers, \
            cell_z_A, cell_r_A_IN, cell_r_A_EX, \
            cell_V = uniform_mesh(cells_ax_new, cells_rad_new, l_tube, r_tube)
    
    
    # --- 2D interpolation
    # Interpolation functions
    func_T          = RectBivariateSpline(np.flip(cell_r_centers_old), cell_z_centers_old, field_T_n)
    func_BET        = RectBivariateSpline(np.flip(cell_r_centers_old), cell_z_centers_old, field_BET_cat)
    func_CH3OH      = RectBivariateSpline(np.flip(cell_r_centers_old), cell_z_centers_old, field_Ci_n[0])
    func_H2O        = RectBivariateSpline(np.flip(cell_r_centers_old), cell_z_centers_old, field_Ci_n[1])
    func_H2         = RectBivariateSpline(np.flip(cell_r_centers_old), cell_z_centers_old, field_Ci_n[2])
    func_CO2        = RectBivariateSpline(np.flip(cell_r_centers_old), cell_z_centers_old, field_Ci_n[3])
    func_CO         = RectBivariateSpline(np.flip(cell_r_centers_old), cell_z_centers_old, field_Ci_n[4])
    func_N2         = RectBivariateSpline(np.flip(cell_r_centers_old), cell_z_centers_old, field_Ci_n[5])
    
    # Interpolation - get new values
    field_T_n       = func_T(np.flip(cell_r_centers), cell_z_centers)
    field_BET_cat   = func_BET(np.flip(cell_r_centers), cell_z_centers)
    field_CH3OH     = func_CH3OH(np.flip(cell_r_centers), cell_z_centers)
    field_H2O       = func_H2O(np.flip(cell_r_centers), cell_z_centers)
    field_H2        = func_H2(np.flip(cell_r_centers), cell_z_centers)
    field_CO2       = func_CO2(np.flip(cell_r_centers), cell_z_centers)
    field_CO        = func_CO(np.flip(cell_r_centers), cell_z_centers)
    field_N2        = func_N2(np.flip(cell_r_centers), cell_z_centers)
    
    # For species fields - join species array them in one 3D array
    field_Ci_n = np.dstack((field_CH3OH, field_H2O, field_H2, field_CO2, field_CO, field_N2))
    # Shuffle around the axes to match wanted output
    field_Ci_n = np.swapaxes(field_Ci_n, 1, 2)
    field_Ci_n = np.swapaxes(field_Ci_n, 0, 1)
    
    
    # --- 1D interpolation (for wall and heating fluid arrays)
    # Wall array
    func_Twall = interp1d(cell_z_centers_old, T_wall, fill_value='extrapolate') # fuction 
    T_wall = func_Twall(cell_z_centers) # interpolation
    # Heating fluid array
    if type(T_hfluid) is not str: # Only do this interpolation if we have a heating fluid here. If we're not using it, the value be in string format saying something like 'n/a'
        func_Thfluid = interp1d(cell_z_centers_old, T_hfluid, fill_value='extrapolate') # function
        T_hfluid = func_Thfluid(cell_z_centers) # interpolation
        
    
    # Make a return list and return
    return_list = [d_z, d_r, cell_z_faces_L, cell_z_faces_R, cell_z_centers, \
            cell_r_faces_IN, cell_r_faces_EX, cell_r_centers, \
            cell_z_A, cell_r_A_IN, cell_r_A_EX, \
            cell_V, \
            field_Ci_n, field_T_n, field_BET_cat, T_wall, T_hfluid]
        
    
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
    CH3OH_rate, H2O_rate, H2_rate, CO2_rate, CO_rate, N2_rate = formation_rates(MSR_rate, MD_rate, WGS_rate, field_BET)
    
    
    
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
    C_outlet = np.zeros((6)) # Make empty array for outlet concetrations
    molar_masses = np.zeros((6)) # Make empty array for molar masses
    
    # Fill empty arrays
    for specie in range(6):
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




def advection_flux_FL(phi_P, phi_E, phi_W, phi_WW, dz, u_s, dt, scheme):
    """
    Calculate advection flux in axial (z) direction, flux limiter included

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

    # Velocity W, WW, and E fields 
    u_s_W = np.roll(u_s, 1)
    u_s_W[0] = u_s_W[1]
    u_s_WW = np.roll(u_s_W, 1)
    u_s_WW[0] = u_s_WW[1]
    u_s_E = np.roll(u_s, -1)
    u_s_E[-1] = u_s_E[-2]
    
    
    # --- Flux limiter functions 
    # Get a measure of smoothness r at right and left faces 
    r_left, r_right = gv.flux.ratio_of_gradients(phi_P, phi_W, phi_WW, phi_E)
    # Get limiter left and right values
    limiter_left, limiter_right = gv.flux.limiter(r_left, r_right)
    
    
    # Left high order fluxes
    ho_flux_l = [lambda : 0,    # Upwind 1st order
                  lambda : phi_W*u_s_W - phi_WW*u_s_WW, # Upwind 2nd order
                  lambda : phi_P*u_s - phi_W*u_s_W - dt/dz*(phi_P*u_s**2 - phi_W*u_s_W**2), # Lax Wendroff
                  lambda : phi_W*u_s_W - phi_WW*u_s_WW - dt/dz*(phi_W*u_s_W**2 - phi_WW*u_s_WW**2)] # Beam Warming
    
    # Right high order fluxes
    ho_flux_r = [lambda : 0, # Upwind 1st order
                  lambda : phi_P*u_s - phi_W*u_s_W, # Upwind 2nd order
                  lambda : phi_E*u_s_E - phi_P*u_s - dt/dz*(phi_E*u_s_E**2 - phi_P*u_s**2), # Lax Wendroff
                  lambda : phi_P*u_s - phi_W*u_s_W - dt/dz*(phi_P*u_s**2 - phi_W*u_s_W**2)] # Beam Warming
    

    # Combine low and high order flux with limiter to get limited values of left and right flux 
    left_flux = phi_W*u_s_W + limiter_left* 0.5 * ho_flux_l[scheme]()
    right_flux = phi_P*u_s + limiter_right* 0.5 * ho_flux_r[scheme]()

    # combine into one single flux at cell point P
    phi_flux = (left_flux - right_flux) / dz
    
    return phi_flux



def advection_flux_no_FL(phi_P, phi_E, phi_W, phi_WW, dz, u_s, dt, scheme):
    """
    Calculate advection flux in axial (z) direction, without flux limiter (not needed in steady or temperature field advection)

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
    
    # Velocity W, WW, and E fields 
    u_s_W = np.roll(u_s, 1)
    u_s_W[0] = u_s_W[1]
    u_s_WW = np.roll(u_s_W, 1)
    u_s_WW[0] = u_s_WW[1]
    u_s_E = np.roll(u_s, -1)
    u_s_E[-1] = u_s_E[-2]
    
    
    # Left high order fluxes
    ho_flux_l = [lambda : 0,    # Upwind 1st order
                  lambda : phi_W*u_s_W - phi_WW*u_s_WW, # Upwind 2nd order
                  lambda : phi_P*u_s - phi_W*u_s_W - dt/dz*(phi_P*u_s**2 - phi_W*u_s_W**2), # Lax Wendroff
                  lambda : phi_W*u_s_W - phi_WW*u_s_WW - dt/dz*(phi_W*u_s_W**2 - phi_WW*u_s_WW**2)] # Beam Warming
    
    # Right high order fluxes
    ho_flux_r = [lambda : 0, # Upwind 1st order
                  lambda : phi_P*u_s - phi_W*u_s_W, # Upwind 2nd order
                  lambda : phi_E*u_s_E - phi_P*u_s - dt/dz*(phi_E*u_s_E**2 - phi_P*u_s**2), # Lax Wendroff
                  lambda : phi_P*u_s - phi_W*u_s_W - dt/dz*(phi_P*u_s**2 - phi_W*u_s_W**2)] # Beam Warming
    

    # Combine low and high order flux with limiter to get limited values of left and right flux 
    left_flux = phi_W*u_s_W + 0.5 * ho_flux_l[scheme]()
    right_flux = phi_P*u_s + 0.5 * ho_flux_r[scheme]()
    
    # combine into one single flux at cell point P
    phi_flux = (left_flux - right_flux) / dz
    
    return phi_flux




def advection_flux_CD_T(phi_P, phi_E, phi_W, phi_WW, dz, u_s, dt, scheme):
    """
    ...

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
    
    
    # Velocity W, WW, and E fields 
    u_s_W = np.roll(u_s, 1)
    u_s_W[0] = u_s[0]
    u_s_WW = np.roll(u_s_W, 1)
    u_s_WW[0] = u_s[0]
    u_s_E = np.roll(u_s, -1)
    u_s_E[-1] = u_s[-1]
    
    phi_flux = (phi_E*u_s_E - phi_W*u_s_W) / (2*dz)
    phi_flux[:,0] = (3*phi_P[:,0]*u_s[0] - 4*phi_W[:,0]*u_s_W[0] + phi_WW[:,0]*u_s_WW[0])*(2*dz[:,0])
    
    return phi_flux



def advection_flux_CD_C(phi_P, phi_E, phi_W, phi_WW, dz, u_s, dt, scheme):
    """
    ...

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
    
    
    # Velocity W, WW, and E fields 
    u_s_W = np.roll(u_s, 1)
    u_s_W[0] = u_s[0]
    u_s_WW = np.roll(u_s_W, 1)
    u_s_WW[0] = u_s[0]
    u_s_E = np.roll(u_s, -1)
    u_s_E[-1] = u_s[-1]
    
    phi_flux = (phi_E*u_s_E - phi_W*u_s_W) / (2*dz)
    phi_flux[:,:,0] = -(3*phi_P[:,:,0]*u_s[0] - 4*phi_W[:,:,0]*u_s_W[0] + phi_WW[:,:,0]*u_s_WW[0]) / (2*dz[:,0])
    # phi_flux[:,:,1] = -(3*phi_P[:,:,1]*u_s[1] - 4*phi_W[:,:,1]*u_s_W[1] + phi_WW[:,:,1]*u_s_WW[1]) / (2*dz[:,1])
    
    return phi_flux





def diffusion_mass_flux(phi_P, phi_EX, phi_EXX, phi_IN, phi_INN, dr, r_P, p_s, scheme):
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
        [] Propagation speed (D_er)
    scheme : int
        Diffusion scheme choice

    Returns
    -------
    phi_flux : 3D array
        Diffusion flux at point P

    """

    # NOTE: in mass diffusion, the speed of propagation D_er (effective radial mass difusion coefficient) depends only on velocity
    # Thus it is constant at any dz, and we can multiply it at the end in total flux

    # Coefficients for 1st derivative via central differencing scheme
    # [EXX, EX, IN, INN, dr]
    coeffs_1d = [[0, 1, -1, 0, 2],     # 2nd order CD
                  [-1, 8, -8, 1, 12]]   # 4th order CD
    
    f1 = coeffs_1d[scheme] # Multiplication factors for first derivative
    
    # first derivative evaluation
    d_phi = (phi_EXX*f1[0] + phi_EX*f1[1] + phi_IN*f1[2] + phi_INN*f1[3]) / (dr*f1[4])
    # Second derivative evaluation
    d2_phi = (phi_EX - 2*phi_P + phi_IN) / (dr**2)
    
    # Total flux     
    phi_flux = p_s * (d2_phi + d_phi/r_P)
    
    
    
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
        [W m-1 K-1] Propagation speed (Lambda_er)
    h_t : 1D array
        [W m-2 K-1] Heat transfer coefficient of tube-side film
    scheme : int
        Diffusion scheme choice

    Returns
    -------
    phi_flux : 2D array
        Diffusion flux at point P

    """
    
    
    # # --- Fields for propagation speeds 
    p_s_EX = np.roll(p_s, 1, 0)
    p_s_EX[0] = p_s_EX[1]
    
    p_s_EXX = np.roll(p_s_EX, 1, 0)
    p_s_EXX[0] = p_s_EXX[1]
    
    p_s_IN = np.roll(p_s, -1, 0)
    p_s_IN[-1] = p_s_IN[-2]
    
    p_s_INN = np.roll(p_s_IN, -1, 0)
    p_s_INN[-1] = p_s[-2] # Mirror condition
    
    
    # # --- Near wall cells
    # Special treatment
    # Fluxes at the wall (r=R) - defined with a boundary condition 
    wall_BC = -(h_t / p_s[0,:]) * (phi_P[0,:] - phi_EX[0,:])  
    # Calculate value at ghost cell in the wall using the defined flux
    phi_gcells = wall_BC*dr[0] + phi_P[0,:]
    # Set ghost cells in external cell arrays 
    phi_EX[0] = phi_gcells
    phi_EXX[0] = phi_gcells
    phi_EXX[1] = phi_gcells
    
    
    ## -- Field cells - normal treatment
    # Coefficients for 1st derivative via central differencing scheme
    # [EXX, EX, IN, INN, dr]
    coeffs_1d = [[0, 1, -1, 0, 2],     # 2nd order CD
                  [-1, 8, -8, 1, 12]]   # 4th order CD

    f1 = coeffs_1d[scheme] # Multiplication factors for first derivative
    
    
    d_phi = (phi_EXX*f1[0]*p_s_EXX + phi_EX*f1[1]*p_s_EX + phi_IN*f1[2]*p_s_IN + phi_INN*f1[3]*p_s_INN) / (dr*f1[4])
    # Second derivative evaluation
    d2_phi = (phi_EX*p_s_EX -2*phi_P*p_s + phi_IN*p_s_IN) / (dr**2)
    
    # Total flux - multiplication with propagation speed (Lambda) and accounting for radial coordinates
    phi_flux = (d2_phi + d_phi/r_P)    
    
    
    
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
    field_C_W[:, :, 0] = np.rot90( np.ones((cells_rad, 6)) * np.asarray(C_in_n), 3 ) # Dirichliet BC
    field_T_W[:,0] = T_in_n # Dirichliet BC
    # WEST WEST - inlet
    field_C_WW[:, :, 0] = np.rot90( np.ones((cells_rad, 6)) * np.asarray(C_in_n), 3 ) # Dirichliet BC
    field_C_WW[:, :, 1] = np.rot90( np.ones((cells_rad, 6)) * np.asarray(C_in_n), 3 ) # Dirichliet BC
    field_T_WW[:,0] = T_in_n # Dirichliet BC
    field_T_WW[:,1] = T_in_n # Dirichliet BC
    # EAST - outlet
    field_C_E[:, :, -1] = field_Ci_n[:, :, -1] # Neumann BC
    field_T_E[:,-1] = field_T_n[:,-1] # Neumann BC
    # EXTERNAL - wall
    field_C_EX[:, 0, :] =  field_Ci_n[:, 0, :] # Neumann BC
    field_T_EX[0, :] = T_wall_n # Dirichliet BC
    # EXTERNAL EXTERNAL - wall
    field_C_EXX[:, 0, :] =  field_Ci_n[:, 1, :] # Neumann BC
    field_C_EXX[:, 1, :] =  field_Ci_n[:, 0, :] # Neumann BC
    field_T_EXX[0, :] = T_wall_n # Dirichliet BC
    field_T_EXX[1, :] = T_wall_n # Dirichliet BC
    # INTERNAL - symmetry axis
    field_C_IN[:, -1, :] =  field_Ci_n[:, -1, :] # Neumann BC
    field_T_IN[-1, :] = field_T_n[-1, :] # Neumann BC
    # INTERNAL INTERNAL 
    field_C_INN[:, -1, :] =  field_Ci_n[:, -2, :] # Neumann BC - symmetric
    field_C_INN[:, -2, :] =  field_Ci_n[:, -1, :] # Neumann BC
    field_T_INN[-1, :] = field_T_n[-2, :] # Neumann BC - symmetryc
    field_T_INN[-2, :] = field_T_n[-1, :] # Neumann BC
    
    return field_C_W, field_C_WW, field_C_E, field_C_EX, field_C_EXX, field_C_IN, field_C_INN, \
        field_T_W, field_T_WW, field_T_E, field_T_EX, field_T_EXX, field_T_IN, field_T_INN



def RK4_fluxes(dt_RK4, times_RK4, cells_rad, cells_ax, cell_V, cell_r_centers, cell_z_centers,
        dz_mgrid, dr_mgrid, z_centers_mgrid, r_centers_mgrid,
        W_cat, SC_ratio, X_i_RK, r_tube, 
        T_wall, T_hfluid, m_condensate, T_in_RK4, flowrate_in_RK4, p_ref_RK4, p_set_pos,
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
    T_wall : 1D array
        [C] Wall temperature profile
    T_hfluid : 1D array
        [C] Heating fluid temperature profile
    m_condensate  : 1D array
        [kg s-1] Rate of condensation axial profile
    T_in_RK4 : 1D array
        [C] Inlet fluid temperature at RK4 times
    flowrate_in_RK4 : 1D array
        [user defined] Inlet flow rate 
    p_ref_RK4 : 1D array
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
    
    # Make empty flux fields
    C_flux_arrays = np.zeros((4, 6, cells_rad, cells_ax))
    T_flux_arrays = np.zeros((4, cells_rad, cells_ax))
    Tw_flux_arrays = np.zeros((4, cells_ax))
    Thf_flux_arrays = np.zeros((4, cells_ax))
    m_cond_flux_arrays = np.zeros((4, cells_ax))
    
    # Make temporary value array
    # Need to do a deep copy, otherwise the original gets changed as well
    C_array = copy.deepcopy(field_Ci)
    T_array = copy.deepcopy(field_T)
        
    # --- Calculate flux F0 first 
    # Some things are already set/calculated for the first flux
    # They are: pressure field, D_er, inlet velocity
    for RKt in range(1): # do this just to keep syntax the same
        # Get inlet molar flow rate and molar fractions at RK time
        ni_in_RK, Xi_in_RK = gv.inlet.CX(flowrate_in_RK4[RKt])    
    # Get inlet/outlet/field velocities and pressures, and some other variables 
        v_in_RK, v_out_RK, field_v_RK, \
            p_in_RK, p_out_RK, field_p, \
            Q_in_RK, Q_out_RK, C_in_RK, rho_in_RK, rho_out_RK = get_IO_velocity_and_pressure(p_ref_RK4[RKt], p_set_pos, T_in_RK4[RKt], ni_in_RK, Xi_in_RK, epsilon, r_tube, l_tube, d_cat_part, cell_z_centers)
        # Radial diffusion coefficient
        D_er_RK = radial_diffusion_coefficient(field_v_RK, d_cat_part, d_tube_in)
       
        # Calculate wall temperature profile
        Tw_flux_arrays[RKt], Thf_flux_arrays[RKt], m_cond_flux_arrays[RKt] = gv.Twall_func.dynamic(times_RK4[RKt], dt_RK4[RKt], T_wall, C_array, T_array, field_v_RK, T_hfluid, m_condensate)
        
        # Get neighbouring cell fields                
        field_C_W, field_C_WW, field_C_E, \
            field_C_EX, field_C_EXX, field_C_IN, field_C_INN, \
            field_T_W, field_T_WW, field_T_E, \
                field_T_EX, field_T_EXX, field_T_IN, field_T_INN = get_neighbour_fields(C_array, T_array, cells_rad, C_in_RK, T_in_RK4[RKt], T_wall)
        
        # Calculate flux
        C_flux_arrays[RKt], T_flux_arrays[RKt] = Euler_fluxes(D_er_RK, field_v_RK, field_p,
            dz_mgrid, dr_mgrid, cell_V, r_centers_mgrid, dt_RK4[RKt],
            C_array, field_C_W, field_C_WW, field_C_E, field_C_IN,field_C_INN, field_C_EX, field_C_EXX,
            T_array, field_T_W, field_T_WW, field_T_E, field_T_IN, field_T_INN, field_T_EX, field_T_EXX,
            bulk_rho_c, BET_cat_P, d_cat_part, cat_cp, cat_shape,
            epsilon, N, adv_scheme, diff_scheme, pi_limit, nu_i, nu_j)
            
    # --- Calculate RK4 fluxes 2,3,4
    # First flux already calculated
    for RKt in range(1,4): # RKt = runge kutta time 
        # First, evolve values with previously calculated slopes/fluxes
        C_array = field_Ci + (dt_RK4[RKt] * C_flux_arrays[RKt-1])
        # Temperature / heat           
        T_array = field_T + (dt_RK4[RKt] * T_flux_arrays[RKt-1])
        
        Tw_array = T_wall + (dt_RK4[RKt] * Tw_flux_arrays[RKt-1])
        Thf_array = T_hfluid + (dt_RK4[RKt] * Thf_flux_arrays[RKt-1])
        m_cond_array = m_condensate + (dt_RK4[RKt]* m_cond_flux_arrays[RKt-1])
        
        # Get inlet molar flow rate and molar fractions at RK time
        ni_in_RK, Xi_in_RK = gv.inlet.CX(flowrate_in_RK4[RKt]) 
        # Get inlet/outlet/field velocities and pressures, and some other variables 
        v_in_RK, v_out_RK, field_v_RK, \
            p_in_RK, p_out_RK, field_p, \
            Q_in_RK, Q_out_RK, C_in_RK, rho_in_RK, rho_out_RK  = get_IO_velocity_and_pressure(p_ref_RK4[RKt], p_set_pos, T_in_RK4[RKt], ni_in_RK, Xi_in_RK, epsilon, r_tube, l_tube, d_cat_part, cell_z_centers)
        # Calculate new radial diffusion coefficient
        D_er_RK = radial_diffusion_coefficient(field_v_RK, d_cat_part, d_tube_in)
        
        # Get fields of neighbouring cells                 
        field_C_W, field_C_WW, field_C_E, \
            field_C_EX, field_C_EXX, field_C_IN, field_C_INN, \
            field_T_W, field_T_WW, field_T_E, \
                field_T_EX, field_T_EXX, field_T_IN, field_T_INN = get_neighbour_fields(C_array, T_array, cells_rad, C_in_RK, T_in_RK4[RKt], Tw_array)
    
        # Calculate fluxes
        C_flux_arrays[RKt], T_flux_arrays[RKt] = Euler_fluxes(D_er_RK, field_v_RK, field_p,
            dz_mgrid, dr_mgrid, cell_V, r_centers_mgrid, dt_RK4[RKt],
            C_array, field_C_W, field_C_WW, field_C_E, field_C_IN,field_C_INN, field_C_EX, field_C_EXX,
            T_array, field_T_W, field_T_WW, field_T_E, field_T_IN, field_T_INN, field_T_EX, field_T_EXX,
            bulk_rho_c, BET_cat_P, d_cat_part, cat_cp, cat_shape,
            epsilon, N, adv_scheme, diff_scheme, pi_limit, nu_i, nu_j)
        
        Tw_flux_arrays[RKt], Thf_flux_arrays[RKt], m_cond_flux_arrays[RKt] = gv.Twall_func.dynamic(times_RK4[RKt], dt_RK4[RKt], Tw_array, C_array, T_array, field_v_RK, Thf_array, m_cond_array)
               
    # --- Finally, make arrays of RK4 fluxes
    Ci_fluxes = (C_flux_arrays[0] + 2*C_flux_arrays[1] + 2*C_flux_arrays[2] + C_flux_arrays[3])/6
    T_fluxes = (T_flux_arrays[0] + 2*T_flux_arrays[1] + 2*T_flux_arrays[2] + T_flux_arrays[3])/6
    
    # Wall and heating fluid
    Tw_fluxes = (Tw_flux_arrays[0] + 2*Tw_flux_arrays[1] + 2*Tw_flux_arrays[2] + Tw_flux_arrays[3])/6
    Thf_fluxes = (Thf_flux_arrays[0] + 2*Thf_flux_arrays[1] + 2*Thf_flux_arrays[2] + Thf_flux_arrays[3])/6
    # Steam condensate 
    m_condensate_fluxes = (m_cond_flux_arrays[0] + 2*m_cond_flux_arrays[1] + 2*m_cond_flux_arrays[2] + m_cond_flux_arrays[3])/6

    return Ci_fluxes, T_fluxes, Tw_fluxes, Thf_fluxes, m_condensate_fluxes



def Euler_fluxes(D_er, v_n, p_P,
        cell_dz, cell_dr, cell_V, cell_r_center, dt,
        Ci_P, Ci_W, Ci_WW, Ci_E, Ci_IN, Ci_INN, Ci_EX, Ci_EXX,
        T_P, T_W, T_WW, T_E, T_IN, T_INN, T_EX, T_EXX,
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
    mass_adv_i = advection_flux_FL(Ci_P, Ci_E, Ci_W, Ci_WW, cell_dz, v_n, dt, adv_scheme)
    
    # Mass diffusion term
    mass_diff_i = diffusion_mass_flux(Ci_P, Ci_EX, Ci_EXX, Ci_IN, Ci_INN, cell_dr, cell_r_center, D_er, diff_scheme)
    
    
    # Mass source term 
    # Mole fractions of species
    X_i = concentration_to_mole_fraction(Ci_P) 
    # partial pressure
    p_i = partial_pressure(p_P, X_i)
    # Reaction rates
    r_R, r_D, r_W = reaction_rates(p_i, T_P, pi_limit) 
    # Formation rates
    r_i = formation_rates(r_R, r_D, r_W, BET_cat_P)
    # Mass source / sink
    mass_source_i = (nu_i * bulk_rho_c * r_i) 
    
    # Mass flux for time stepping (dCi / dt)
    mass_flux = (mass_adv_i + mass_diff_i + mass_source_i)/epsilon
    
    # --- Heat transport 
    # Specific heat capacity of every specie
    Cm_i = Cm_species(T_P)
    # Mixture heat capacity - specific and molar
    Cm_mix = Cm_mixture(X_i, Cm_i) # [J mol-1 K-1]
    # Density of a mixture 
    rho_mix_mol = mol_density_mixture(Ci_P) 
    # Effective radial thermal conductivity
    Lambda_er = radial_thermal_conductivity(v_n, rho_mix_mol, Cm_mix, d_cat_part, X_i, T_P, epsilon, N, cat_shape)    
    # Get heat transfer coefficient on the tube side
    h_t = heat_transfer_coeff_tube(v_n, d_cat_part, T_P, Ci_P, epsilon)
    
    # Heat advection term
    heat_adv_i = Cm_mix*rho_mix_mol* advection_flux_no_FL(T_P, T_E, T_W, T_WW, cell_dz, v_n, dt, adv_scheme)
    
    # Heat diffusion term
    heat_diff_i = heat_diffusion_flux(T_P, T_EX, T_EXX, T_IN, T_INN, cell_dr, cell_r_center, Lambda_er, h_t, diff_scheme)
    
    # Heat source / sink from individual reactions
    source_R = nu_j * r_R * (- enthalpy_R(Cm_i, T_P) )
    source_D = nu_j * r_D * (- enthalpy_D(Cm_i, T_P) )
    source_W = nu_j * r_W * (- enthalpy_W(Cm_i, T_P) )

    # Total heat source/sink   
    heat_source_i = (source_R + source_D + source_W) * bulk_rho_c *  BET_cat_P
    
    # Heat flux for time stepping (dT / dt)
    heat_flux = (heat_adv_i + heat_diff_i + heat_source_i) / (epsilon * rho_mix_mol * Cm_mix + bulk_rho_c * cat_cp)
    
    return mass_flux, heat_flux






def steady_crank_nicholson(field_Ci_n, field_T_n, cells_rad, C_in_n, T_in_n, T_wall_n, T_hfluid_n, m_condensation_n,\
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
    T_hfluid_n : 1D array
        [C] Array of heating fluid temperature
    m_condensation_n : 1D array
        [kg s-1] Amount of steam condensing on the reactor tube outer surface
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
    Tw_fluxes_CN : 1D array
        [C s-1] Flux field of wall temperature
    Thf_fluxes_CN : 1D array
        [C s-1] Flux field of heating fluid temperature
    mcond_fluxes_CN : 1D array
        [kg s-2] Flux field of steam condensation mass flow for condensing steam case

    """
    
     # Get fields of neighbouring cells
    field_C_W, field_C_WW, field_C_E, \
        field_C_EX, field_C_EXX, field_C_IN, field_C_INN, \
        field_T_W, field_T_WW, field_T_E, \
            field_T_EX, field_T_EXX, field_T_IN, field_T_INN = get_neighbour_fields(field_Ci_n, field_T_n, cells_rad, C_in_n, T_in_n, T_wall_n)
    
    # Get first fluxes
    C_fluxes, T_fluxes = steady_Euler_fluxes(field_D_er, field_v, field_p,
        dz_mgrid, dr_mgrid, cell_V, r_centers_mgrid, relax,
        field_Ci_n, field_C_W, field_C_WW, field_C_E, field_C_IN, field_C_INN, field_C_EX, field_C_EXX, 
        field_T_n, field_T_W, field_T_WW, field_T_E, field_T_IN, field_T_INN, field_T_EX, field_T_EXX,
        rho_cat_bulk, field_BET_cat, d_cat_part, cat_cp, cat_shape,
        epsilon, N, adv_index, diff_index, pi_limit, nu_i, nu_j)
    
    
    Tw_fluxes, Thf_fluxes, mcond_fluxes = gv.Twall_func.steady(T_wall_n, field_Ci_n, field_T_n, field_v, T_hfluid_n, m_condensation_n)

    
    # From fluxes at n, get timestep n+1
    field_Ci_n1 = field_Ci_n + C_fluxes * relax
    field_T_n1 = field_T_n + T_fluxes * relax
    
    T_wall_n1 = T_wall_n + Tw_fluxes * relax
    T_hfluid_n1 = T_hfluid_n + Thf_fluxes * relax
    m_condensation_n1 = m_condensation_n + mcond_fluxes * relax
    
    
    # Get fields of neighbouring cells for n1
    field_C_W, field_C_WW, field_C_E, \
        field_C_EX, field_C_EXX, field_C_IN, field_C_INN, \
        field_T_W, field_T_WW, field_T_E, \
            field_T_EX, field_T_EXX, field_T_IN, field_T_INN = get_neighbour_fields(field_Ci_n1, field_T_n1, cells_rad, C_in_n, T_in_n, T_wall_n1)
    
    # Get another set of fluxes at n+1
    C_fluxes_n1, T_fluxes_n1 = steady_Euler_fluxes(field_D_er, field_v, field_p,
        dz_mgrid, dr_mgrid, cell_V, r_centers_mgrid, relax,
        field_Ci_n1, field_C_W, field_C_WW, field_C_E, field_C_IN, field_C_INN, field_C_EX, field_C_EXX, 
        field_T_n1, field_T_W, field_T_WW, field_T_E, field_T_IN, field_T_INN, field_T_EX, field_T_EXX,
        rho_cat_bulk, field_BET_cat, d_cat_part, cat_cp, cat_shape,
        epsilon, N, adv_index, diff_index, pi_limit, nu_i, nu_j)
    
    
    Tw_fluxes_n1, Thf_fluxes_n1, mcond_fluxes_n1 = gv.Twall_func.steady(T_wall_n1, field_Ci_n1, field_T_n1, field_v, T_hfluid_n1, m_condensation_n1)

    
    C_fluxes_CN = (1/2)*(C_fluxes + C_fluxes_n1)
    T_fluxes_CN = (1/2)*(T_fluxes + T_fluxes_n1)
    
    Tw_fluxes_CN = (1/2)*(Tw_fluxes + Tw_fluxes_n1)
    Thf_fluxes_CN = (1/2)*(Thf_fluxes + Thf_fluxes_n1)
    
    mcond_fluxes_CN = (1/2)*(mcond_fluxes + mcond_fluxes_n1)
    
    return C_fluxes_CN, T_fluxes_CN, Tw_fluxes_CN, Thf_fluxes_CN, mcond_fluxes_CN





def steady_Euler_fluxes(D_er, v_n, p_P,
        cell_dz, cell_dr, cell_V, cell_r_center, dt,
        Ci_P, Ci_W, Ci_WW, Ci_E, Ci_IN, Ci_INN, Ci_EX, Ci_EXX,
        T_P, T_W, T_WW, T_E, T_IN, T_INN, T_EX, T_EXX,
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
    mass_adv_i = advection_flux_no_FL(Ci_P, Ci_E, Ci_W, Ci_WW, cell_dz, v_n, dt, adv_scheme)
    # mass_adv_i = advection_flux_CD_C(Ci_P, Ci_E, Ci_W, Ci_WW, cell_dz, v_n, dt, adv_scheme)
    
    # Mass diffusion term
    mass_diff_i = diffusion_mass_flux(Ci_P, Ci_EX, Ci_EXX, Ci_IN, Ci_INN, cell_dr, cell_r_center, D_er, diff_scheme)

    # Mass source term 
    # Mole fractions of species
    X_i = concentration_to_mole_fraction(Ci_P) 
    
    # partial pressure
    p_i = partial_pressure(p_P, X_i)
    
    # Reaction rates
    r_R, r_D, r_W = reaction_rates(p_i, T_P, pi_limit) 
    # r_R, r_D, r_W = 0,0,0
    # Clip MSR rate because it can get very high from the initial guess in steady state
    r_R = np.clip(r_R, None, 1e-2)
    # Formation rates
    r_i = formation_rates(r_R, r_D, r_W, BET_cat_P) 
    # Mass source / sink
    mass_source_i = (nu_i * bulk_rho_c * r_i)
    
    # Total mass flux
    mass_flux = (mass_adv_i + mass_diff_i + mass_source_i) /epsilon
    
    # --- Heat transport 
    # Specific molar heat capacity of every specie
    Cm_i = Cm_species(T_P) 
    # Mixture specific heat capacity
    Cm_mix = Cm_mixture(X_i, Cm_i) # [J mol-1 K-1]
    
    # Density of a mixture 
    rho_mix_mol = mol_density_mixture(Ci_P) 
    
    # Effective radial thermal conductivity
    Lambda_er = radial_thermal_conductivity(v_n, rho_mix_mol, Cm_mix, d_cat_part, X_i, T_P, epsilon, N, cat_shape)    
    # Get heat transfer coefficient on the tube side
    h_t = heat_transfer_coeff_tube(v_n, d_cat_part, T_P, Ci_P, epsilon)
    
    # Heat advection term
    heat_adv_i = Cm_mix*rho_mix_mol* advection_flux_no_FL(T_P, T_E, T_W, T_WW, cell_dz, v_n, dt, adv_scheme)
    # heat_adv_i = Cm_mix*rho_mix_mol*advection_flux_CD_T(T_P, T_E, T_W, T_WW, cell_dz, v_n, dt, adv_scheme)
    
    # Heat diffusion term
    heat_diff_i = heat_diffusion_flux(T_P, T_EX, T_EXX, T_IN, T_INN, cell_dr, cell_r_center, Lambda_er, h_t, diff_scheme)
    
    # Heat source / sink from individual reactions
    source_R = nu_j * r_R * (- enthalpy_R(Cm_i, T_P) )
    source_D = nu_j * r_D * (- enthalpy_D(Cm_i, T_P) ) 
    source_W = nu_j * r_W * (- enthalpy_W(Cm_i, T_P) )
    # Total heat source/sink   
    heat_source_i = (source_R + source_D + source_W) * bulk_rho_c * BET_cat_P
    
    # Heat flux for time stepping (dT / dt)
    heat_flux = (heat_adv_i + heat_diff_i + heat_source_i) / (Cm_mix * rho_mix_mol)  #/ (epsilon * rho_mix_mol * Cm_mix + bulk_rho_c * cat_cp)
    
    
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
    Calculates second derivatives of wall temperature array with 4th order central bounded difference, used for heat diffusion in wall

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
    
    T_E = np.roll(T_P, -1)
    T_E[-1] = T_P[-1] # Eastmost boundary
    T_EE = np.roll(T_E,-1)
    T_EE[-1] = T_P[-2] # Eastmost boundary
    
    T_W = np.roll(T_P, 1)
    T_W[0] = T_P[0] # Westmost boundary
    T_WW = np.roll(T_W, 1)
    T_WW[0] = T_P[1] # Westmost boundary
    
    
    # Second derivative evaluation 2nd order
    d2_T_wall = (T_E - 2*T_P + T_W) / (dz**2)
    
    # Second derivative evaluation 4th order
    # d2_T_wall = (-T_EE + 16*T_E - 30*T_P + 16*T_W - T_WW) / (12*dz**2)
    
    return d2_T_wall






def flue_gas_first_derivative(T_P, dz, inlet_value, flow_direction):
    """
    Calculates first derivatives of flue gas temperature array with 2nd order upwind scheme

    Parameters
    ----------
    T_P : 1D array
        [K] Flue gas temperatures
    dz : float
        [m] Cell spacing array
    inlet_value : float
        [C] West-most value of transported value (temperature)
    flow_direction : float
        [-] factor for determining flow direction in the upwind scheme

    Returns
    -------
    d_T_fgas : 1D array
        [K m-1] First derivative of flue gas array

    """
    # Choose whether we work with flipped array or not (upwind scheme depends on the flow direction)
    # Do this so that we always work with flue gas flow west-to-east
    T_P = (0.5+flow_direction*0.5)*T_P + (0.5-flow_direction*0.5)*np.flip(T_P)
    
    # Make arrays for W and WW cells 
    T_W = np.roll(T_P, 1)
    T_W[0] = inlet_value # Westmost boundary
    T_WW = np.roll(T_W, 1)
    T_WW[0] = inlet_value # Westmost boundary
    
    
    # Calculate first derivative (upwind)
    d_T_fgas = ( (3*T_P -4*T_W + T_WW) / (dz*2) ) # 2nd order
    # d_T_fgas = ( (T_P -T_W) / (dz) ) # 1st order
    
    
    # Flip the array back (if necessary)
    d_T_fgas = (0.5+flow_direction*0.5)*d_T_fgas + (0.5-flow_direction*0.5)*np.flip(d_T_fgas)
    
    
    
    
    
    # # CENTRAL DIFFERENCING
    # # Make arrays for W and WW cells 
    # T_W = np.roll(T_P, 1)
    # T_W[0] = inlet_value # Westmost boundary
    # T_WW = np.roll(T_W, 1)
    # T_WW[0] = inlet_value # Westmost boundary
    # T_E = np.roll(T_P, -1)
    # T_E[-1] = T_P[-1]
    
    # d_T_fgas = (T_E - T_W) / (dz*2)
    # d_T_fgas[0] = (3*T_P[0] - 4*T_W[0] + T_WW[0]) / (dz*2)
    
    
    return d_T_fgas







# -------------------------------------------------------------
# --- REACTOR HEATING FUNCTIONS
# -------------------------------------------------------------


# -- flue gas heating

def log_mean_diameter(d_out, d_in):
    """
    Calculate log mean diameter of a reactor tube

    Parameters
    ----------
    d_out : float
        [m] Outer tube diameter
    d_in : float
        [Inner tube diameter]

    Returns
    -------
    d_lm : float
        [m] Log mean diameter

    """
    
    # Log mean diameter
    d_lm = (d_out - d_in) / np.log((d_out/d_in))
    
    return d_lm
    

def baffle_window_area_fraction(d_shell, opening_percent_d):
    """
    Calculate area fraction of baffle plate window based on diameter percentage that is window

    Parameters
    ----------
    d_shell : float
        [m] Reactor shell inner diameter
    opening_percent_d : float
        [-] Baffle window height as a percentage of diameter opening

    Returns
    -------
    f_bw : float
        [-] Area fraction of a baffle window (circle segment)

    """
    #Segment area (A_seg) is sector area (A_sec) minus triangle area (A_tr)
    r_shell = d_shell/2
    # Height (sagitta) of segment area
    h = d_shell * opening_percent_d
    # Area of segment
    A_seg = r_shell**2 * np.arccos((1 - (h/r_shell))) - (r_shell-h)*np.sqrt( (r_shell**2 - (r_shell-h)**2) )
    
    # Area of shell cross section
    A_shell = r_shell**2 * np.pi
    # Area fraction 
    f_bw = A_seg / A_shell
    return f_bw



def gas_mixture_thermal_conductivity(X_i, species_indices, T_C):
    """
    Calculate thermal conductivity of a gas mixture

    Parameters
    ----------
    X_i : 3D         
        [-] Molar fraction of components
    species_indices : array
        [-] indices of species used in this estimation (indices within Specie class, _reg)
    T_C : 2D array            
        [C] Temperature
    Returns
    -------
    Lambda_f : 2D array     
        [W m-1 K-1] Gas mixture thermal conductivity
    
    """
    
    N_s = len(X_i) # Number of species used in this calculation
    
    # Convert temperature to Kelvin 
    T = T_C + 273.15
    
    
    # Lambda_i - thermal conductivites of component i
    Lambda_i = np.zeros((N_s, np.shape(T)[0]))
    
    
    for i in range(N_s):
        i_reg = species_indices[i] # Get index of the species we are using in this calculation
        Lambda_i[i] = Specie._reg[i_reg].La + Specie._reg[i_reg].Lb*T + Specie._reg[i_reg].Lc*T**2 + Specie._reg[i_reg].Ld*T**3 +Specie._reg[i_reg].Le*(1/T)**2

    # Lambda_i is in unit [mW m-1 K-1], convert to [W m-1 K-1]
    Lambda_i = Lambda_i * 0.001

    # NOTE: Array Lambda_i will be the numerators in calculation of Lambda_f
    
    # Calculate denominators - use np.ones to avoid adding one to each denominator sum for when i = j
    Lambda_f_denoms = np.zeros((N_s, np.shape(T)[0]))
    
    # Calculate arrays of denominators
    for i in range(N_s): # i loop - numerator
        j_list = list(range(N_s))
        for j in j_list: # j loop - denominator
            i_reg = species_indices[i]
            j_reg = species_indices[j]
        
            phi_ij = (Specie._reg[j_reg].mol_mass / Specie._reg[i_reg].mol_mass)**(1/2) 
                
            # sum of phi 
            Lambda_f_denoms[i] += phi_ij * X_i[j] 

    # divide and sum up - zeros in numerator cancel out ones in denominator
    Lambda_f = sum((Lambda_i.T * X_i /Lambda_f_denoms.T).T) # [W m-1 K-1] # Thermal conductivity of gas mixture
    

    return Lambda_f



def flue_gas_viscosity(X_i, species_indices, T_C):
    """
    Calculate the (DYNAMIC) VISCOSITY OF GAS MIXTURE

    Parameters
    ----------
    X_i : Array     
        [-] Mole fraction of specie i in order (CH3OH, H2O, H2, CO2, CO)
    species_indices : array
        [-] indices of species used in this estimation (indices within Specie class, _reg)
    T_C :             
        [C] Fluid temperature 

    Returns
    -------
    mu_f :          
        [Pa s] Gas mixture viscosity 

    """
    
    N_s = len(X_i) # Number of species used in this calculation
    
    # Convert temperature in Celsius to Kelvin
    T = T_C + 273.15 
    
    # First, calculate the first order solution for local the pure gas viscosities (mu_i)
    # mu_i is calculated according to:
    # The Properties of Gases and Liquids - Book by Bruce E. Poling, John Paul O'Connell, and John Prausnitz
    # p. 9.7 - method of Chung et.al.
    mu_i =[] # empty list of local pure gas viscosities
    
    for i in range(N_s):
        i_reg = species_indices[i]
        mu_i.append( Specie._reg[i_reg].mu_i(T) ) # Calculation of each pure gas viscosity is done within the class Specie
    
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
    for i in range(N_s): # i loop - numerator
        j_list = list(range(N_s))
        denom_temp_sum = 0 # Temporary sum of denominator 
        for j in j_list: # j loop - denominator
            i_reg = species_indices[i]
            j_reg = species_indices[j]
            # numerator of phi
            k1 = ( 1 + (mu_i[i] / mu_i[j])**(1/2) * (Specie._reg[j_reg].mol_mass / Specie._reg[i_reg].mol_mass)**(1/4) )**2
            # denominator of phi
            k2 = math.sqrt(8) * (1 + (Specie._reg[i_reg].mol_mass / Specie._reg[j_reg].mol_mass))**(1/2)
            # phi
            phi_ij = k1 / k2
            # sum of phi 
            denom_temp_sum += phi_ij * X_i[j]
        denoms.append(denom_temp_sum) # Append to list of denominators
    denoms = np.array(denoms) # Convert to array in the end of loop

    mu_f = sum( (X_i*mu_i.T / denoms.T ).T)
    
    
    return mu_f



# -- Steam condensation heating


def pressure_and_latent_heat_at_temperature(temperature):
    """
    Interpolate steam pressure and latent heat of vaporization for targeted temperature point on shell side of the heat exchanger 
    
    Steam tables taken from: 
    https://pressbooks.pub/thermo/chapter/saturation-properties-temperature-table/

    Parameters
    ----------
    temperature : float
        [C] Target temperature on shell side of heat exchanger

    Returns
    -------
    pressure : float
        [Pa] Required pressure targeted condensation temperature
    heat : float
        [J kg-1] Latent heat of vaporization at specified pressure

    """
    
    # [C]
    temperature_array = np.asarray([0.01,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,373.95])
    # [bar]
    pressure_array = np.asarray([0.006117,0.008726,0.01228,0.01706,0.02339,0.0317,0.04247,0.05629,0.07385,0.09595,0.1235,0.1576,0.1995,0.2504,0.312,0.386,0.4741,0.5787,0.7018,0.8461,1.014,1.434,1.987,2.703,3.615,4.762,6.182,7.922,10.028,12.552,15.549,19.077,23.196,27.971,33.469,39.762,46.923,55.03,64.166,74.418,85.879,98.651,112.84,128.58,146.01,165.29,186.66,210.44,220.64])
    # [kJ kg-1]
    heat_array = np.asarray([2500.9,2489.1,2477.2,2465.3,2453.5,2441.7,2429.8,2417.9,2406,2394,2382,2369.8,2357.6,2345.4,2333,2320.6,2308,2295.3,2282.5,2269.5,2256.4,2229.7,2202.1,2173.7,2144.2,2113.7,2081.9,2048.8,2014.2,1977.9,1939.7,1899.7,1857.3,1812.7,1765.4,1715.1,1661.6,1604.4,1543,1476.7,1404.6,1325.7,1238.4,1140.1,1027.3,892.7,719.8,443.8,0])
    
    # Get target pressure [Pa]
    pressure = np.interp(temperature, temperature_array, pressure_array) * 1e5 
    # Get resulting latent heat of vaporization [J kg-1] 
    heat = np.interp(temperature, temperature_array, heat_array) * 1e3
    
    # Convert to [Pa] and [J kg-1] at return
    return pressure, heat




def saturated_water_k_and_mu(temperature):
    """
    Interpolate thermal conductivity, viscosity, and density for saturated water from steam tables: 
    https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118534892.app2
        

    Parameters
    ----------
    temperature : float
        [C] Saturated water temperature (at saturation pressure)

    Returns
    -------
    k_sw : float
        [W m-1 K-1] Thermal conductivity
    mu_sw : float
        [Pa s] Viscosity
    rho_sw : float
        [kg m-3] Density

    """
    # [C]
    temperature_array = np.asarray([0.01,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,373.9])
    # [mW m-1 K-1]
    k_array = np.asarray([547.5,567.4,586,602.9,617.8,630.4,640.9,649.5,656.2,661.3,665.1,667.6,669.1,669.7,669.4,668.3,666.4,663.7,660.2,655.9,650.7,644.7,637.6,629.5,620.3,609.8,598,584.5,569.4,552.3,533.1,511.3,486.8,458.8,426.9,389.7,344.4,280.7,210.8])
    # [micro Pa s]
    mu_array = np.asarray([1792,1307,1002,797.7,653.3,547.1,466.6,404,354.5,314.5,281.9,254.8,232.1,213,196.6,182.5,170.3,159.6,150.2,141.8,134.4,127.7,121.6,116,110.9,106.2,101.7,97.55,93.56,89.71,85.95,82.22,78.46,74.57,70.45,65.88,60.39,52.25,41.95])
    # [kg m-3]
    rho_array = np.asarray([1000,999.7,998.2,995.6,992.2,988,983.2,977.7,971.8,965.3,958.4,951,943.2,934.9,926.2,917.1,907.5,897.5,887.1,876.1,864.7,852.8,840.3,827.3,813.5,799.1,783.8,767.7,750.5,732.2,712.4,691,667.4,641,610.8,574.7,528.1,453.1,349.4])
    
    # Interpolate thermal conductivity and convert to [W m-1 K-1]
    k_sw = np.interp(temperature, temperature_array, k_array) * 1e-3
    # Interpolate viscosity and convert to [Pa s]
    mu_sw = np.interp(temperature, temperature_array, mu_array) * 1e-6
    # Interpolate density [kg m-3]
    rho_sw = np.interp(temperature, temperature_array, rho_array)
    
    return k_sw, mu_sw, rho_sw




def steam_horizontal_heat_transfer_coefficient(rho_condensate, k_condensate, mu_condensate, m_condensate, d_out, N_t, l_tube):
    """
    Calculate heat transfer coefficient from shell side for condensing steam - horizontal arrangement
    More info on this estimate in chapter 12 of "Kern - Process heat transfer"

    Parameters
    ----------
    rho_condensate : float
        [kg m-3] Condensate density
    k_condensate : float
        [W m-1 K-1] Condensate thermal conductivity
    mu_condensate : float
        [Pa s] Condensate viscosity
    m_condensate : float
        [kg s-1] Condensation rate 
    d_out : float
        [m] Tube external diameter
    N_t : int
        [-] Number of tubes in heat exchanger
    l_tube : float
        [m] Tube length per single pass (equal to tube length in single pass)

    Returns
    -------
    h_shell_steam : float
        [W m-2 K-1] Heat transfer coefficient

    """
    
    # Gravitational acceleration
    g = 9.8067
    
    # Horizontal arrangement of heat exchanger (Condensate drips from the tubes) 
    alpha = 1.51 
    
    # Condensate loading for horizontal tubes
    G = m_condensate / (l_tube * N_t**(2/3))
    
    # Vertical arrangement of heat exchanger (Condensate flows down the tubes, forming a thicker film at the bottom)
    # alpha = 1.47
    # M = m_steam / (np.pi * d_out * N_t)
    
    h_shell_steam = alpha * (rho_condensate**2 * g * k_condensate**3 / mu_condensate**2 )**(1/3) * (4 * G / mu_condensate)**(-1/3)
    
    return h_shell_steam
    
    
    
# -------------------------------------------------------------
# --- Wall temperature boundary condition
# -------------------------------------------------------------

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
                   'joule':'joule heating parameters',
                   'steam': 'steam heating parameters'}
    
    
    
    
    # Retreive the heating choice
    heating_choice = json.load(open(input_json_fname))['reactor heating parameters']['heating type (temperature-profile / flue-gas / joule / steam)']
    # Ready heating parameters for chosen 
    steady_heating_params = json.load(open(input_json_fname))['reactor heating parameters'][heating_dict[heating_choice.lower()]]
    dynamic_heating_params = json.load(open(input_json_fname))['dynamic boundary conditions']['dynamic heating parameters'][heating_dict[heating_choice.lower()]]

    # Define relative cell centers (from 0 to 1) that will be used to interpolate 
    rel_zcell_centers = z_cell_centers / l_tube
    
    # See whether we use dynamic boundary conditions for t wall
    dyn_BC = json.load(open(input_json_fname))['dynamic boundary conditions']
    
    # See whether we're using dynamic boundary conditions
    condition = bool(dyn_BC['use dynamic wall heating'])
    
    
    # --- Read parameters
    # These parameters/properties are used across all heating cases
    
    # Get material properties
    material_props = json.load(open(input_json_fname))['reactor parameters']
    rho_tube_h = material_props['material density [kg m-3]']
    k_tube_h = material_props['material thermal conductivity [W m-1 K-1]']
    cp_tube_h = material_props['material specific heat capacity [J kg-1 K-1]']
    
    # Geometric parameters
    d_tube_in_h = material_props['single tube inner diameter [m]'] 
    s_tube_h = material_props['single tube wall thickness [m]']
    d_shell_h = material_props['reactor shell inner diameter [m]']
    p_t_h = material_props['tube pitch [m]']
    N_t_h = material_props['total number of tubes in the reactor [-]']
    
    # Baffle parameters
    N_t_bw_h = material_props['number of tubes in baffle window [-]']
    N_bp_h = material_props['number of baffle plates [-]']
    BS_op_h = material_props['baffle window height as percentage of baffle diameter [-]']
        
    # The current formulas are derived for total wall thickness across diameter so below we use this s2
    s2_tube_h = s_tube_h*2
    # Outer diameter of reactor tube
    d_tube_out_h = d_tube_in_h + s2_tube_h
    
    # Axial cell spacing array
    n_cells_h = len(z_cell_centers) # Number of cells
    dz_h = (l_tube / n_cells_h) 
        
    
    if heating_choice == 'flue-gas' or heating_choice == 'steam': # In case of STHE, make a quick check regarding the physicality 
        if d_tube_out_h >= p_t_h: # is the pitch larger than the outer tube diameter
            raise ValueError('Non physical reactor dimensions received: tube pitch must be larger than reactor tube outer diameter!')
    
    
    # --- Calculate some geometric variables
    A_tube_out_dz_h = np.pi * d_tube_out_h *dz_h # Outer area of reactor tube segment
    A_tube_in_dz_h = np.pi * d_tube_in_h *dz_h # Inner area of reactor tube segment
    CS_tube_h = np.pi/4 * (d_tube_out_h**2 - d_tube_in_h**2) # Tube cross section 
    V_tube_dz_h = CS_tube_h * dz_h # Volume of reactor tube segment
    
    
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

        # Make a function that always returns the zero as a flux        
        def T_wall_func_steady(self, *args, **kwargs):
            return 0, 'n/a'
        
        
        # -- Dynamic boundary condition function 
        if not dyn_bc or not condition: # If we're not using dynamic boundary conditions entirely or if we're not using them for wall temp
            # If dynamic 
            def T_wall_func_dynamic(self, *args, **kwargs):
                return 0, 'n/a'
                
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
                    return 0, 'n/a'
            
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
                    
                    return Tw_profile, 'n/a'
        
        
        
    elif heating_choice == 'joule':
        
        
        # Joule heating specifics
        I_tube_h = steady_heating_params['current through single tube [A]']
        rho_e_tube_h = steady_heating_params['tube material electrical resistivity [ohm m]']
        
        
        def T_wall_func_steady(self, T_wall, C_field, T_field, u_s, *args, **kwargs): 
            """
            Calculates wall temperature from Joule heating

            Parameters
            ----------
            relax : float
                [-] Underrelaxation factor
            T_wall : 1D array
                [C] Current wall temperature profile
            C_field : 3D array
                [mol m-3] Specie concentration array
            T_field : 2D array
                [C] Temperature field array
            u_s : 1D array
                [m s-1] Superficial velocity

            Returns
            -------
            wall_T_flux : 1D array
                [C] Newly calculated wall temperature profile

            """
            
            T_near_wall = T_field[0]
            
            # First get film transfer coefficient
            ht = heat_transfer_coeff_tube(u_s, self.d_p, T_field, C_field, self.epsilon)
            
            d2_T_wall = T_wall_second_derivative(T_wall, dz_h) # Second derivative along z [K m-2]
            Q_ax = d2_T_wall * k_tube_h
            
            # # Heat radially transferred to the reactor [kg s-3 m-1]
            Q_rad = ht * (T_wall - T_near_wall) * A_tube_in_dz_h / V_tube_dz_h
            
            # # Heat generated by Joule effect [kg s-3 m-1]
            Q_gen = rho_e_tube_h * dz_h / CS_tube_h * I_tube_h**2 / V_tube_dz_h
            
            # wall_flux = (Q_ax + Q_gen - Q_rad) / (rho_tube_jh*cp_tube_jh)
            wall_flux = (Q_ax + Q_gen - Q_rad) / 100
            
            
            return wall_flux, 0, 0
        
        
        
        # -- Dynamic boundary condition function 
        if not dyn_bc or not condition: # If at least one is false, we have a steady current
            # Make a lambda function for I that always gives the same value
            I_in_func = lambda t : I_tube_h

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
            
            
        def T_wall_func_dynamic(self, t, dt, T_wall, C_field, T_field, u_s, *args, **kwargs):
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
            C_field : 3D array
                [mol m-3] Specie concentration array
            T_field : 2D array
                [C] Temperature field array
            u_s : 1D array
                [m s-1] Superficial velocity

            Returns
            -------
            wall_T_profile : 1D array
                [C] Newly calculated wall temperature profile

            """
            T_near_wall = T_field[0]
            
            # # Get current at time t
            I_tube_t = I_in_func(t)
            
            # # First get film transfer coefficient
            ht = heat_transfer_coeff_tube(u_s, self.d_p, T_field, C_field, self.epsilon)
            
            # Get heat transfer due to axial diffusion along the reactor 
            d2_T_wall = T_wall_second_derivative(T_wall, dz_h) # Second derivative along z [K m-2]
            Q_ax =  d2_T_wall * k_tube_h
            
            # Heat radially transferred to the reactor [kg s-3 m-1]
            Q_rad = ht * (T_wall - T_near_wall) * A_tube_in_dz_h / V_tube_dz_h
            
            # Heat generated by Joule effect [kg s-3 m-1]
            Q_gen = rho_e_tube_h * dz_h / CS_tube_h * I_tube_t**2 / V_tube_dz_h
            
            # Sum up heat fluxes and divide by tube density and cp to get dT_wall / dt
            wall_flux = (Q_ax - Q_rad + Q_gen) / (rho_tube_h * cp_tube_h)
            
            
            return wall_flux, 0, 0
        
        
    elif heating_choice == 'flue-gas': 
        
        # Flue gas specifics
        fgas_composition = steady_heating_params['flue gas composition (combusted-methane)']
        fgas_flowdir = steady_heating_params['flue gas flow direction (co-current / counter-current)']
        T_in_fgas = steady_heating_params['flue gas inlet temperature [C]']
        p_in_fgas = steady_heating_params['flue gas inlet pressure [bar]']
        m_fgas = steady_heating_params['flue gas mass flow [kg s-1]']

        
        l_baf = material_props['reactor tube length [m]'] / (N_bp_h+1) # Baffle spacing
        f_bw = baffle_window_area_fraction(d_shell_h, BS_op_h) # [-] Area fraction of baffle window opening
        A_bw = f_bw*(np.pi * d_shell_h**2 / 4) - N_t_bw_h*(np.pi * d_tube_out_h**2 / 4) # [m2] Area in baffle window available for flue gas flow (area of window - cross section area of all tubes in baffle window)
        A_pe = l_baf * d_shell_h * (p_t_h- d_tube_out_h)/p_t_h  # Interstitial area available for crossflow perpendicular to the bank of tubes at te widest point in the shell
       
        A_shell_CS = (np.pi/4 * d_shell_h**2) - N_t_h * d_tube_out_h**2 *np.pi/4 # Cross sectional area of shell available for fluid flow
        
        
        # --- flue gas flow direction
        if fgas_flowdir.lower() == 'co-current': # Convert the flow direction input to a +/- factor that we'll use in the equations
            fgas_flowdir = 1
        elif fgas_flowdir.lower() == 'counter-current':
            fgas_flowdir = -1
        else:
            raise ValueError('Flue gas flow direction not recognized:', fgas_flowdir)
        
        # --- Make functions for dependent flue gas variables
        if fgas_composition.lower() == 'combusted-methane':
            Mw_fgas = 0.095*CO2.mol_mass + 0.19*H2O.mol_mass + 0.715*N2.mol_mass # [g/mol] - molecular weight
            
            def Cp_fgas_mixture(T): # Define a function of flue gas mixture Cp dependant on temperature
                Cp_mixture = (0.095*CO2.Cm_i(T) + 0.19*H2O.Cm_i(T) + 0.715*N2.Cm_i(T) ) / Mw_fgas * 1000
                # [J kg-1 K-1]
                return Cp_mixture 
            #Indices and molar fractions of corresponding indices needed to calculate thermal conductivity of gas mixture
            reg_class_indices = np.asarray([1,3,5]) # 0=CH3OH, 1=H2O, 2=H2, 3=CO2, 4=CO, 5=N2
            X_i_fgas = np.asarray([0.19, 0.095, 0.715]) # Molar fraction of gases used 
            
        else: 
            raise ValueError('Flue gas composition not recognized:', fgas_composition)
        
        # --- Flue gas properties 
        # Define a small function to get density dependant on temperature (assume no pressure drop)
        def rho_fgas(T): 
            # T is in [C]
            R = 8.31446261815324 # [J K-1 mol-1] Ideal gas constant 
            rho_fgas = (p_in_fgas *1e5)/(R * (T+273.15)) * Mw_fgas*1e-3 # Ideal gas law for flue gas density
            return rho_fgas # [kg m-3]
        
        # hydraulic diameter (shell side equivalent diameter) for 60 degrees triangular pitch
        D_h = 1.1/d_tube_out_h * (p_t_h**2 - 0.917*d_tube_out_h**2)
        
        # Flue gas flow 
        Q_fgas = m_fgas / rho_fgas(T_in_fgas) # [m3 s-1] Volumetric flow of flue gas
        u_s_fgas = Q_fgas/A_pe # [m s-1] inlet superficial velocity of flue gas
        # Note - superficial velocity will  change if we stop using constant pressure at some point
        
        
        def gas_to_wall_h_s(T_fgas, u_s_fgas):
            """
            Calculate heat transfer coefficient from flue gas to tube wall inner edge according to modified Donohue equation
            Source book: Chemical Engineering Design - Principles, Practice and Economics of Pland and Process Design, 2nd edition by Gavin Towler and Ray Sinnott (0)

            Parameters
            ----------
            T_fgas : array
                [C] Flue gas temperature profile

            Returns
            -------
            h_s : array
                [W m-2 K-1] Heat transfer coefficient for the shell side fluid 

            """
            # --- First get some flue gas properties that are dependant on temperature
            # Get flue gas viscosity
            mu_fgas = flue_gas_viscosity(X_i_fgas, reg_class_indices, T_fgas)
            # Get flue gas thermal conductivity
            k_fgas = gas_mixture_thermal_conductivity(X_i_fgas, reg_class_indices, T_fgas)
            # Get flue gas thermal capacity
            Cp_fgas = Cp_fgas_mixture(T_fgas)
            
            # Reynolds number
            Re = u_s_fgas * rho_fgas(T_in_fgas) * D_h / mu_fgas
            
            # Prandtl number
            Pr = (Cp_fgas * mu_fgas / k_fgas)
            
            # Equation from Kern, page 137. They say it's valid for Re range of 2,000<Re<1,000,000 but seems to be good enough for initial results even outside of this range
            h_s = 0.36*Re**0.55*Pr**0.33 * (k_fgas / D_h) 
            # This equation is from Zhu
            # h_s = 0.2*Re**0.6*Pr**0.33 * (k_fgas / D_h)
            
            return h_s
        
        
        # --- Twall functions
        
        def T_wall_func_steady(self, T_wall, C_field, T_field, u_s, T_fgas, *args, **kwargs): 
            """
            Calculates wall temperature from flue gas heating

            Parameters
            ----------
            relax : float
                [-] Underrelaxation factor
            T_wall : 1D array
                [C] Current wall temperature profile
            C_field : 3D array
                [mol m-3] Specie concentration array
            T_field : 2D array
                [C] Temperature field array
            u_s : 1D array
                [m s-1] Superficial velocity
            T_fgas : 1D array
                [C] Flue gas temperature array

            Returns
            -------
            new_T_wall : 1D array
                [C] Newly calculated wall temperature profile
            new_T_fgas : 1D array
                [C] Newly calculated flue gas temperature profile

            """
            T_near_wall = T_field[0]
            
            # # First get film transfer coefficient
            ht = heat_transfer_coeff_tube(u_s, self.d_p, T_field, C_field, self.epsilon)
            
            d2_T_wall = T_wall_second_derivative(T_wall, dz_h) # Second derivative along z [K m-2]
            Q_ax = d2_T_wall * k_tube_h
            
            # -- Tube heating by hot flue gas
            # First get heat transfer coefficient from flue gas to wall
            h_s = gas_to_wall_h_s(T_fgas, u_s_fgas)
            # [kg s-3 m-1] Heat transfer from flue gas to the wall: heat_transfer_coeff*reactor_tube_outer_area/reactor_tube_volume
            # Calculation for only one reactor tube 
            Q_fg_w = h_s * (T_fgas - T_wall) * A_tube_out_dz_h / V_tube_dz_h  # Omitted number of tubes from the equation because this is the heat given to ONE reactor tube
            # -- Heat radially transferred from the wall to the reactor [kg s-3 m-1]
            Q_w_r = ht * (T_wall - T_near_wall) * A_tube_in_dz_h / V_tube_dz_h
            
            
            # --- New wall temperature profile
            wall_flux = (Q_ax + Q_fg_w - Q_w_r)/100 #/ rho_tube_fgas / cp_tube_fgas
            # wall_flux = (Q_ax + Q_fg_w - Q_w_r)/ rho_tube_h #/ cp_tube_fgas
            
            # --- New flue gas temperature 
            # Formulation from Varna&Morbidelli (using mass flow and 1D cells, opposed to velocity and 2D cells used in the reactor)
            Cp_fgas = Cp_fgas_mixture(T_fgas) # [J kg-1 K-1] Specific heat capacity
            d_T_fgas = flue_gas_first_derivative(T_fgas, dz_h, T_in_fgas, fgas_flowdir) # [K m-1] dT/dr evaluation
            Q_adv_fgas = Cp_fgas * m_fgas * d_T_fgas  # [W m-1] Total heat transferred by advection
            
            Q_rad_fgas = d_tube_out_h * np.pi * N_t_h * h_s * (T_fgas - T_wall) # [W m-1] Total heat transfered by convection to reactors 

            hfluid_flux = (-Q_adv_fgas - Q_rad_fgas) / Cp_fgas / rho_fgas(T_fgas) / A_pe
            
            return wall_flux, hfluid_flux, 0
        
        
        ##
        ## --- Dynamic flue gas functions 
        ## 
        # -- Dynamic boundary condition function 
        if not dyn_bc or not condition: # If at least one is false, we have a steady flue gas input
            # Make a lambda function for I that always gives the same value
            T_in_fgas_func = lambda t : T_in_fgas
            m_in_fgas_func = lambda t : m_fgas
            p_in_fgas_func = lambda t : p_in_fgas

        else: # Otherwise make an interpolation function
            # Read values from dictionary
            # --- Matrices containing times and variables
            # Flue gas temperature inlet       
            if bool(dynamic_heating_params['use dynamic temperature inlet']): # Run a check whether we're using all of these dynamic inputs
                T_in_fgas_matrix = np.asarray(dynamic_heating_params['gas inlet temperature in time (t[s], T_in[C])'])
                T_in_fgas_m = T_in_fgas_matrix[:,1] # Extract respective arrays from the matrix
                time_T_in_fgas = T_in_fgas_matrix[:,0]
                
                # Make some checks check
                if not ( sorted(time_T_in_fgas) == time_T_in_fgas).all(): # Time should be given in ascending order
                    raise ValueError('In "dynamic boundary conditions", time for flue gas T_inlet should be given in ascending order')
                    
                if len(T_in_fgas_m) == 1: # If one element in list 
                    T_in_fgas_func = lambda t : T_in_fgas_m[0] # Dont bother with interpolation, just return that value
                else: # If multiple values in list
                    def T_in_fgas_func(t): 
                        """
                        Interpolates inlet temperature of flue gas in time
            
                        Parameters
                        ----------
                        t : float
                            [s] time
            
                        Returns
                        -------
                        T_interp : float
                            [C] Inlet flue gas temperature in time 
                        """
                        T_interp = np.interp(t, time_T_in_fgas, T_in_fgas_m)
                        return T_interp
                
            else:
                T_in_fgas_func = lambda t : T_in_fgas
                
                
            # Mass flow inlet
            if bool(dynamic_heating_params['use dynamic mass flow inlet']):
                m_in_fgas_matrix = np.asarray(dynamic_heating_params['gas inlet mass flow in time (t[s], m_in[kg s-1])'])
                m_in_fgas_m = m_in_fgas_matrix[:,1]
                time_m_in_fgas = m_in_fgas_matrix[:,0]
                
                if not ( sorted(time_m_in_fgas) == time_m_in_fgas).all(): # Time should be given in ascending order
                    raise ValueError('In "dynamic boundary conditions", time flue gas mass inlet should be given in ascending order')
                    
                if len(m_in_fgas_m) == 1: # If one element in list 
                    m_in_fgas_func = lambda t : m_in_fgas_m[0] # Dont bother with interpolation, just return that value
                else: # If multiple values in list
                    def m_in_fgas_func(t): 
                        """
                        Interpolates mass flow of flue gas in time
            
                        Parameters
                        ----------
                        t : float
                            [s] time
            
                        Returns
                        -------
                        m_interp : float
                            [kg s-1] Inlet flue gas mass flow 
                        """
                        m_interp = np.interp(t, time_m_in_fgas, m_in_fgas_m)
                        return m_interp
            else:
                m_in_fgas_func = lambda t : m_fgas
                
            # pressure flue gas inlet
            if bool(dynamic_heating_params['use dynamic pressure inlet']):
                p_in_fgas_matrix = np.asarray(dynamic_heating_params['gas inlet pressure in time (t[s], p_in[bar])'])
                p_in_fgas_m = p_in_fgas_matrix[:,1]
                time_p_in_fgas = p_in_fgas_matrix[:,0]
                
                if not ( sorted(time_p_in_fgas) == time_p_in_fgas).all(): # Time should be given in ascending order
                    raise ValueError('In "dynamic boundary conditions", time flue gas mass inlet should be given in ascending order')
                    
                if len(p_in_fgas_m) == 1: # If one element in list 
                    p_in_fgas_func = lambda t : p_in_fgas_m[0] # Dont bother with interpolation, just return that value
                def p_in_fgas_func(t): 
                    """
                    Interpolates inlet pressure of flue gas in time
        
                    Parameters
                    ----------
                    t : float
                        [s] time
        
                    Returns
                    -------
                    p_interp : float
                        [bar] Inlet pressure of flue gas
                    """
                    p_interp = np.interp(t, time_p_in_fgas, p_in_fgas_m)
                    return p_interp
            else:
                p_in_fgas_func = lambda t : p_in_fgas
            
            
        ### - OTHER dynamic flue gas functions 
        
        # - Dynamic T wall specific functions
        def rho_fgas_dyn(T, p_in_fgas_dyn): 
            # T is in [C]
            R = 8.31446261815324 # [J K-1 mol-1] Ideal gas constant 
            rho_fgas = (p_in_fgas_dyn *1e5)/(R * (T+273.15)) * Mw_fgas*1e-3 # Ideal gas law for flue gas density
            return rho_fgas # [kg m-3]
        
        
        def T_wall_func_dynamic(self, t, dt, T_wall, C_field, T_field, u_s, T_fgas, *args, **kwargs):
            """
            Calculates wall temperature from flue gas heating

            Parameters
            ----------
            relax : float
                [-] Underrelaxation factor
            T_wall : 1D array
                [C] Current wall temperature profile
            C_field : 3D array
                [mol m-3] Specie concentration array
            T_field : 2D array
                [C] Temperature field array
            u_s : 1D array
                [m s-1] Superficial velocity
            T_fgas : 1D array
                [C] Flue gas temperature array

            Returns
            -------
            new_T_wall : 1D array
                [C] Newly calculated wall temperature profile
            new_T_fgas : 1D array
                [C] Newly calculated flue gas temperature profile

            """
            
            # Dynamic input flue gas values
            T_in_fgas_d = T_in_fgas_func(t)
            m_fgas_d = m_in_fgas_func(t)
            p_in_fgas_d = p_in_fgas_func(t)
            
            # --- Recalculate flue gas properties based on changing inlet flue gas values
            rho_fgas_d = rho_fgas_dyn(T_fgas, p_in_fgas_d) # Density
            Q_fgas_d = m_fgas_d / rho_fgas_d # [m3 s-1] Volumetric flow of flue gas
            u_s_fgas_d = Q_fgas_d/A_pe # [m s-1] superficial velocity of flue gas

            T_near_wall = T_field[0]
            
            # # First get film transfer coefficient
            ht = heat_transfer_coeff_tube(u_s, self.d_p, T_field, C_field, self.epsilon)
            
            d2_T_wall = T_wall_second_derivative(T_wall, dz_h) # Second derivative along z [K m-2]
            Q_ax = d2_T_wall * k_tube_h
            
            # -- Tube heating by hot flue gas
            # First get heat transfer coefficient from flue gas to wall
            h_s = gas_to_wall_h_s(T_fgas, u_s_fgas_d) 
            # [kg s-3 m-1] Heat transferred radially from flue gas to the wall
            Q_fg_w = h_s * (T_fgas - T_wall) * A_tube_out_dz_h / V_tube_dz_h  # Omitted number of tubes from the equation because this is the heat given to ONE reactor tube
            
            # -- Heat radially transferred from the wall to the reactor [kg s-3 m-1]
            Q_w_r = ht * (T_wall - T_near_wall) * A_tube_in_dz_h / V_tube_dz_h
            
            
            # --- New wall delta T 
            wall_flux = (Q_ax + Q_fg_w - Q_w_r) / (rho_tube_h * cp_tube_h)


            # --- New flue gas temperature 
            # Formulation from Varna&Morbidelli (using mass flow and 1D cells, opposed to velocity and 2D cells used in the reactor)
            Cp_fgas = Cp_fgas_mixture(T_fgas) # [J kg-1 K-1] Specific heat capacity
            d_T_fgas = flue_gas_first_derivative(T_fgas, dz_h, T_in_fgas_d, fgas_flowdir) # [K m-1] dT/dr evaluation
            Q_adv_fgas = Cp_fgas * m_fgas_d * d_T_fgas # [W m-1] Total heat transferred by advection
            
            Q_rad_fgas = d_tube_out_h * np.pi * N_t_h * h_s * (T_fgas - T_wall) # [W m-1] Total heat transfered by convection to reactors 
            
            hfluid_flux = (-Q_adv_fgas - Q_rad_fgas) / Cp_fgas / rho_fgas_d / A_pe
            
            return wall_flux, hfluid_flux, 0
        
    elif heating_choice == 'steam':
        
        
        # Steam heating specific parameters
        T_steam = steady_heating_params['condensation temperature [C]']
        
        # l_baf = material_props['reactor tube length [m]'] / (N_bp_h+1) # Baffle spacing
        # f_bw = baffle_window_area_fraction(d_shell_h, BS_op_h) # [-] Area fraction of baffle window opening
        # A_pe = l_baf * d_shell_h * (p_t_h- d_tube_out_h)/p_t_h  # Interstitial area available for crossflow perpendicular to the bank of tubes at te widest point in the shell
        
        A_steam_cs = (d_shell_h**2 * np.pi / 4) - N_t_h*(d_tube_out_h**2 * np.pi / 4) # Cross section area available for steam flow 
        V_steam_flow = A_steam_cs * dz_h # Volume available for steam flow
        
        
        
        
        # tube log mean diameter 
        d_lm = (d_tube_out_h - d_tube_in_h) / np.log(d_tube_out_h / d_tube_in_h)
        
        # tube wall heat transfer coefficient inverse
        tw_htc_i = s_tube_h / k_tube_h * d_tube_in_h / d_lm
        
        # ratio of inner vs outer tube diameter 
        ratio_dti_dto = d_tube_in_h / d_tube_out_h
        
        
        '''
        # Get material properties
        rho_tube_h = material_props['material density [kg m-3]']
        k_tube_h = material_props['material thermal conductivity [W m-1 K-1]']
        cp_tube_h = material_props['material specific heat capacity [J kg-1 K-1]']
        
        # Geometric parameters
        d_tube_in_h = material_props['single tube inner diameter [m]'] 
        s_tube_h = material_props['single tube wall thickness [m]']
        d_shell_h = material_props['reactor shell inner diameter [m]']
        p_t_h = material_props['tube pitch [m]']
        N_t_h = material_props['total number of tubes in the reactor [-]']
        
        # Baffle parameters
        N_t_bw_h = material_props['number of tubes in baffle window [-]']
        N_bp_h = material_props['number of baffle plates [-]']
        BS_op_h = material_props['baffle window height as percentage of baffle diameter [-]']
            
        # The current formulas are derived for total wall thickness across diameter so below we use this s2
        s2_tube_h = s_tube_h*2
        # Outer diameter of reactor tube
        d_tube_out_h = d_tube_in_h + s_tube_h*2
        
        # Axial cell spacing array
        n_cells_h = len(z_cell_centers) # Number of cells
        dz_h = (l_tube / n_cells_h) 
            
        '''


        # -- (saturated) steam properties
        # Pressure and latent heat of evaporation
        p_steam_ss, H_ss = pressure_and_latent_heat_at_temperature(T_steam)

        # -- (saturated) liquid/condensate properties 
        # thermal conductivity, viscosity, and density
        k_l_ss, mu_l_ss, rho_l_ss = saturated_water_k_and_mu(T_steam)
        
        
        def T_wall_func_steady(self, T_wall, C_field, T_field, u_s, T_fgas, m_condensed, *args, **kwargs): 
            """
            Calculates wall temperature from steam heating

            Parameters
            ----------
            relax : float
                [-] Underrelaxation factor
            T_wall : 1D array
                [C] Current wall temperature profile
            C_field : 3D array
                [mol m-3] Specie concentration array
            T_field : 2D array
                [C] Temperature field array
            u_s : 1D array
                [m s-1] Superficial velocity
            T_fgas : 1D array
                [C] Flue gas temperature array

            Returns
            -------
            wall_flux : 1D array
                [C] Temperature flux for reactor tube wall
            hfluid_flux : 1D array
                [C] Temperature flux for heating fluid 

            """
            T_near_wall = T_field[0] # Take only near wall temperature
            
            # First get film transfer coefficient for tube side
            ht = heat_transfer_coeff_tube(u_s, self.d_p, T_field, C_field, self.epsilon)
            
            #  Heat radially transferred to the reactor from the tube wall [kg s-3 m-1]
            Q_w_r =  ht * (T_wall - T_near_wall) * A_tube_in_dz_h / V_tube_dz_h
            
            # Axial heat transfer in the tube 
            d2_T_wall = T_wall_second_derivative(T_wall, dz_h) # Second derivative along z [K m-2]
            Q_ax = d2_T_wall * k_tube_h # [W m-3]
            
            # heat transfer coefficient on shell side
            # hs_steam = steam_horizontal_heat_transfer_coefficient(rho_l_ss, k_l_ss, mu_l_ss, 0.00092, d_tube_out_h, N_t_h, l_tube) # [W m-2 K-1]
            hs_steam = steam_horizontal_heat_transfer_coefficient(rho_l_ss, k_l_ss, mu_l_ss, sum(m_condensed), d_tube_out_h, N_t_h, l_tube) # [W m-2 K-1]

            # Total shell side heat transfer coefficient (combined shell film coefficient and heat transfer through tube coefficient)
            Uw_shell = 1 / (tw_htc_i + ratio_dti_dto/hs_steam)
            
            # Heat transfer rate from steam to tube wall
            Q_steam = Uw_shell * (T_steam - T_wall) * A_tube_out_dz_h # [W] (array)
            
            Q_steam[Q_steam < 0] = 0 # Assumption: heat is always transferred from steam to tube, never the other way around
            
            # Q_steam is divided by reactor tube volume since everything else is also per unit volume in wall heat flux
            Q_steam_wall = Q_steam / V_tube_dz_h # [W m-3]
            
            # - Total wall heat flux
            wall_flux = (Q_ax + Q_steam_wall - Q_w_r) / rho_tube_h #/ cp_tube_h
            
            # --- Steam - mass and temperature fluxes
            # - Mass
            # Calculate amount of condensed water based on exchanged heat
            m_condensed_new = Q_steam *N_t_h / H_ss  # [kg s-1] (array)
            
            # Total rate of condensation across the entire reactor 
            m_condensed_flux = (m_condensed_new - m_condensed) # [kg s-2]
            
            return wall_flux, 0, m_condensed_flux
        
        
        ##
        ## --- Dynamic flue gas functions 
        ## 
        # -- Dynamic boundary condition function 
        if not dyn_bc or not condition: # If at least one is false, we have a steady flue gas input
            # Make a lambda function for I that always gives the same value
            T_in_steam_func = lambda t : T_steam

        else: # Otherwise make an interpolation function
            # Read values from dictionary
            # --- Matrices containing times and variables
            # Flue gas temperature inlet       
            if bool(dynamic_heating_params['use dynamic condensation temperature']): # Run a check whether we're using all of these dynamic inputs
                T_steam_matrix = np.asarray(dynamic_heating_params['condensation temperature in time (t[s], T[C])'])
                T_steam_array = T_steam_matrix[:,1] # Extract respective arrays from the matrix
                time_T_steam_array = T_steam_matrix[:,0]
                
                # Make some checks check
                if not ( sorted(time_T_steam_array) == time_T_steam_array).all(): # Time should be given in ascending order
                    raise ValueError('In "dynamic boundary conditions", time for steam condensation temperature should be given in ascending order')
                    
                if len(T_steam_array) == 1: # If one element in list 
                    T_in_steam_func = lambda t : T_steam_array[0] # Dont bother with interpolation, just return that value
                else: # If multiple values in list
                    def T_in_steam_func(t): 
                        """
                        Interpolates condensation temperature of steam in time
            
                        Parameters
                        ----------
                        t : float
                            [s] time
            
                        Returns
                        -------
                        T_steam_interp : float
                            [C] Inlet flue gas temperature in time 
                        """
                        T_steam_interp = np.interp(t, time_T_steam_array, T_steam_array)
                        return T_steam_interp
                
            else:
                T_in_steam_func = lambda t : T_steam
                
            
        
        
        def T_wall_func_dynamic(self, t, dt, T_wall, C_field, T_field, u_s, T_fgas, m_condensed, *args, **kwargs):
            """
            Calculates wall temperature from steam heating

            Parameters
            ----------
            relax : float
                [-] Underrelaxation factor
            T_wall : 1D array
                [C] Current wall temperature profile
            C_field : 3D array
                [mol m-3] Specie concentration array
            T_field : 2D array
                [C] Temperature field array
            u_s : 1D array
                [m s-1] Superficial velocity
            T_fgas : 1D array
                [C] Flue gas temperature array
            T_fgas : 1D array
                [C] Flue gas temperature array
            
            Returns
            -------
            wall_flux : 1D array
                [C] Temperature flux for reactor tube wall
            hfluid_flux : 1D array
                [C] Temperature flux for heating fluid 

            """
            # -- Dynamic steam temperature input and steam/condensate properties
            T_steam_d = T_in_steam_func(t)
            # Pressure and latent heat of evaporation
            p_steam_d, H_d = pressure_and_latent_heat_at_temperature(T_steam_d)
            # thermal conductivity, viscosity, and density
            k_l_d, mu_l_d, rho_l_d = saturated_water_k_and_mu(T_steam_d)
            
            T_near_wall = T_field[0] # Take only near wall temperature
            
            # First get film transfer coefficient for tube side
            ht = heat_transfer_coeff_tube(u_s, self.d_p, T_field, C_field, self.epsilon)
            
            #  Heat radially transferred to the reactor [kg s-3 m-1]
            Q_w_r =  ht * (T_wall - T_near_wall) * A_tube_in_dz_h / V_tube_dz_h
            
            # Axial heat transfer in the tube 
            d2_T_wall = T_wall_second_derivative(T_wall, dz_h) # Second derivative along z [K m-2]
            Q_ax = d2_T_wall * k_tube_h # [W m-3]
            
            # heat transfer coefficient 
            hs_steam = steam_horizontal_heat_transfer_coefficient(rho_l_d, k_l_d, mu_l_d, m_condensed, d_tube_out_h, N_t_h, l_tube) # [W m-2 K-1]
            # Total shell side heat transfer coefficient (combined shell film coefficient and heat transfer through tube coefficient)
            Uw_shell = 1 / (tw_htc_i + ratio_dti_dto/hs_steam)
            
           
            # Heat transfer rate from steam to tube wall
            Q_steam = Uw_shell * (T_steam_d - T_wall) * A_tube_out_dz_h # [W] (array)
            Q_steam[Q_steam < 0] = 0 # Assumption: heat is always transferred from steam to tube, never the other way around
            
            # Q_steam is divided by reactor tube volume since everything else is also per unit volume in wall heat flux
            Q_steam_wall = Q_steam / V_tube_dz_h # [W m-3]
            
            # --- Total wall heat flux
            wall_flux = (Q_ax + Q_steam_wall - Q_w_r)/ rho_tube_h #/ cp_tube_fgas
            
            # --- Steam - mass and temperature fluxes
            # - Mass
            # Calculate amount of condensed water based on exchanged heat
            m_condensed_new = Q_steam * N_t_h / H_d # [kg s-1] (array)
            # Total rate of condensation across the entire reactor 
            # m_condensed_total = sum(m_condensed_new)  # [kg s-1] (float)

            m_condensed_flux = m_condensed_new - m_condensed # [kg s-2]
            
            return wall_flux, 0, m_condensed_flux
        
        
        
        
        
        pass
    
    else: # Catch misspellings 
        raise ValueError('Reactor heating choice not recognized: ', heating_choice) 
        
        
        
    class Twall(): # Create an empty class that will contain methods for calculating Twall
        pass 
    
    # Add defined steady and dynamic functions as methods to Twall class
    Twall.steady = T_wall_func_steady
    Twall.dynamic = T_wall_func_dynamic
    
    # Define a class instance within global variables module
    gv.Twall_func = Twall()
    
    # --- Define additional variables and functions in class for output file writing
    
    # Joule heating 
    if heating_choice == 'joule':
        def I_func(self, t):
            I = I_in_func(t)
            return I
        Twall.I_func = I_func
    else: 
        Twall.I_func = lambda self, t : 'n/a'
        
    # Flue gas
    if heating_choice == 'flue-gas':
        def T_fgas_func(self, t):
            T = T_in_fgas_func(t)
            return T
        
        def m_fgas_func(self, t):
            m = m_in_fgas_func(t)
            return m

        def p_fgas_func(self, t):
            p = p_in_fgas_func(t)
            return p
        
        Twall.T_in_fgas_func = T_fgas_func
        Twall.m_in_fgas_func = m_fgas_func
        Twall.p_in_fgas_func = p_fgas_func
        
        # Steady values 
        Twall.T_in_fgas_steady = T_in_fgas 
        Twall.m_in_fgas_steady = m_fgas
        Twall.p_in_fgas_steady = p_in_fgas 
        
    else: 
        Twall.T_in_fgas_func = lambda self, t: 'n/a'
        Twall.m_in_fgas_func = lambda self, t: 'n/a'
        Twall.p_in_fgas_func = lambda self, t: 'n/a'
        
        Twall.T_in_fgas_steady = 'n/a'
        Twall.m_in_fgas_steady = 'n/a'
        Twall.p_in_fgas_steady = 'n/a' 
        
    if heating_choice == 'steam':
        def T_steam_func(self, t):
            T = T_in_steam_func(t)
            # p, H = pressure_and_latent_heat_at_temperature(T)
            return T
        
        def p_steam_func(self, t):
            T = T_in_steam_func(t)
            p, H = pressure_and_latent_heat_at_temperature(T)
            return p
        
        Twall.T_steam_func = T_steam_func
        Twall.p_steam_func = p_steam_func
        
        # Steady values 
        Twall.T_steam_steady = T_steam 
        Twall.p_steam_steady = p_steam_ss
        
    else: 
        Twall.T_steam_func = lambda self, t: 'n/a'
        Twall.p_steam_func = lambda self, t: 'n/a'
        
        Twall.T_steam_steady = 'n/a' 
        Twall.p_steam_steady = 'n/a'
        
        return heating_choice










