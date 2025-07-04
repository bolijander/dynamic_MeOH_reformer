# -*- coding: utf-8 -*-
"""
Main file 
Methanol steam reforming model
plug flow packed bed reactor 

Author:         Bojan Grenko
Institution:    Delft University of Technology 

This solver was written for the purpose of completing my PhD thesis. Permissions for code usage are detailed in the accompanying licence file. 
The present code builds upon the work and coding performed in PhD thesis of Dr. Jimin Zhu ( https://doi.org/10.54337/aau513444640 )

The present work is part of MENENS project ( https://menens.nl/ )
"""

import numpy as np
# import math

# import os
import sys
import copy

import time
from datetime import datetime

# My files
import model_funcs as mf
import io_funcs as io
import global_vars as gv


# -------------------------------------------------
# ------ SCRIPT PRE-RUN VARIABLES
# -------------------------------------------------

try: 
    input_json_fname = str(sys.argv[1])
    if not input_json_fname.endswith('.json'):
        input_json_fname = input_json_fname + '.json'
except:
    input_json_fname = 'simulation_setup.json'
    


runtime_start_exe = time.time() # exe time measure
start_time = datetime.now()
timestamp = start_time.strftime("%y-%d-%m--%H-%M") # get timestamp for directory names
original_stdout = sys.stdout # save original stdout


# -------------------------------------------------
# ------ INPUTS 
# -------------------------------------------------

# --- Read all inputs from .json file
cat_shape, cat_dimensions, cat_BET_area, known_cat_density, rho_cat, rho_cat_bulk, cat_pore_d, cat_composition, \
    n_tubes, l_tube, d_tube_in, s_tube, rho_tube, h_tube, cp_tube, T_fluid_in,\
    cells_ax, cells_rad, adv_scheme, diff_scheme, CFL, pi_limit, Ci_limit,\
    SC_ratio, p_ref_n, p_set_pos, T_in_n, init_T_field, flowrate_in_n,\
    sim_type, max_iter, convergence, field_relax_start, dt, t_dur, dyn_bc, cont_sim, path_old_sim_data,\
    results_dir_name, saving_only_terminal, steady_save_every, dyn_save_every, dynsim_converge_first, dynsim_cont_converg,\
    steady_write_every, dyn_write_every, keep_last_jsons, saving_timestamp, saving_json_out, saving_log, saving_files_in = io.simulation_inputs(input_json_fname)
       
    
'''   
# --- Catalyst parameters
# cat_shape -   [] Shape of catalyst particle
# cat_dimensions
# cat_BET_area -    [m2 kg-1] BET surface area    

# known_cat_density - (density / bulk density / both) - option: which catalyst density is known?
# rho_cat -     [kg m-3] Catalyst material density
# rho_cat_bulk - [kg m-3] Catalyst bulk (shipping) density 
# cat_pore_d - [m] Average catalyst pore diameter
# cat_composition - [-] Catalyst composition - weight percentages of CuO and Al2O3

# --- Reactor parameters
# n_tubes   [-] Total number of tubes in the reactor
# l_tube    [m] single tube length
# d_tube_in    [m] single tube inner diameter
# s_tube    [m] reactor tube thickness
# rho_tube  [kg m-3] tube material density
# h_tube    [W m-1 K-1] tube material thermal conductivity
# cp_tube   [J kg-1 K-1] tube material specific heat capacity

#  --- Reactor heating parameters
# T_fluid_in  [C] heating fluid inlet temperature

# --- Numerical/discretization parameters 
# cells_ax  [-] number of axial cells
# cells_rad [-] number of radial cells
# adv_scheme [] Advection discretization scheme
# diff_scheme [] Diffusion discretization scheme
# CFL       [-] CFL constraint
# pi_limit  [Pa] partial pressure limit for conversion model

# --- Initial operation parameters
# SC_ratio  [-] steam to carbon molar ratio (water to methanol molar ratio)
# p_ref_n    [bar] Reactor given pressure
# p_set_pos  [str] Position at which reactor pressure is given (inlet / outlet)
# T_in_n   [C] inlet flow temperature
# init_T_field  [C] Initial temperature in the reactor
# flowrate_in_n   - inlet flow rate, unit defined in the input file

# --- Simulation parameters
# sim_type              [-] simulation type (steady / dynamic)
# max_iter              [-] Maximum allowed iterations in steady simulation
# convergence           [-] Convergence criteria in steady simulation
# field_relax_start     [-] Relaxation factor in steady simulation
# t_dur                 [s] Unsteady sim. duration time 
# dt                    [s] Unsteady sim. timestep size
# dyn_bc                [-] Unsteady sim. dynamic boundary conditions
# cont_sim              [-] Continue simulation from file?

# --- 'logistics' parameters
# saving_dir_name        [-] Name of results/output directory name
# saving_only_terminal   [-] Show results only in terminal (i.e. don't save them) yes/no
# save_every             [-] Saving .json file frequency
# write_every            [-] Writing to terminal frequency
# keep_last_jsons        [-] Save how many last .json files for steady state solver
# saving_timestamp       [-] Use timestamp for results directory name yes/no
# saving_json_out        [-] Save .json file of simulation results yes/no
# saving_log             [-] record log yes/no
# saving_files_in        [-] copy input .json file yes/no
'''


R = 8.31446261815324 # Universal gas constant [J K-1 mol-1] 
# uniform_dz = True  # Parameter to check whether uniform axial mesh with constant dz is used

# -------------------------------------------------
# ------ Preparing directories and files
# -------------------------------------------------

# Make a directory for our results if needed
if not saving_only_terminal:
    path_subdir_results, path_sim_data = io.create_result_dir(results_dir_name, cont_sim, sim_type, saving_timestamp, timestamp)
    
    if saving_files_in: # copy the original json and dynamic BCs if specified
        io.copy_json(input_json_fname, path_subdir_results)
    
else:
    path_subdir_results, path_sim_data = '', ''


# Get a function that either writes the output json file, or does nothing depending on our choice
if sim_type == 'steady':
    write_out_json_steady, write_info_json_steady = io.output_write_settings(saving_json_out)
    write_out_json_dynamic, write_info_json_dynamic = io.output_write_settings(False)
elif sim_type == 'dynamic':
    write_out_json_steady, write_info_json_steady = io.output_write_settings(False)
    write_out_json_dynamic, write_info_json_dynamic = io.output_write_settings(saving_json_out)


# Create a simulation log and if needed 
logname =  'log.txt'
sysexit = io.open_log(saving_log, path_subdir_results, logname, original_stdout)

colw = 50 # Character column width for printing statements
print('Script started on:')
print('dd/mm/yyyy\thh:mm:ss')
print(datetime.now().strftime("%d/%m/%Y\t%H:%M:%S"))

print('\nSimulation setup file:', input_json_fname)

# -------------------------------------------------
# ------ INITIALIZATION
# -------------------------------------------------
# Convert pressure from bar to Pascal
p_ref_n = p_ref_n * 1e5 # Convert to Pascal


# -- Define global unchanging variables - geometric and mesh variables
# This is a function because its called at different times depending on whether we continue the simulation from file or make a new one
def set_globals():
    
    # -- Mesh variables
    gv.dz = cell_dz[0] # delta z of a mesh [float]
    gv.cell_dz = cell_dz
    gv.cell_dr = cell_dr
    gv.mesh_volumes = cell_V # Cell volumes 2D array
    gv.cell_center_z_positions = cell_z_centers # Cell center positions along Z axis
    gv.cell_center_r_positions = cell_r_centers # Cell center positions along r axis

    # Ratio of wall cell outer are to wall cell volume [m-1]
    gv.A_V_ratio_wall_cell = (np.pi * r_tube*2) / (np.pi/4 * ( (r_tube*2)**2 - ( (r_tube-cell_dr[0])*2 )**2 ) )
    
    # number of cells
    gv.rad_n_cells = cells_rad
    gv.ax_n_cells = cells_ax
    # Meshgrid formats of mesh  
    # First get required meshgrids
    dz_mgrid, dr_mgrid = np.meshgrid(cell_dz, cell_dr) 
    z_centers_mgrid, r_centers_mgrid = np.meshgrid(cell_z_centers, cell_r_centers)
    gv.dz_mgrid = dz_mgrid
    gv.dr_mgrid = dr_mgrid
    gv.z_centers_mgrid = z_centers_mgrid
    gv.r_centers_mgrid = r_centers_mgrid
    
    gv.cell_face_z_positions = np.append(0, cell_z_faces_R) # Cell face Z positions array
    gv.cell_V_1D = r_tube**2 *np.pi * cell_dz # 1D cell sizes
    
    # -- Reactor variables
    gv.epsilon = epsilon # Porosity
    gv.r_tube = r_tube # Radius of one reactor tube 
    gv.d_tube = r_tube*2 # Internal diameter of one reactor tube
    gv.l_tube = l_tube # Length of one reactor tube
    gv.A_tube_CS = r_tube**2 * np.pi # Internal cross sectional area of one tube 
    gv.N_aspect_ratio = N # Aspect ratio of catalyst particle diameter and reactor tube diameter
    
    # -- Catalyst properties
    gv.d_p = d_cat_part_v # volume equivalent catalyst particle diameter
    gv.catalyst_rho_bulk = rho_cat_bulk # Bulk / bed catalyst density
    gv.catalyst_cp = cat_cp # Catalyst thermal capacity
    gv.catalyst_shape = cat_shape # Catalyst shape [string]
    gv.cat_pore_d = cat_pore_d # Catalyst pore size
    
    # -- Numerical properties / variables
    # Set the amount of ghost cells on each domain side based on chosen discretization schemes
    adv_index, diff_index = mf.discretization_schemes_and_ghost_cells(adv_scheme, diff_scheme)
    gv.advection_scheme_index  = adv_index # numerical scheme index
    gv.diffusion_scheme_index  = diff_index # numerical scheme index
    gv.pressure_low_limit = pi_limit # Low limit of pressure to avoid division by zero in reaction rate calculation
    
    return



# --- Read data for simulation continuation
if cont_sim:
    
    # --- Read parameters and fields from .json file
    t_abs, field_Ci_n, field_T_n, field_p, field_v, field_BET_cat, T_wall_n, T_hfluid_n, m_condensed_n,\
        SC_ratio, n_tubes, l_tube, d_tube_in, s_tube, rho_tube, h_tube, cp_tube, \
        N, epsilon, cells_ax_old, cells_rad_old,\
        cat_shape, cat_dimensions, rho_cat, rho_cat_bulk, cat_composition, cat_cp, cat_BET_area = io.read_cont_sim_data(path_old_sim_data)
        
    # --- Calculate catalyst particle (equivalent) volume and diameter 
    V_cat_part, d_cat_part_s, d_cat_part_v = mf.catalyst_dimensions(cat_shape, cat_dimensions)
        
    # --- Reactor parameters
    r_tube = d_tube_in/2   # [m] Reactor tube radius
    
    V_tube = np.pi * r_tube**2 * l_tube # [m3] internal volume of reactor tube
    W_cat = rho_cat_bulk * V_tube # [kg] Weight of catalyst in one reactor tube
    gv.inletClass.W_cat = W_cat # Set weight inside one tube in the class

    # --- Geometry discretization
    
    if cells_ax == cells_ax_old and cells_rad == cells_rad_old: 
        # If mesh in new and old simulation is the same, just make an uniform mesh
        cell_dz, cell_dr, \
            cell_z_faces_L, cell_z_faces_R, cell_z_centers, \
            cell_r_faces_IN, cell_r_faces_EX, cell_r_centers, \
            cell_z_A, cell_r_A_IN, cell_r_A_EX, \
            cell_V = mf.uniform_mesh(cells_ax, cells_rad, l_tube, r_tube)
    else:
        # Otherwise, we interpolate fields
        
        cell_dz, cell_dr, \
            cell_z_faces_L, cell_z_faces_R, cell_z_centers, \
            cell_r_faces_IN, cell_r_faces_EX, cell_r_centers, \
            cell_z_A, cell_r_A_IN, cell_r_A_EX, \
            cell_V, \
            field_Ci_n, field_T_n, field_BET_cat, \
            T_wall_n, T_hfluid_n, field_p, m_condensed_n = mf.interpolate_to_new_mesh(l_tube, r_tube, 
                                                                cells_ax_old, cells_rad_old, cells_ax, cells_rad,
                                                                field_Ci_n, field_T_n, field_BET_cat, T_wall_n, T_hfluid_n, field_p, m_condensed_n)
            
    # Get inlet molar flow rates and molar fractions            
    ni_in_n, Xi_in_n = gv.inlet.CX(flowrate_in_n)
    
    set_globals() # define global variables
    
    # --- Get and prepare inlet/outlet values
    v_in_n, v_out_n, field_v, \
        p_in_n, p_out_n, field_p, \
        Q_in_n, Q_out_n, C_in_n = mf.get_IO_velocity_and_pressure(p_ref_n, field_T_n, field_Ci_n, field_v, field_p, ni_in_n, T_in_n, Xi_in_n)

elif not cont_sim:

    # --- Catalyst parameters
    # Calculate catalyst particle (equivalent) volume and diameter 
    V_cat_part, d_cat_part_s, d_cat_part_v = mf.catalyst_dimensions(cat_shape, cat_dimensions) 
        
    # Aspect ratio - size ratio of reactor tube to catalyst particle 
    N = mf.aspect_ratio(d_tube_in, d_cat_part_v)
    
    # Reactor (packed bed) porosity
    epsilon = mf.porosity(d_tube_in, cat_dimensions, cat_shape)
    
    # Catalyst material and shipping densities - if one of them is unknown
    rho_cat, rho_cat_bulk = mf.catalyst_densities(rho_cat, rho_cat_bulk, epsilon, known_cat_density)
    
    # Catalyst material specific heat capacity
    cat_cp = mf.catalyst_cp(cat_composition)
    
    # --- Reactor parameters
    r_tube = d_tube_in/2   # [m] Reactor tube radius
    
    V_tube = np.pi * r_tube**2 * l_tube # [m3] internal volume of reactor tube
    W_cat = rho_cat_bulk * V_tube # [kg] Weight of catalyst in one 
    gv.inletClass.W_cat = W_cat # Set weight inside one tube in the class
    
    # --- Geometry discretization
    cell_dz, cell_dr, \
        cell_z_faces_L, cell_z_faces_R, cell_z_centers, \
        cell_r_faces_IN, cell_r_faces_EX, cell_r_centers, \
        cell_z_A, cell_r_A_IN, cell_r_A_EX, \
        cell_V = mf.uniform_mesh(cells_ax, cells_rad, l_tube, r_tube)
    
    # --- Operation parameters
    
    # --- Prepare some variables and do field initialization
    # Get inlet molar flow rates and molar fractions            
    ni_in_n, Xi_in_n = gv.inlet.CX(flowrate_in_n)
    
    # -- Make some field initial guesses
    
    # Guess inlet molar concentrations 
    C_in_n = mf.X_to_concentration(Xi_in_n, p_ref_n, T_in_n)
    
    # Guess the pressure field, p inlet and outlet
    p_in_n = p_ref_n
    p_out_n = p_ref_n
    field_p = np.ones(cells_ax) * p_ref_n
    
    # Guess the initial velocity field 
    Q_in_n, v_in_n = mf.velocity_and_flowrate(ni_in_n, p_ref_n, T_in_n, r_tube)
    field_v = np.ones(cells_ax) * v_in_n
    v_out_n = copy.deepcopy(v_in_n)
    
    # Make temperature field - initial guess
    field_T_n = np.ones((cells_rad, cells_ax)) * init_T_field
    
    # Take initial wall temperature guess to be the same as outer cell in T field
    T_wall_n = field_T_n[0,:]
    T_hfluid_n = np.ones(cells_ax)*T_fluid_in
    # A guessed array of condensed steam on the tube for condensing steam case
    m_condensed_n = np.ones(cells_ax) *1e-4
    
    
    # Make concentration array
    field_Ci_n = np.zeros((6, cells_rad, cells_ax)) # Empty array
    # Assume reactor is filled with inlet mixture at t=0
    field_Ci_n[0, :,:] = C_in_n[0] # CH3OH
    field_Ci_n[1, :,:] = C_in_n[1] # H2O
    field_Ci_n[2, :,:] = C_in_n[2] # H2
    field_Ci_n[3, :,:] = C_in_n[3] # CO2
    field_Ci_n[4, :,:] = C_in_n[4] # CO
    field_Ci_n[5, :,:] = C_in_n[5] # N2

    # define global variables
    set_globals() 
    
    # --- Get and prepare inlet/outlet values
    v_in_n, v_out_n, field_v, \
        p_in_n, p_out_n, field_p, \
        Q_in_n, Q_out_n, C_in_n = mf.get_IO_velocity_and_pressure(p_ref_n, field_T_n, field_Ci_n, field_v, field_p, ni_in_n, T_in_n, Xi_in_n) 

    # Catalyst surface area array
    field_BET_cat = np.ones((cells_rad, cells_ax)) * cat_BET_area
    
    # Simulation time is always zero in new simulation
    t_abs = 0.0

# Effective radial diffusion coefficient
field_D_er = mf.radial_diffusion_coefficient(field_v, d_cat_part_v, d_tube_in)

# Define simulation absolute end time 
t_end_abs = t_dur + t_abs
# Relative time
t_rel = 0.

# Read dynamic simulation boundary conditions and make a function from it 
if sim_type == 'dynamic':
    T_in_func, flowrate_in_func, p_ref_func = io.read_and_set_dynamic_BCs(input_json_fname, dyn_bc, T_in_n, flowrate_in_n, p_ref_n)
    # We do this now because then we have BCs in the script and we don't need the input .json anymore


# Set wall heating choice function in globals
wall_heating = mf.read_and_set_T_wall_BC(input_json_fname, dyn_bc, cell_z_centers, l_tube)
# Set effectiveness factor choice and read what is it
effectiveness_factor = mf.read_and_set_effectiveness_factor(input_json_fname)

# Global variables in wall heating functions
gv.Twall_func.d_p = d_cat_part_v
gv.Twall_func.epsilon = epsilon




print('\n=======================================')
print('******* SIMULATION RUNDOWN')
print('=======================================\n')


print('# --- Simulation type:'.ljust(colw) +'{0}'.format(sim_type))
print('Mesh (radial cells x axial cells):'.ljust(colw) +'({0} x {1})'.format(cells_rad, cells_ax))

if adv_scheme == 'upwind_1o':
    print('Advection (z) discretization scheme:'.ljust(colw) +'{0}'.format('Upwind 1st order'))
elif adv_scheme == 'upwind_2o':
    print('Advection (z) discretization scheme:'.ljust(colw) +'{0}'.format('Upwind 2nd order'))
    
if diff_scheme == 'central_2o':
    print('Diffusion (r) discretization scheme:'.ljust(colw) +'{0}'.format('Central differencing 2nd order'))
elif diff_scheme == 'central_4o':
    print('Diffusion (r) discretization scheme:'.ljust(colw) +'{0}'.format('Central differencing 4th order'))
    
if sim_type == 'steady':
    print('\n# --- Steady simulation parameters:')
    print('Maximum iterations:'.ljust(colw) +'{0}'.format(max_iter))
    print('Convergence criteria:'.ljust(colw) +'{0:.2e}'.format(convergence))
    print('Starting underrelaxation factor:'.ljust(colw) +'{0:.2e}'.format(field_relax_start))
elif sim_type == 'dynamic':
    print('\n# --- Dynamic simulation parameters:')
    print('Simulated time duration:'.ljust(colw) +'{0}'.format(t_dur))
    print('Simulation timestep size [s]:'.ljust(colw) +'{0:.2e}\n'.format(dt))
    
    if dyn_bc:
        print('# --- Dynamic boundary conditions used in simulation')
        io.check_used_dynamic_BCs(input_json_fname, p_set_pos, colw)
    
if cont_sim:
    print('\n# --- Simulation continued from:'.ljust(colw) + 'results{0}'.format(path_old_sim_data.split('results')[-1].split('sim_data')[0] ))
    if not saving_only_terminal:
        print('Current simulation data path:'.ljust(colw) + 'results{0}'.format(path_subdir_results.split('results')[-1]))
    
    print('Continuing from simulation time:'.ljust(colw) +'{0:.5f} [s]'.format(t_abs))
    print('Simulation absolute end time:'.ljust(colw) +'{0:.5f} [s]'.format(t_end_abs))
 
 
print('\n\n=====================================================')
print('=== Parameter Initialization:')
print('=====================================================\n')

print('# --- Calculated catalyst parameters for %s catalyst' %(cat_shape.upper())) 
print('-----------------------------------------------------\n')

# print('Cat. particle (equivalent) diameter:'.ljust(colw) +'{0:.5f} [m]'.format(d_cat_part_v))
# print('Cat. particle (equivalent) volume:'.ljust(colw) + '{0:.3e} [m3]\n'.format(V_cat_part))
print('Packed bed aspect ratio (d_tube_in / d_cat):'.ljust(colw) +'{0:.2f} [-]'.format(N))
print('Reactor porosity:'.ljust(colw) +'{0:.5f} [-]\n'.format(epsilon))
print('Catalyst material density:'.ljust(colw) +'{0:.2f} [kg m-3]'.format(rho_cat))
print('Catalyst bulk density:'.ljust(colw) +'{0:.2f} [kg m-3]'.format(rho_cat_bulk))
# print('Catalyst specific heat capacity:'.ljust(colw) +'{0:.2f} [J kg-1 K-1]\n'.format(cat_cp))
print('Reactor (single) tube volume:'.ljust(colw) +'{0:.3e} [m3]'.format(V_tube))
print('Cat. weight in reactor (single) tube:'.ljust(colw) +'{0:.5f} [kg]\n'.format(W_cat))

print('\n# --- Calculated operation parameters') 
print('-----------------------------------------------------\n')

# print('Init. inlet flow rate (single tube):'.ljust(colw) +'{0:.3f} [user defined]'.format(flowrate_in_n))
# print('Init. inlet vol. flow rate (single tube):'.ljust(colw) +'{0:.3e} [m3 s-1]'.format(Q_in_n))
print('Init. inlet flow (superficial) velocity:'.ljust(colw) +'{0:.5f} [m s-1]\n'.format(v_in_n))
# print('Init. inlet mixture density:'.ljust(colw) +'{0:.3e} [kg m-3]'.format(rho_in_n))
# print('Init. inlet mixture (dynamic) viscosity:'.ljust(colw) +'{0:.5e} [Pa s]\n'.format(mu_in_n))
# print('Init. reactor pressure drop:'.ljust(colw) +'{0:.5f} [bar]\n'.format( (p_in_n - p_out_n)/1e5  ))

print( ('Initial given pressure at {0}:'.format(p_set_pos.upper())).ljust(colw) + '{0:.2f} [bar]'.format(p_ref_n/1e5)) 
# print('Init. inlet radial diffusion coeff.:'.ljust(colw) +'{0:.3e} [m2 s-1]\n'.format(field_D_er[0]))


# -------------------------------------------------
# ------ SCRIPT INTERNAL PARAMETERS / NAMES 
# -------------------------------------------------


# Set counters 
counter_terminal = 0
counter_save = 0

# --- Simulation setup file write - one time writing
# List of name keys for file writing
names = ['simulation type', 'simulated time', 'S/C', 'wall heating type' ,'n tubes', 'tube l', 'tube r in', 'tube s', 'tube rho', 'tube h', 'tube cp',\
         'tube V' ,'aspect ratio', 'porosity',\
         'n ax cells', 'n rad cells', 'dz', 'dy', 'cell z centers', 'cell r centers', 'cell volumes',\
         'cat shape', 'cat dimensions', 'cat particle d', 'fresh cat BET', 'cat composition', 'cat cp', 'cat rho', 'cat rho bulk', 'cat tube weight']

if sim_type == 'steady' : t_dur = 'n/a' # simulation duration not applicable if we have steady simulation
# List of fields and values for .json file writing
values = [sim_type, t_dur, SC_ratio, wall_heating, n_tubes, l_tube, r_tube, s_tube, rho_tube, h_tube, cp_tube, \
          V_tube, N, epsilon,\
          cells_ax, cells_rad, cell_dz, cell_dr, cell_z_centers, cell_r_centers, cell_V,\
          cat_shape, cat_dimensions, d_cat_part_v, cat_BET_area, cat_composition, cat_cp, rho_cat, rho_cat_bulk, W_cat]

# Run the function that writes   
write_info_json_steady(names, values, path_sim_data)
write_info_json_dynamic(names, values, path_sim_data)


# -------------------------------------------------
# ------ STEADY SIMULATION
# -------------------------------------------------


converg_flag = False # Flag for recognizing convergence
residuals_C = 1
residuals_T = 1

if sim_type == 'steady' or dynsim_converge_first: # Crank Nicolson scheme for steady state 
    print('\n=======================================')
    print('******* STEADY SIMULATION START')
    print('=======================================\n')
    
    # --- Simulation setup file write - future writing
    # List of name keys for file writing
    names = ['simulation type', 't', 'iteration', 'residuals C', 'residuals T', \
             'C in', 'W/F in', 'T in', 'T wall', 'T hfluid', 'T in hfluid', 'm in hfluid', 'p in hfluid',\
             'T in steam', 'p in steam', 'm out condensate', 'total m out condensate',\
             'Q in', 'Q out', 'm in', 'm out',\
             'v in', 'v out', 'v', 'p in', 'p out', 'p', \
             'CH3OH', 'H2O', 'H2', 'CO2', 'CO', 'N2', \
             'T', 'BET',\
             'rate MSR', 'rate MD', 'rate WGS', 'rate CH3OH', 'rate H2O', 'rate H2', 'rate CO2', 'rate CO'] 
       
    # Small function that updates writing values
    steady_value_list = lambda : [sim_type, t_abs, iteration, residuals_C, residuals_T, \
              C_in_n, (W_cat/ni_in_n[0] if ni_in_n[0]!=0 else 0), T_in_n, T_wall_n, T_hfluid_n, gv.Twall_func.T_in_fgas_steady, gv.Twall_func.m_in_fgas_steady, gv.Twall_func.p_in_fgas_steady,\
              gv.Twall_func.T_steam_steady, gv.Twall_func.p_steam_steady, m_condensed_n, sum(m_condensed_n),\
              Q_in_n, Q_out_n, m_inlet, m_outlet,\
              v_in_n, v_out_n, field_v, p_in_n, p_out_n, field_p, \
              np.asarray(field_Ci_n1[0]), np.asarray(field_Ci_n1[1]), np.asarray(field_Ci_n1[2]), np.asarray(field_Ci_n1[3]), np.asarray(field_Ci_n1[4]), np.asarray(field_Ci_n1[5]),\
              field_T_n1, field_BET_cat,\
              MSR_rate, MD_rate, WGS_rate, CH3OH_rate, H2O_rate, H2_rate, CO2_rate, CO_rate]
    
    # Make value arrays for current and next "timestep"
    # Need to do a deep copy, otherwise the original gets changed as well
    field_Ci_n1 = copy.deepcopy(field_Ci_n)
    field_T_n1 = copy.deepcopy(field_T_n)

    # Get fields of: rates of reaction and formation,
    MSR_rate, MD_rate, WGS_rate, CH3OH_rate, H2O_rate, H2_rate, CO2_rate, CO_rate = mf.get_rate_fields(field_Ci_n1, field_T_n1, field_p, field_BET_cat,
                                                                                                pi_limit)
    # Get mass flows at inlet and outlet
    m_inlet, m_outlet = mf.get_mass_flow(ni_in_n, v_out_n, field_Ci_n1)
    
    iteration = 0
    values = steady_value_list()
        
    write_out_json_steady(names, values, path_sim_data, t_abs, 0)
    
    # Don't change starting value of relaxation factor
    field_relax = copy.deepcopy(field_relax_start)
    
    
    # field_v = np.linspace(0.13846, 0.18886, cells_ax)
    
    # Enter a while loop, condition to drop below convergence limit
    for iteration in range(1,max_iter+1):
        
        # Get steady fluxes from crank nicholson scheme
        C_fluxes_CN, T_fluxes_CN, Twall_fluxes_CN, Thfluid_fluxes_CN, mcond_fluxes_CN =\
            mf.steady_crank_nicholson(field_Ci_n, field_T_n, C_in_n, T_in_n, \
                                      T_wall_n, T_hfluid_n, m_condensed_n,\
                                      field_D_er, field_v, field_p,\
                                      field_BET_cat, field_relax,\
                                      p_ref_n, ni_in_n, Xi_in_n)       
            
        
        # From fluxes at n and n+1, get timestep n+1
        field_Ci_n1 = field_Ci_n + C_fluxes_CN  * field_relax
        field_T_n1 = field_T_n + T_fluxes_CN * field_relax
        
        T_wall_n = T_wall_n + Twall_fluxes_CN*field_relax
        T_hfluid_n = T_hfluid_n + Thfluid_fluxes_CN*field_relax
        m_condensed_n = m_condensed_n + mcond_fluxes_CN# *field_relax
        
        # # Update pressure and velocity field
        # v_in_n, v_out_n, field_v, \
        #     p_in_n, p_out_n, field_p, \
        #     Q_in_n, Q_out_n, C_in_n = mf.get_IO_velocity_and_pressure(p_ref_n, field_T_n, field_Ci_n, field_v, field_p, ni_in_n, T_in_n, Xi_in_n)
        
        
        # # Effective radial diffusion coefficient
        # field_D_er = mf.radial_diffusion_coefficient(field_v, d_cat_part_v, d_tube_in)
        
        # Calculate residuals
        residuals_T = np.max(abs(T_fluxes_CN))
        residuals_C = np.max(abs(C_fluxes_CN))
        residuals = max(residuals_T, residuals_C)
        
        # Save .json ticker
        counter_save += 1
        if counter_save >= steady_save_every:
            
            # Get fields of: rates of reaction and formation, and viscosity
            MSR_rate, MD_rate, WGS_rate, CH3OH_rate, H2O_rate, H2_rate, CO2_rate, CO_rate = mf.get_rate_fields(field_Ci_n1, field_T_n1, field_p, field_BET_cat,
                                                                                                        pi_limit)
            # Get mass flows at inlet and outlet
            m_inlet, m_outlet = mf.get_mass_flow(ni_in_n, v_out_n, field_Ci_n1)
            
            # List of fields and values for .json file writing
            values = steady_value_list()            
            write_out_json_steady(names, values, path_sim_data, t_abs, iteration)
            counter_save = 0.
        
        # Write residuals to console ticker
        counter_terminal += 1
        if counter_terminal >= steady_write_every:
            print(('Iteration:\t{0} \t Residuals T: {1:.3e} \t Residuals C: {2:.3e}' ).format(str(iteration).ljust(10), residuals_T, residuals_C))
            counter_terminal = 0
            
            
        # Update old arrays
        field_Ci_n  = copy.deepcopy( field_Ci_n1 )
        field_T_n = copy.deepcopy( field_T_n1 )
        
        # # Update relaxation factor 
        # field_relax = mf.new_relax_factor(relax_start, relax_max, residuals, residuals_high_limit, residuals_low_limit)
        
        if residuals < convergence:
            print('\n\nConvergence criteria reached!')
            converg_flag = True
            # Make tickers 1 if we have not written anything in previous timestep so that we write at last iteration
            break
    
    if not converg_flag: 
        print('\n\nConvergence criteria not reached after maximum amount of iterations.')
    
    print('---------------------------------------------------')
    print(('Iteration:\t{0} Residuals: \t {1:.5e}').format(str(iteration).ljust(10), residuals))
    print('Convergence criteria: \t\t\t\t', convergence)
    
    # --- Write last file 
    # Get fields of: rates of reaction and formation
    MSR_rate, MD_rate, WGS_rate, CH3OH_rate, H2O_rate, H2_rate, CO2_rate, CO_rate = mf.get_rate_fields(field_Ci_n1, field_T_n1, field_p, field_BET_cat,
                                                                                                pi_limit)
    # Get mass flows at inlet and outlet
    m_inlet, m_outlet = mf.get_mass_flow(ni_in_n, v_out_n, field_Ci_n1)
    
    # List of fields and values for .json file writing
    values = steady_value_list()
    
    # Write out last .json
    write_out_json_steady(names, values, path_sim_data, t_abs, iteration)
    
    # Cleanup the files in data directory
    io.steady_sim_cleanup(saving_json_out, path_sim_data, keep_last_jsons)
    

# -------------------------------------------------
# ------ DYNAMIC SIMULATION
# -------------------------------------------------

# First make a flag for whether to proceed in a time loop or not 
if dynsim_converge_first and not dynsim_cont_converg and not converg_flag :  # If doing auto continuation from steady state
    # If we're doing steady convergence first AND we said don't proceed if steady state is not converged AND convergence flag is still false:
    avoid_timeloop = True # Make a flag to avoid time loop
else: 
    avoid_timeloop = False # Otherwise we will go into time loop
    
if sim_type == 'dynamic':
    print('\n=======================================')
    print('******* DYNAMIC SIMULATION START')
    print('=======================================\n')
    
    # Reset counters in case we had a steady simulation before
    counter_save = 0 # Save file counter
    counter_terminal = 0 # Write in terminal counter
    
    # --- Simulation setup file write - future writing
    # List of name keys for file writing
    names = ['simulation type', 't', 'dt', \
             'C in', 'W/F in', 'T in', 'I', 'T wall', 'T hfluid', 'T in hfluid', 'm in hfluid', 'p in hfluid',\
             'T in steam', 'p in steam', 'm out condensate', 'total m out condensate', \
             'Q in', 'Q out', 'm in', 'm out',\
             'v in', 'v out', 'v', 'p in', 'p out', 'p', \
             'CH3OH', 'H2O', 'H2', 'CO2', 'CO', 'N2',\
             'T', 'BET',\
             'rate MSR', 'rate MD', 'rate WGS', 'rate CH3OH', 'rate H2O', 'rate H2', 'rate CO2', 'rate CO'] 
    
    dynamic_value_list = lambda : [sim_type, round(t_abs,7), dt,\
              C_in_n, (W_cat/ni_in_n[0] if ni_in_n[0]!=0 else 0), T_in_n, gv.Twall_func.I_func(t_abs), T_wall_n, T_hfluid_n, gv.Twall_func.T_in_fgas_func(t_abs), gv.Twall_func.m_in_fgas_func(t_abs), gv.Twall_func.p_in_fgas_func(t_abs),\
              gv.Twall_func.T_steam_func(t_abs), gv.Twall_func.p_steam_func(t_abs), m_condensed_n, sum(m_condensed_n),\
              Q_in_n, Q_out_n, m_inlet, m_outlet,\
              v_in_n, v_out_n, field_v, p_in_n, p_out_n, field_p, \
              np.asarray(field_Ci_n[0]), np.asarray(field_Ci_n[1]), np.asarray(field_Ci_n[2]), np.asarray(field_Ci_n[3]), np.asarray(field_Ci_n[4]), np.asarray(field_Ci_n[5]),\
              field_T_n, field_BET_cat,\
              MSR_rate, MD_rate, WGS_rate, CH3OH_rate, H2O_rate, H2_rate, CO2_rate, CO_rate]
        
    v_in_prev = v_in_n
        
    # --- Calculate timestep size according to CFL criterion
    dt = min(dt, mf.CFL_criterion(CFL, v_in_n, min(cell_dz)))

    # --- Prepare arrays for time stepping (these arrays don't have ghost cells)
    # Empty array of specie concentrations for time n+1
    field_Ci_n1 = copy.deepcopy(field_Ci_n)
    # Empty array of temperatures for time n+1
    field_T_n1 = copy.deepcopy(field_T_n)
        
    
    # Get values from dynamic BC file
    T_in_n = T_in_func(t_rel)
    p_ref_n = p_ref_func(t_rel)
    flowrate_in_n = flowrate_in_func(t_rel)
    
    # Get fields of: rates of reaction and formation
    MSR_rate, MD_rate, WGS_rate, CH3OH_rate, H2O_rate, H2_rate, CO2_rate, CO_rate = mf.get_rate_fields(field_Ci_n, field_T_n, field_p, field_BET_cat,
                                                                                                pi_limit)
    # Get mass flows at inlet and outlet
    m_inlet, m_outlet = mf.get_mass_flow(ni_in_n, v_out_n, field_Ci_n1)
    
    # -- Write .json at 0th timestep
    values = dynamic_value_list()
    write_out_json_dynamic(names, values, path_sim_data, round(t_abs,7))
    
    # After we've written timestep t=0, see if we need to avoid proceeding in time loop
    if avoid_timeloop: # IF the flag was raised: 
        print('\nSteady state convergence not achieved. Skipping dynamic simulation.')
        
    else:
    
        # --- Time loop
        while t_rel < t_dur:
            
            # --- RK4 arrays
            # Time 
            RK_dt = np.asarray([0, dt/2, dt/2, dt])
            RK_times = RK_dt + t_rel
            # Inlet temperature
            RK_T_in = np.asarray([T_in_n, T_in_func(t_rel + dt/2), T_in_func(t_rel+dt)])
            RK_T_in = np.insert(RK_T_in, 1, RK_T_in[1])
            # Inlet pressure
            RK_p_ref = np.asarray([p_ref_n, p_ref_func(t_rel + dt/2), p_ref_func(t_rel+dt)])
            RK_p_ref = np.insert(RK_p_ref, 1, RK_p_ref[1])
            # Inlet flow rate
            RK_flowrate_in = np.asarray([flowrate_in_n, flowrate_in_func(t_rel + dt/2), flowrate_in_func(t_rel+dt)])
            RK_flowrate_in = np.insert(RK_flowrate_in, 1, RK_flowrate_in[1])
            
            
            # --- Calculate fluxes with Runge Kutta 4th order scheme
            Ci_fluxes, T_fluxes, Tw_fluxes, Thf_fluxes, m_condensate_fluxes = mf.RK4_fluxes(RK_dt, RK_times, SC_ratio, Xi_in_n,
                T_wall_n, T_hfluid_n, m_condensed_n, RK_T_in, RK_flowrate_in, RK_p_ref, field_v, field_p, C_in_n,
                field_Ci_n, field_T_n,
                field_BET_cat)
            
            # --- Evolve in time using fluxes and dt
            field_Ci_n1 = field_Ci_n + dt*Ci_fluxes
            field_T_n1 = field_T_n + dt*T_fluxes
            
            T_wall_n = T_wall_n + dt*Tw_fluxes
            T_hfluid_n = T_hfluid_n + dt*Thf_fluxes
            m_condensed_n = m_condensed_n + dt*m_condensate_fluxes
            
            # --- Update n fields 
            # Update the molar concentration fields
            field_Ci_n = copy.deepcopy(field_Ci_n1)
            # Update the temperature field
            field_T_n = copy.deepcopy(field_T_n1)

            # --- Update values
            # Times
            t_abs += dt # absolute time
            t_rel += dt # relative time
            
            # Copy previous velocity
            v_in_prev = v_in_n
        
            T_in_n = T_in_func(t_rel)
            p_ref_n = p_ref_func(t_rel)
            flowrate_in_n = flowrate_in_func(t_rel)
            ni_in_n, Xi_in_n = gv.inlet.CX(flowrate_in_n)
            
            # Get velocity, pressure, and some other variables
            v_in_n, v_out_n, field_v, \
                p_in_n, p_out_n, field_p, \
                Q_in_n, Q_out_n, C_in_n = mf.get_IO_velocity_and_pressure(p_ref_n, field_T_n, field_Ci_n, field_v, field_p, ni_in_n, T_in_n, Xi_in_n)

            # Calculate new radial diffusion coefficient
            # field_D_er = mf.radial_diffusion_coefficient(field_v, d_cat_part_v, d_tube_in)
        
            counter_save += 1
            if counter_save >= dyn_save_every:
                # Get fields of: rates of reaction and formation, and viscosity
                MSR_rate, MD_rate, WGS_rate, CH3OH_rate, H2O_rate, H2_rate, CO2_rate, CO_rate = mf.get_rate_fields(field_Ci_n1, field_T_n1, field_p, field_BET_cat,
                                                                                                            pi_limit)
                # Get mass flows at inlet and outlet
                m_inlet, m_outlet = mf.get_mass_flow(ni_in_n, v_out_n, field_Ci_n1)
                
                # List of fields and values for .json file writing
                values = dynamic_value_list()
                write_out_json_dynamic(names, values, path_sim_data, round(t_abs,7))
                counter_save = 0.
                
            # Write to console - time
            # ?? Inputs (how they change in time), residuals
            counter_terminal += 1
            if counter_terminal >= dyn_write_every:
                print(('Timestep:\t{0}').format(round(t_abs,7)))
                counter_terminal = 0
                
        # --- Write last timestep to .json
        # Get fields of: rates of reaction and formation, and viscosity
        MSR_rate, MD_rate, WGS_rate, CH3OH_rate, H2O_rate, H2_rate, CO2_rate, CO_rate = mf.get_rate_fields(field_Ci_n1, field_T_n1, field_p, field_BET_cat,
                                                                                                    pi_limit)
        
        # Get mass flows at inlet and outlet
        m_inlet, m_outlet = mf.get_mass_flow(ni_in_n, v_out_n, field_Ci_n1)
        
        # Write final .json
        values = dynamic_value_list()            
        write_out_json_dynamic(names, values, path_sim_data,  round(t_end_abs,7))
        print('\nDynamic simulation completed successfully.')
        
        print(('Simulaton end time:\t{0}').format(round(t_end_abs,7)))


# Calculate elapsed execution time
delta_time = (datetime.now() - start_time)
hours, remainder = divmod(delta_time.seconds, 3600) # do some parsing
minutes, seconds = divmod(remainder, 60)
exe_time = str(hours).zfill(2) + ':' + str(minutes).zfill(2) + ':' + str(seconds).zfill(2) # Make a nice string


print('\n\n===== Script end reached ======')

print('\nScript ended on:')
print('dd/mm/yyyy\thh:mm:ss')
print(datetime.now().strftime("%d/%m/%Y\t%H:%M:%S"))

print('\nExecution time:')
print(exe_time)
print ('hh:mm:ss')

# Close log if it was opened
io.close_log(saving_log, original_stdout)    
    
    
    
    
    
    
    
    
    




       









