# -*- coding: utf-8 -*-
"""
Main file 
Methanol steam reforming model
plug flow packed bed reactor 


Author:         Bojan Grenko
Institution:    Delft University of Technology 

This code uses a model which is a dynamic expansion of the model of Dr. Jimin Zhu, developed in their PhD thesis.

This code was produced within the MENENS project. 

"""

import numpy as np
import math

import os
import sys
import copy

import time
from datetime import datetime

# My files
import model_funcs as mf
# from model_funcs import Specie, CH3OH, H2O, H2, CO2, CO
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

# !!! s_tube, rho_tube, h_tube, cp_tube,
# -------------------------------------------------
# ------ INPUTS 
# -------------------------------------------------

# --- Read all inputs from .json file
cat_shape, cat_dimensions, cat_BET_area, known_cat_density, rho_cat, rho_cat_bulk, cat_composition, \
    n_tubes, l_tube, d_tube, s_tube, rho_tube, h_tube, cp_tube,\
    cells_ax, cells_rad, adv_scheme, diff_scheme, CFL, pi_limit, Ci_limit,\
    SC_ratio, p_ref_n, p_set_pos, T_in_n, init_T_field, WF_in_n,\
    sim_type, max_iter, convergence, relax_start, dt, t_dur, dyn_bc, cont_sim, path_old_sim_data,\
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
# cat_composition - [-] Catalyst composition - weight percentages of CuO and Al2O3

# --- Reactor parameters
# n_tubes   [-] Total number of tubes in the reactor
# l_tube    [m] single tube length
# d_tube    [m] single tube diameter
# s_tube    [m] reactor tube thickness
# rho_tube  [kg m-3] tube material density
# h_tube    [W m-1 K-1] tube material thermal conductivity
# cp_tube   [J kg-1 K-1] tube material specific heat capacity

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
# WF_in_n   [kg s mol-1] - convert this to volumetric flow rate and velocity

# --- Simulation parameters
# sim_type              [-] simulation type (steady / dynamic)
# max_iter              [-] Maximum allowed iterations in steady simulation
# convergence           [-] Convergence criteria in steady simulation
# relax                 [-] Relaxation factor in steady simulation
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


uniform_dz = True  # Parameter to check whether uniform axial mesh with constant dz is used


'''
TO ADD
'''

# !!! CODE IN: 
    # Effectiveness factors
    # Add residuals to dynamic sym
    # n_tubes - handle this somehow? - maybe in postprocessing. only relevant if i have burner gas in this sim

    # Effectiveness factors
nu_i = 1
nu_j = 1


# -------------------------------------------------
# ------ Preparing directories and files
# -------------------------------------------------

# Make a directory for our results if needed
if saving_only_terminal == 'no':
    path_subdir_results, path_sim_data = io.create_result_dir(results_dir_name, cont_sim, sim_type, saving_timestamp, timestamp)
    
    if saving_files_in == 'yes': # copy the original json and dynamic BCs if specified
        io.copy_json(input_json_fname, path_subdir_results)
    
else:
    path_subdir_results, path_sim_data = '', ''


# Get a function that either writes the output json file, or does nothing depending on our choice
if sim_type == 'steady':
    write_out_json_steady, write_info_json_steady = io.output_write_settings(saving_json_out)
    write_out_json_dynamic, write_info_json_dynamic = io.output_write_settings('no')
elif sim_type == 'dynamic':
    write_out_json_steady, write_info_json_steady = io.output_write_settings('no')
    write_out_json_dynamic, write_info_json_dynamic = io.output_write_settings(saving_json_out)


# Create a simulation log and if needed 
logname =  ' log.txt'
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

# --- Read data for simulation continuation
if cont_sim == 'yes':
    
    # --- Read parameters and fields from .json file
    t_abs, field_Ci_n, field_T_n, field_p, field_v, field_BET_cat, T_wall_n,\
        SC_ratio, n_tubes, l_tube, d_tube, N, epsilon, cells_ax_old, cells_rad_old,\
        cat_shape, cat_dimensions, rho_cat, rho_cat_bulk, cat_composition, cat_cp, cat_BET_area = io.read_cont_sim_data(path_old_sim_data)
    
    # --- Calculate catalyst particle (equivalent) volume and diameter 
    V_cat_part, d_cat_part = mf.catalyst_dimensions(cat_shape, cat_dimensions)
        
    # --- Reactor parameters
    r_tube = d_tube/2   # [m] Reactor tube radius
    
    V_tube = np.pi * r_tube**2 * l_tube # [m3] internal volume of reactor tube
    W_cat = rho_cat_bulk * V_tube # [kg] Weight of catalyst in one 

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
            field_Ci_n, field_T_n, field_BET_cat = mf.interpolate_to_new_mesh(l_tube, r_tube, 
                                                                              cells_ax_old, cells_rad_old, cells_ax, cells_rad,
                                                                              field_Ci_n, field_T_n, field_BET_cat)

    # --- Get and prepare inlet/outlet values
    v_in_n, v_out_n, field_v, \
        p_in_n, p_out_n, field_p, \
        Q_in_n, Q_out_n, C_in_n, C_out_n, rho_in_n, rho_out_n, \
        X_in_n, X_out_n, mu_in_n, mu_out_n = mf.get_IO_velocity_and_pressure(p_ref_n, p_set_pos, T_in_n, WF_in_n, SC_ratio, W_cat, epsilon, r_tube, l_tube, d_cat_part, cell_z_centers)

    

elif cont_sim == 'no':

    # --- Catalyst parameters
    # Calculate catalyst particle (equivalent) volume and diameter 
    V_cat_part, d_cat_part = mf.catalyst_dimensions(cat_shape, cat_dimensions) 
        
    # Aspect ratio - size ratio of reactor tube to catalyst particle 
    N = mf.aspect_ratio(d_tube, d_cat_part)
    
    # Reactor (packed bed) porosity
    epsilon = mf.porosity(d_tube, d_cat_part, cat_shape)
    
    # Catalyst material and shipping densities - if one of them is unknown
    rho_cat, rho_cat_bulk = mf.catalyst_densities(rho_cat, rho_cat_bulk, epsilon, known_cat_density)
    
    # Catalyst material specific heat capacity
    cat_cp = mf.catalyst_cp(cat_composition)
    
    # --- Reactor parameters
    r_tube = d_tube/2   # [m] Reactor tube radius
    
    V_tube = np.pi * r_tube**2 * l_tube # [m3] internal volume of reactor tube
    W_cat = rho_cat_bulk * V_tube # [kg] Weight of catalyst in one 
    
    
    # --- Geometry discretization
    cell_dz, cell_dr, \
        cell_z_faces_L, cell_z_faces_R, cell_z_centers, \
        cell_r_faces_IN, cell_r_faces_EX, cell_r_centers, \
        cell_z_A, cell_r_A_IN, cell_r_A_EX, \
        cell_V = mf.uniform_mesh(cells_ax, cells_rad, l_tube, r_tube)
    
    # --- Operation parameters
    
    # --- Prepare some variables and do field initialization
    v_in_n, v_out_n, field_v, \
        p_in_n, p_out_n, field_p, \
        Q_in_n, Q_out_n, C_in_n, C_out_n, rho_in_n, rho_out_n, \
        X_in_n, X_out_n, mu_in_n, mu_out_n = mf.get_IO_velocity_and_pressure(p_ref_n, p_set_pos, T_in_n, WF_in_n, SC_ratio, W_cat, epsilon, r_tube, l_tube, d_cat_part, cell_z_centers)
    
    
    # Make temperature field
    field_T_n = np.ones((cells_rad, cells_ax)) * init_T_field
    
    # Take initial wall temperature guess to be the same as outer cell in T field
    T_wall_n = field_T_n[0,:]

    # Make concentration array
    field_Ci_n = np.zeros((5, cells_rad, cells_ax)) # Empty array
    # Assume reactor is filled with inlet mixture at t=0
    field_Ci_n[0, :,:] = C_in_n[0] # CH3OH
    field_Ci_n[1, :,:] = C_in_n[1] # H2O
    
    # Catalyst surface area array
    field_BET_cat = np.ones((cells_rad, cells_ax)) * cat_BET_area
    
    # Simulation time is always zero in new simulation
    t_abs = 0.0


# Set the amount of ghost cells on each domain side based on chosen discretization schemes
adv_index, diff_index = mf.discretization_schemes_and_ghost_cells(adv_scheme, diff_scheme)

# !!! 
# Add ghost cells to concentration and temperature fields 
# field_Ci_n, field_T_n = mf.append_ghost_cells(field_Ci_n, field_T_n, gcells_z_in, gcells_z_out, gcells_r_wall, gcells_r_ax)

# Effective radial diffusion coefficient
field_D_er = mf.radial_diffusion_coefficient(field_v, d_cat_part, d_tube)

# Define simulation absolute end time 
t_end_abs = t_dur + t_abs
# Relative time
t_rel = 0.

# Read dynamic simulation boundary conditions and make a function from it 
if sim_type == 'dynamic':
    T_in_func, WF_in_func, p_ref_func = io.read_and_set_BCs(input_json_fname, dyn_bc, T_in_n, WF_in_n, p_ref_n)
    # We do this now because then we have BCs in the script and we don't need the input .json anymore

# Get wall heating choice, and make a class (Twall_func) in global_vars that has wall temperature methods for steady and dynamic operation
# Also get a temperature function that 
wall_heating = mf.read_and_set_T_wall_BC(input_json_fname, dyn_bc, cell_z_centers, l_tube)

# Add some unchanging variables to this class that uses it in some methods 
gv.Twall_func.d_p = d_cat_part
gv.Twall_func.epsilon = epsilon


'''
Things to add here:

- non-uniform mesh (currently everywhere it is assumed that mesh is uniform in z direction) ???

'''

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
    print('Starting underrelaxation factor:'.ljust(colw) +'{0:.2e}'.format(relax_start))
elif sim_type == 'dynamic':
    print('\n# --- Dynamic simulation parameters:')
    print('Simulated time duration:'.ljust(colw) +'{0}'.format(t_dur))
    print('Simulation timestep size [s]:'.ljust(colw) +'{0:.2e}\n'.format(dt))
    
    if dyn_bc == 'yes':
        print('# --- Dynamic boundary conditions used in simulation')
        io.check_used_dynamic_BCs(input_json_fname, p_set_pos, colw)
    
if cont_sim == 'yes':
    print('\n# --- Simulation continued from:'.ljust(colw) + 'results{0}'.format(path_old_sim_data.split('results')[-1].split('sim_data')[0] ))
    if saving_only_terminal == 'no':
        print('Current simulation data path:'.ljust(colw) + 'results{0}'.format(path_subdir_results.split('results')[-1]))
    
    print('Continuing from simulation time:'.ljust(colw) +'{0:.5f} [s]'.format(t_abs))
    print('Simulation absolute end time:'.ljust(colw) +'{0:.5f} [s]'.format(t_end_abs))
 
 
print('\n\n=====================================================')
print('=== Parameter Initialization:')
print('=====================================================\n')

print('# --- Calculated catalyst parameters for %s catalyst' %(cat_shape.upper())) 
print('-----------------------------------------------------\n')

print('Cat. particle (equivalent) diameter:'.ljust(colw) +'{0:.5f} [m]'.format(d_cat_part))
print('Cat. particle (equivalent) volume:'.ljust(colw) + '{0:.3e} [m3]\n'.format(V_cat_part))

print('Packed bed aspect ratio (d_tube / d_cat):'.ljust(colw) +'{0:.2f} [-]'.format(N))
print('Reactor porosity:'.ljust(colw) +'{0:.5f} [-]\n'.format(epsilon))

print('Catalyst material density:'.ljust(colw) +'{0:.2f} [kg m-3]'.format(rho_cat))
print('Catalyst bulk density:'.ljust(colw) +'{0:.2f} [kg m-3]'.format(rho_cat_bulk))

print('Catalyst specific heat capacity:'.ljust(colw) +'{0:.2f} [J kg-1 K-1]\n'.format(cat_cp))

print('Reactor (single) tube volume:'.ljust(colw) +'{0:.3e} [m3]'.format(V_tube))
print('Cat. weight in reactor (single) tube:'.ljust(colw) +'{0:.5f} [kg]\n'.format(W_cat))

print('\n# --- Calculated operation parameters') 
print('-----------------------------------------------------\n')

print('Init. inlet W_cat/F_CH3OH (single tube):'.ljust(colw) +'{0:.3f} [kg s mol-1]'.format(WF_in_n))
print('Init. inlet vol. flow rate (single tube):'.ljust(colw) +'{0:.3e} [m3 s-1]'.format(Q_in_n))
print('Init. inlet flow (superficial) velocity:'.ljust(colw) +'{0:.5f} [m s-1]\n'.format(v_in_n))
    
# print('Init. inlet mixture density:'.ljust(colw) +'{0:.3e} [kg m-3]'.format(rho_in_n))
# print('Init. inlet mixture (dynamic) viscosity:'.ljust(colw) +'{0:.5e} [Pa s]\n'.format(mu_in_n))

print('Init. reactor pressure drop:'.ljust(colw) +'{0:.5f} [bar]\n'.format( (p_in_n - p_out_n)/1e5  ))


print( ('Initial given pressure at {0}:'.format(p_set_pos.upper())).ljust(colw) + '{0:.2f} [bar]'.format(p_ref_n/1e5)) 
print('Init. inlet radial diffusion coeff.:'.ljust(colw) +'{0:.3e} [m2 s-1]\n'.format(field_D_er[0]))


# -------------------------------------------------
# ------ SCRIPT INTERNAL PARAMETERS / NAMES 
# -------------------------------------------------


# Set counters 
counter_terminal = 0
counter_save = 0

# --- Simulation setup file write - one time writing
# List of name keys for file writing
names = ['simulation type', 'simulated time', 'S/C', 'wall heating type' ,'n tubes', 'tube l', 'tube r', 'tube V' ,'aspect ratio', 'porosity',\
         'n ax cells', 'n rad cells', 'dz', 'dy', 'cell z centers', 'cell r centers', 'cell volumes',\
         'cat shape', 'cat dimensions', 'cat particle d', 'fresh cat BET', 'cat composition', 'cat cp', 'cat rho', 'cat rho bulk', 'cat tube weight']

if sim_type == 'steady' : t_dur = 'n/a' # simulation duration not applicable if we have steady simulation
# List of fields and values for .json file writing
values = [sim_type, t_dur, SC_ratio, wall_heating, n_tubes, l_tube, r_tube, V_tube, N, epsilon,\
          cells_ax, cells_rad, cell_dz, cell_dr, cell_z_centers, cell_r_centers, cell_V,\
          cat_shape, cat_dimensions, d_cat_part, cat_BET_area, cat_composition, cat_cp, rho_cat, rho_cat_bulk, W_cat]

# Run the function that writes   
write_info_json_steady(names, values, path_sim_data)
write_info_json_dynamic(names, values, path_sim_data)

# Get meshgrids of 
dz_mgrid, dr_mgrid = np.meshgrid(cell_dz, cell_dr) 
z_centers_mgrid, r_centers_mgrid = np.meshgrid(cell_z_centers, cell_r_centers)


# -------------------------------------------------
# ------ STEADY SIMULATION
# -------------------------------------------------


converg_flag = False # Flag for recognizing convergence

if sim_type == 'steady' or dynsim_converge_first =='yes': # Crank Nicolson scheme for steady state 
    print('\n=======================================')
    print('******* STEADY SIMULATION START')
    print('=======================================\n')
    
    # --- Simulation setup file write - future writing
    # List of name keys for file writing
    names = ['simulation type', 't', 'iteration', 'residuals', \
             'C in', 'W/F in', 'T in', 'T wall', 'Q in', 'Q out', 'm in', 'm out',\
             'v in', 'v out', 'v', 'p in', 'p out', 'p', \
             'CH3OH', 'H2O', 'H2', 'CO2', 'CO',\
             'T', 'BET',\
             'rate MSR', 'rate MD', 'rate WGS', 'rate CH3OH', 'rate H2O', 'rate H2', 'rate CO2', 'rate CO'] 
       
    # Small function that updates writing values
    steady_value_list = lambda : [sim_type, t_abs, 0, iteration,\
              C_in_n, WF_in_n, T_in_n, T_wall_n, Q_in_n, Q_out_n, m_inlet, m_outlet,\
              v_in_n, v_out_n, field_v, p_in_n, p_out_n, field_p, \
              np.asarray(field_Ci_n1[0]), np.asarray(field_Ci_n1[1]), np.asarray(field_Ci_n1[2]), np.asarray(field_Ci_n1[3]), np.asarray(field_Ci_n1[4]),\
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
    m_inlet, m_outlet = mf.get_mass_flow(v_in_n, v_out_n, r_tube, C_in_n, field_Ci_n1)
    
    iteration = 0
    values = steady_value_list()
        
    write_out_json_steady(names, values, path_sim_data, t_abs, 0)
    
    # !!! Copy relaxation factor in case we change it later
    relax = copy.deepcopy(relax_start)
    
    # Enter a while loop, condition to drop below convergence limit
    for iteration in range(1,max_iter+1):

        # Calculate/retrieve wall temperature
        T_wall_n = gv.Twall_func.steady(relax, T_wall_n, field_Ci_n[:,0,:], field_T_n[0,:], field_v)
        
        # Get steady fluxes from crank nicholson scheme
        C_fluxes_CN, T_fluxes_CN = mf.steady_crank_nicholson(field_Ci_n, field_T_n, cells_rad, C_in_n, T_in_n, T_wall_n,\
                           field_D_er, field_v, field_p,\
                           dz_mgrid, dr_mgrid, cell_V, r_centers_mgrid,
                           rho_cat_bulk, field_BET_cat, d_cat_part, cat_cp, cat_shape,
                           epsilon, N, adv_index, diff_index, pi_limit, nu_i, nu_j,\
                           relax)        
        
        # From fluxes at n and n+1, get timestep n+1
        field_Ci_n1 = field_Ci_n + C_fluxes_CN  * relax
        field_T_n1 = field_T_n + T_fluxes_CN * relax
        
        # Calculate residuals - use largest absolte T flux
        residuals = np.max(abs(T_fluxes_CN))

        # Save .json ticker
        counter_save += 1
        if counter_save >= steady_save_every:
            
            # Get fields of: rates of reaction and formation, and viscosity
            MSR_rate, MD_rate, WGS_rate, CH3OH_rate, H2O_rate, H2_rate, CO2_rate, CO_rate = mf.get_rate_fields(field_Ci_n1, field_T_n1, field_p, field_BET_cat,
                                                                                                        pi_limit)
            # Get mass flows at inlet and outlet
            m_inlet, m_outlet = mf.get_mass_flow(v_in_n, v_out_n, r_tube, C_in_n, field_Ci_n1)
            
            # List of fields and values for .json file writing
            values = steady_value_list()            
            write_out_json_steady(names, values, path_sim_data, t_abs, iteration)
            counter_save = 0.
        
        # Write residuals to console ticker
        counter_terminal += 1
        if counter_terminal >= steady_write_every:
            print(('Iteration:\t{0} Residuals: \t {1:.3e}').format(str(iteration).ljust(10), residuals))
            counter_terminal = 0
            print('{0:.5f}-{1:.5f} \t {2:.2e}'.format(T_wall_n[0], T_wall_n[-1], relax))
        
        # Update old arrays
        field_Ci_n  = copy.deepcopy( field_Ci_n1 )
        field_T_n = copy.deepcopy( field_T_n1 )
        
        # Update relaxation factor 
        # relax = new_relax_factor(relax_start, relax_max, residuals, residuals_high_limit, residuals_low_limit)
        
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
    # Get Wall temperature profile
    T_wall_n = gv.Twall_func.steady(relax, T_wall_n, field_Ci_n[:,0,:], field_T_n[0,:], field_v)

    # Get mass flows at inlet and outlet
    m_inlet, m_outlet = mf.get_mass_flow(v_in_n, v_out_n, r_tube, C_in_n, field_Ci_n1)
    
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
if dynsim_converge_first == 'yes' and dynsim_cont_converg == 'no' and converg_flag == False:  # If doing auto continuation from steady state
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
             'C in', 'W/F in', 'T in', 'I', 'T wall', 'Q in', 'Q out', 'm in', 'm out',\
             'v in', 'v out', 'v', 'p in', 'p out', 'p', \
             'CH3OH', 'H2O', 'H2', 'CO2', 'CO',\
             'T', 'BET',\
             'rate MSR', 'rate MD', 'rate WGS', 'rate CH3OH', 'rate H2O', 'rate H2', 'rate CO2', 'rate CO'] 
    
    dynamic_value_list = lambda : [sim_type, round(t_abs,7), dt,\
              C_in_n, WF_in_n, T_in_n, gv.Twall_func.I_func(t_abs), T_wall_n, Q_in_n, Q_out_n, m_inlet, m_outlet,\
              v_in_n, v_out_n, field_v, p_in_n, p_out_n, field_p, \
              np.asarray(field_Ci_n[0]), np.asarray(field_Ci_n[1]), np.asarray(field_Ci_n[2]), np.asarray(field_Ci_n[3]), np.asarray(field_Ci_n[4]),\
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
        
    # Effectiveness factors?
    
    # Get values from dynamic BC file
    T_in_n = T_in_func(t_rel)
    p_ref_n = p_ref_func(t_rel)
    WF_in_n = WF_in_func(t_rel)
    
    # Get fields of: rates of reaction and formation
    MSR_rate, MD_rate, WGS_rate, CH3OH_rate, H2O_rate, H2_rate, CO2_rate, CO_rate = mf.get_rate_fields(field_Ci_n, field_T_n, field_p, field_BET_cat,
                                                                                                pi_limit)
    # Get mass flows at inlet and outlet
    m_inlet, m_outlet = mf.get_mass_flow(v_in_n, v_out_n, r_tube, C_in_n, field_Ci_n1)
    
    # -- Write .json at 0th timestep
    values = dynamic_value_list()
    write_out_json_dynamic(names, values, path_sim_data, round(t_abs,7))
    
    # After we've written timestep t=0, see if we need to avoid proceeding in time loop
    if avoid_timeloop: # IF the flag was raised: 
        print('\nSteady state convergence not achieved. Skipping dynamic simulation.')
        
    else:
    
        # --- Time loop
        while t_rel < t_dur:
            
            # Change time step size for last time step
            # dt = min(dt, t_dur - t_rel)
             
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
            # Inlet WF
            RK_WF_in = np.asarray([WF_in_n, WF_in_func(t_rel + dt/2), WF_in_func(t_rel+dt)])
            RK_WF_in = np.insert(RK_WF_in, 1, RK_WF_in[1])
            
            # --- Calculate fluxes with Runge Kutta 4th order scheme
            Ci_fluxes, T_fluxes = mf.RK4_fluxes(RK_dt, RK_times, cells_rad, cells_ax, cell_V, cell_r_centers, cell_z_centers,
                dz_mgrid, dr_mgrid, z_centers_mgrid, r_centers_mgrid,
                W_cat, SC_ratio, r_tube,
                T_wall_n, RK_T_in, RK_WF_in, RK_p_ref, p_set_pos,
                field_Ci_n, field_T_n,
                rho_cat_bulk, field_BET_cat, d_cat_part, cat_cp, cat_shape, d_tube, l_tube,
                epsilon, N, pi_limit, nu_i, nu_j, adv_index, diff_index)
            
            # --- Evolve in time using fluxes and dt
            field_Ci_n1 = field_Ci_n + dt*Ci_fluxes
            field_T_n1 = field_T_n + dt*T_fluxes
                    
            # --- Update n fields 
            # Update the molar concentration fields
            for i in range(5):
                field_Ci_n = copy.deepcopy(field_Ci_n1)
            
            # Update the temperature field
            field_T_n = copy.deepcopy(field_T_n1)
        
            # --- Update values
            # Times
            t_abs += dt # absolute time
            t_rel += dt # relative time
            
            # Copy previous velocity
            v_in_prev = v_in_n
        
            # Boundary conditions
            T_wall_n = gv.Twall_func.dynamic(t_rel, dt, T_wall_n, field_Ci_n[:,0,:], field_T_n[0,:], field_v)
            
            T_in_n = T_in_func(t_rel)
            p_ref_n = p_ref_func(t_rel)
            WF_in_n = WF_in_func(t_rel)
            
            
            # Get velocity, pressure, and some other variables
            v_in_n, v_out_n, field_v, \
                p_in_n, p_out_n, field_p, \
                Q_in_n, Q_out_n, C_in_n, C_out_n, rho_in_n, rho_out_n, \
                X_in_n, X_out_n, mu_in_n, mu_out_n = mf.get_IO_velocity_and_pressure(p_ref_n, p_set_pos, T_in_n, WF_in_n, SC_ratio, W_cat, epsilon, r_tube, l_tube, d_cat_part, cell_z_centers)
    
                    
            if v_in_n != v_in_prev: # If velocity changed
            # Calculate new radial diffusion coefficient
                field_D_er = mf.radial_diffusion_coefficient(field_v, d_cat_part, d_tube)
        
            counter_save += 1
            if counter_save >= dyn_save_every:
                # Get fields of: rates of reaction and formation, and viscosity
                MSR_rate, MD_rate, WGS_rate, CH3OH_rate, H2O_rate, H2_rate, CO2_rate, CO_rate = mf.get_rate_fields(field_Ci_n1, field_T_n1, field_p, field_BET_cat,
                                                                                                            pi_limit)
                # Get mass flows at inlet and outlet
                m_inlet, m_outlet = mf.get_mass_flow(v_in_n, v_out_n, r_tube, C_in_n, field_Ci_n1)
                
                
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
        m_inlet, m_outlet = mf.get_mass_flow(v_in_n, v_out_n, r_tube, C_in_n, field_Ci_n1)
        
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
    
    
    
    
    
    
    
    
    




       









