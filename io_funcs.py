# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:24:12 2023

@author: bgrenko
"""

import sys 
import os
import shutil

import numpy as np
import pandas as pd
import json

import math



''' 
==============================================================================================================================
------------------------------------------------------------------------------------------------------------------------------
--------------------- INPUTS & PRE-SOLVER
------------------------------------------------------------------------------------------------------------------------------
==============================================================================================================================
'''



def simulation_inputs(input_json_fname):
    """
    Reads and imports user inputs from .json file
    
    Parameters
    ----------
    input_json_fname : str
        Input .json file name

    Raises
    ------
    ImportError
        (One of the) user input(s) is not recognized

    Returns
    -------
    return_list : list
        List of user inputs

    """
    'simulation_setup.json'
    file = json.load(open(input_json_fname))
    
    catalyst = file['catalyst parameters']
    reactor = file['reactor parameters']
    numerics = file['numerical parameters']
    field = file['field parameters']
    operation = file['simulation parameters']
    saving = file['save file parameters']
    
    solver_dir = os.getcwd() # Current directory
    
    # --- Catalyst parameters
    cat_shape = str(catalyst['catalyst particle shape (sphere / cylinder)']).lower()
    
    if cat_shape == 'sphere':
        cat_dimensions = catalyst['sphere particle diameter [m]']
    elif cat_shape == 'cylinder':
        cat_dimensions = catalyst['cylinder particle (height, diameter) [m,m]']
        
        if type(cat_dimensions) != list or len(cat_dimensions) != 2: 
            raise NameError('Cylindrical catalyst particles must be defined by with a list of two parameters - [height, diameter]')
            
    else: 
        raise NameError('Catalyst shape incorrectly set')
        
    cat_BET_area =  catalyst['BET surface area [m2 kg-1]']
    
    cat_composition = catalyst['catalyst composition: molar percentage of CuO and ZnO; Al2O3 to balance [-]']
    if type(cat_composition) != list or len(cat_composition) != 2:
        raise NameError('catalyst composition must be defined by with a list of two parameters - [CuO, ZnO]')

    if sum(cat_composition) > 1:
        raise ValueError('Weight percentages of catalyst composition must be below 1!')
    
    known_cat_density = str(catalyst['known catalyst density (density / bulk density / both)']).lower()
    if known_cat_density != 'density' and known_cat_density != 'bulk density' and known_cat_density != 'both':
        raise NameError('Given known catalyst density incorrectly set: {}'.format(known_cat_density))
    
    rho_cat = float(catalyst['catalyst density [kg m-3]'])
    rho_cat_bulk = float(catalyst['catalyst bulk density [kg m-3]'])

    
    # --- Reactor parameters
    n_tubes = int(reactor['total number of tubes in the reactor [-]'])
    tube_l = float(reactor['single tube length [m]'])
    tube_d = float(reactor['single tube diameter [m]'])
    
    
    # --- Numerical parameters
    ax_cells = int(numerics['number of axial cells [-]'])
    rad_cells = int(numerics['number of radial cells [-]'])
    
    adv_scheme = str(numerics['advection discretization scheme (upwind_1o / upwind_2o)']).lower()
    diff_scheme = str(numerics['diffusion discretization scheme (central_2o / central_4o)']).lower()
    
    # Check if we're passing correct choices
    if adv_scheme != 'upwind_1o' and adv_scheme != 'upwind_2o': 
        raise NameError('Advection discretization scheme not recognized: {}'.format(adv_scheme))
    if diff_scheme != 'central_2o' and diff_scheme != 'central_4o':
        raise NameError('Diffusion discretization scheme not recognized: {}'.format(diff_scheme))
    
    CFL = float(numerics['CFL number [-]'])
    partP_limit = float(numerics['conversion model partial pressure low limit [Pa]'])
    Ci_limit = float(numerics['specie concentration low limit [mol m-3]'])

    # --- field parameters
    SC_ratio = float(field['feed steam to carbon ratio [-]'])
    p_ref = float(field['reactor pressure [bar]'])
    p_set_pos = str(field["position of given reactor pressure (inlet / outlet)"]).lower()
    if p_set_pos != 'inlet' and p_set_pos != 'outlet':
        raise NameError('Given position of reactor pressure not recognized: {}'.format(p_set_pos))
    
    
    inlet_gas_T = float(field['inlet gas temperature [C]'])
    init_reactor_T = float(field['initial reactor temperature [C]'])
    wall_T_rel_profile = np.asarray(field['reactor wall temperature profile [C]'])
    wall_T_rel_positions = np.asarray(field['wall temperature relative axial positions [-]'])
    
    # Calculate wall temperature distribution 
    rel_dz_half = (1 / ax_cells) / 2 # Get relative axial cell size (uniform mesh)
    # Mesh in relative coordinates (from 0 to 1)
    rel_mesh = np.linspace(rel_dz_half,1.-rel_dz_half,int(ax_cells)) 
    # Get initial wall T profile
    wall_T_profile = [np.interp(x, wall_T_rel_positions, wall_T_rel_profile) for x in rel_mesh]
    
    flow_rate = str(field['flow rate given with (WF / WHSV)']).upper()
    if flow_rate != 'WF' and flow_rate != 'WHSV':
        raise NameError('Given known initial flow rate not recongized: {0}'.format(flow_rate))
        
    if flow_rate == 'WF':
        W_F = field['W_cat/F_CH3OH - catalyst weight per CH3OH molar feed rate [kg s mol-1]']
    elif flow_rate == 'WHSV': 
        W_F = field['WHSV - hourly CH3OH molar feed rate per catalyst weight [mol h-1 kg-1]']
        W_F = 1/(W_F / 3600)# get [ kg s mol-1]
    
    # --- Simulation type and operation parameters 
    sim_type = str(operation['simulation type (steady / dynamic)']).lower() 
    
    if sim_type != 'steady' and sim_type != 'dynamic':
        raise NameError('Simulation type not recongized: "{0}"'.format(sim_type))
    
    # Set all simulation variables to None, since not all of them will be defined
    max_iter = convergence = relax = dt = dyn_bc = cont_data_dir_path = save_last_jsons = None
    dur_time = 0. # It's better for some variables to be 0
    
    # Continue simulation yes/no
    cont_sim = str(operation['continue simulation from file (yes / no)']).lower()
    if cont_sim != 'yes' and cont_sim != 'no':
        raise NameError('Continuation of dynamic simulation not recongized: {0}'.format(cont_sim))
    
    if cont_sim == 'yes':  # If we're continuing a previous simulation
        cont_dir = str(operation['continuation dir. in results dir.'])
        
        # Check if specified result directory exists
        cont_dir_path = os.path.join(solver_dir, 'results', cont_dir) # Directory of continuation file
        if not os.path.exists(cont_dir_path): # Check if the directory exists
            raise NameError('Continuation of dynamic simulation not possible:\nDirectory {0} not found'.format(cont_dir))

        
        # Get subdirectory
        subdirs = [] # Empty list for directories
        for file in os.listdir(cont_dir_path): # List everything in this directory
            if os.path.isdir(os.path.join(cont_dir_path, file)) and file.startswith('SIM') : # Check if it's a SIM directory
                subdirs.append(file)
        
        # Raise error if empty list (subdirectories not found)
        if not len(subdirs) > 0: # If there is anything in this list
            raise NameError('Continuation of dynamic simulation not possible:\nDirectory {0} has no subdirectories with simulation results'.format(cont_dir))

        subdirs.sort() # Sort this list 
        cont_subdir_path = os.path.join(cont_dir_path, subdirs[-1]) # Take latest entry in list and make path from it 
                            

        # # Check for simulation_setup .json
        # cont_jsons = []
        # for file in os.listdir(cont_subdir_path):
        #     if file.endswith('.json'):
        #         cont_jsons.append(file)
        # if len(cont_jsons) != 1:
        #     raise NameError('Continuation of dynamic simulation not possible:\nsimulation setup .json file not found in directory {0}'.format(cont_dir))
        # input_json = os.path.join(cont_subdir_path, cont_jsons[0]) # Path to json
        
        # Check for sim_data directory
        cont_data_dir_path = os.path.join(cont_subdir_path, 'sim_data')
        if not os.path.exists(cont_data_dir_path):
            raise NameError('Continuation of dynamic simulation not possible:\n"sim_data" directory not found in directory {0}'.format(cont_dir))

        # Get the latest sim_data directory
        dirs = [] # Empty list for directories
        for file in os.listdir(cont_subdir_path):
            if os.path.isdir(os.path.join(cont_subdir_path, file)) and file.startswith('sim_data'): # Get a list of all directories that start with sim_data
                 dirs.append(file)
        dirs.sort(reverse=True) # Sort them
        cont_data_dir_path = os.path.join(cont_subdir_path, dirs[0]) # Update path for continuation data directory 
        
        # CHeck for simulation_info.json in the latest sim_data directory
        if not os.path.exists( os.path.join(cont_data_dir_path, 'simulation_info.json') ):
            raise NameError('Continuation of dynamic simulation not possible:\n"simulation_info.json" not found in directory {0}'.format(dirs[0]))
        
        # Check if there are files with t_ prefix
        if not any(file.startswith('t_') for file in os.listdir(cont_data_dir_path)):
            raise NameError('Continuation of dynamic simulation not possible:\nNo "t_" files found in directory {0}'.format(dirs[0]))

    # if sim_type == 'steady': 
    params = operation['steady simulation parameters']
    max_iter = int(params['maximum iterations'])
    convergence = float(params['convergence criteria'])
    relax = float(params['underrelaxation factor'])
    
    # if sim_type == 'dynamic':
    params = operation['dynamic simulation parameters']
    dynsim_converge_first = str(params['run a steady simulation first (yes / no)']).lower()
    if dynsim_converge_first != 'yes' and dynsim_converge_first != 'no':
        raise NameError('Unrecognized input given in "start with steady converged solution"')
        
    dynsim_cont_convergence = str(params['continue to dynamic simulation if convergence is not achieved (yes / no)']).lower()
    if dynsim_cont_convergence != 'yes' and dynsim_cont_convergence != 'no':
        raise NameError('Unrecognized input given in "continue to dynamic simulation if convergence is not achieved"')
    
    dur_time = float(params['simulated time duration [s]'])
    dt = float(params['timestep size [s]'])
    
    dyn_bc = str(params['dynamic boundary conditions (yes / no)']).lower()
    if dyn_bc != 'yes' and dyn_bc != 'no':
        raise NameError('Dynamic boundary condition not recongized: {0}'.format(dyn_bc))
    
    
    # --- Save file parameters
    s_only_terminal = str(saving['show results only in terminal (yes / no)']).lower()
    if s_only_terminal != 'yes' and s_only_terminal != 'no':
        s_only_terminal = 'no'
        print('Showing incorrectly defined (show results only in terminal), saving them anyways...')
        
        
    # --- When to save and print during the simulations
    # if sim_type == 'dynamic':
    dyn_save_every = saving['dynamic simulation']['save .json every']
    freq_unit = str(saving['dynamic simulation']['frequency saving unit (seconds / timesteps)']).lower()

    if freq_unit == 'seconds': # Redefine save_every if given with seconds
        dyn_save_every = round(dyn_save_every / dt)
    elif freq_unit == 'timesteps': # 
        dyn_save_every = round(dyn_save_every) # Make it a round integer if timesteps 
    else: # Otherwise incorrectly defined, raise error
        raise NameError('Frequency saving unit not correctly defined: {0}'.format(freq_unit)) 
    
    dyn_write_every = int(saving['dynamic simulation']['write in terminal every x timesteps'])
        
    # elif sim_type == 'steady':
    steady_save_every = int(saving['steady simulation']['save .json every x iterations'])
    steady_write_every = int(saving['steady simulation']['write in terminal every x iterations'])
    save_last_jsons = saving['steady simulation']['keep last x .json files (int / all)']
    
    try: 
        save_last_jsons = int(save_last_jsons)
    except:
        save_last_jsons = str(save_last_jsons).lower()
        if save_last_jsons != 'all':
            raise NameError('.json saving frequency incorrectly defined: {0}'.format(save_last_jsons)) 
        
    # See whether we're only displaying in terminal    
    if s_only_terminal == 'yes': # set all saving parameters to 'no' if we only want them to show up in terminal (e.g. for code testing )
        s_dir_name = ""
        s_timestamp = s_json_out = s_log = s_files_in = 'no'
        
    
    else: # If we want to save results 
        
        # Define continuation directory name
        if cont_sim == 'yes':
            s_dir_name = cont_dir # We already defined a name if we're continuing a simulation
        else:
            s_dir_name = str(saving['new results directory name']) # Read user defined name if we're starting a new sim
        
        s_timestamp = str(saving['timestamp in directory name (yes / no)']).lower()
        if s_timestamp != 'yes' and s_timestamp != 'no':
            s_timestamp = 'no'
            print('Timestamp choice incorrectly defined (timestamp in directory name), omitting timestamp...')
        
        s_json_out = str(saving['save simulation output .json files (yes / no)']).lower()
        if s_json_out != 'yes' and s_json_out != 'no':
            s_json_out = 'yes'
            print('.json file saving choice incorrectly defined (save .json simulation file), saving it anyways...')
        
        s_log = str(saving['save log (yes / no)']).lower()
        if s_log != 'yes' and s_log != 'no':
            s_log = 'yes'
            print('Log saving incorrectly defined (save log), saving it anyways...')
    
        s_files_in = str(saving['save input files  (yes / no)']).lower()
        if s_files_in != 'yes' and s_files_in != 'no':
            s_files_in = 'yes'
            print('Input file saving incorrectly defined (save input files), saving it anyways...')
            
    return_list = [cat_shape, cat_dimensions, cat_BET_area, known_cat_density, rho_cat, rho_cat_bulk, cat_composition, \
                   n_tubes, tube_l, tube_d,\
                   ax_cells, rad_cells, adv_scheme, diff_scheme, CFL, partP_limit, Ci_limit,\
                   SC_ratio, p_ref, p_set_pos, inlet_gas_T, init_reactor_T, wall_T_profile, W_F,\
                   sim_type, max_iter, convergence, relax, dt, dur_time, dyn_bc, cont_sim, cont_data_dir_path,\
                   s_dir_name, s_only_terminal, steady_save_every, dyn_save_every, dynsim_converge_first, dynsim_cont_convergence,\
                   steady_write_every, dyn_write_every, save_last_jsons, s_timestamp, s_json_out, s_log, s_files_in]
    
    
    return return_list



def check_used_dynamic_BCs(input_json_fname, p_ref_pos, colw):
    """
    Print used boundary conditions

    Parameters
    ----------
    input_json_fname : str
        Input .json file name
    p_ref_pos : str
        Reference pressure setting position (inlet / outlet)
    colw : 
        Column width
    

    Raises
    ------
    NameError
        

    Returns
    -------
    None.

    """


    colw = int(colw/2)



    # Read boundary conditions from file 
    dyn_BC = json.load(open(input_json_fname))['dynamic boundary conditions']
    yn_list = ['yes', 'no'] # List of yes no choices

    # List of parameters to read from dynamic boundary condition dictionary
    read_list = [ 
        'use dynamic wall temperature profile (yes / no)', 
        'use dynamic inlet temperature (yes / no)',
        'use dynamic inlet mass flow (yes / no)',
        'use dynamic pressure (yes / no)'
        ]


    bc_name_list = [
        'Wall temperature profile',
        'Inlet gas temperature',
        'Inlet mass flow',
        'Pressure at {0}'.format(p_ref_pos.upper())
        ]

    bc_units = ['[C]', '[C]', '[kg s mol-1]', '[bar]']

    value_keys = [
        'reactor wall temperature profile in time (t[s], T_profile[C])',
        'inlet gas temperature in time (t[s], T_in[C])',
        'inlet mass flow in time (t[s], W_cat/F_CH3OH[kg s mol-1])',
        'pressure in time (t[s], p[bar])'
        ]


    for pos in range(len(read_list)):
        condition = dyn_BC[read_list[pos]].lower()
        
        if condition not in yn_list:
            raise NameError('Choice in "dynamic boundary conditions" not recognized for: "{0}"'.format(read_list[pos]))
            
        if condition == 'yes':
            if bc_name_list[pos] != 'Wall temperature profile':
                array = np.asarray(dyn_BC[value_keys[pos]])
                times = array[:,0]
                values = array[:,1]
                print('{0}:'.format(bc_name_list[pos]))
                
            else: 
                Twall_tuple = dyn_BC[value_keys[pos]]
                values = np.asarray([row[1] for row in Twall_tuple])
                times = np.asarray([row[0] for row in Twall_tuple])
                rp_coords = np.asarray(dyn_BC['reactor wall temperature profile z relative positions'])
                print('{0}: \n(relative wall profile positions: {1})'.format(bc_name_list[pos], rp_coords))
                
            print(('Time [s]').ljust(int(colw)) + 'value/profile {0}'.format(bc_units[pos]))        
        
            for i in range(len(times)):
                print( ('\t{0}'.format(times[i])).ljust(colw) + '{0}'.format(values[i]) )
        
            print('\n')

    return



def read_and_set_BCs(input_json_fname,dyn_bc, Tw_in_const, T_in_const, WF_in_const, p_ref_const, z_cell_centers, l_tube):
    """
    Read boundary conditions and define time dependant functions of boundary conditions 

    Parameters
    ----------
    input_json_fname : str
        Input .json file name
    dyn_bc : string
        (yes / no) Use dynamic boundary conditions or not 
    Tw_const : arrray
        [T] .json defined constant wall temperature profile
    Tw_in_const : float
        [T] .json defined constant inlet gas temperature 
    WF_in_const : float
        [kg s mol-1] Constant inlet methanol molar flow rate 
    p_ref_const : float
        [Pa] Constant reference pressure 
    z_cell_centers : array
        [m] Distances of axial 
    l_tube : float
        [m] Reactor tube length

    Raises
    ------
    ValueError
        

    Returns
    -------
    T_wall_func, T_in_func, WF_in_func, p_in_func : functions
        Interpolation functions for wall temperature profile, inlet gas temperature, inlet methanol feed, inlet gas pressure

    """
    # Define relative cell centers (from 0 to 1) that will be used to interpolate 
    rel_zcell_centers = z_cell_centers / l_tube
    
    if dyn_bc == 'no': # If not using dynamic BCs, just set all functions to return a steady value
        
        # --- Wall temperature profile function
        # Make a function that just returns the set temperature from .json
        T_wall_func = lambda t : Tw_in_const
        
        # --- Inlet temperature function
        T_in_func = lambda t : T_in_const # Return from .json
        
        # --- Methanol inlet molar flow rate
        WF_in_func = lambda t : WF_in_const # Return from .json
        
        # --- Inlet pressure function
        p_ref_func = lambda t : p_ref_const
        
    else:

        
        # Read boundary conditions from file 
        dyn_BC = json.load(open(input_json_fname))['dynamic boundary conditions']
        yn_list = ['yes', 'no'] # List of yes no choices
        
        
        
        
        # --- Wall temperature time profile
        condition = dyn_BC['use dynamic wall temperature profile (yes / no)'].lower()
        if condition not in yn_list: 
            raise NameError('Dynamic boundary conditions: dynamic wall temperature choice not recognized')
        
        
        if condition == 'no': # If we're not using this dynamic BC
            # Make a function that just returns the set temperature from .json
            T_wall_func = lambda t : Tw_in_const
            
        else: 
            # - read values from dictionary
            # Get axial relative profile
            Tw_ax_profile = np.asarray(dyn_BC['reactor wall temperature profile z relative positions'])
            # Tuple containing times(floats) and Twall profiles at times (tuples)
            Twall_tuple = dyn_BC['reactor wall temperature profile in time (t[s], T_profile[C])']
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
                T_wall_func = lambda t : Tw_profile
            
            else: # If array has more than 1 temporal point, we have to 
                # Reshape Tw - "rotate" matrix 90 degrees 
                # In reshaped matrix each row is evolution of temperature in time for a single point
                Tw = np.ravel(Tw, order='F').reshape(len(Tw[0]), len(Tw)) 
                
                def T_wall_func(t):
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
        
        
        
        
        
        
        # --- Inlet temperature
        # Get condition first
        condition = dyn_BC['use dynamic inlet temperature (yes / no)'].lower()
        if condition not in yn_list: 
            raise NameError('Dynamic boundary conditions: dynamic inlet temperature choice not recognized')
        
        if condition == 'no': # If we're not using this dynamic BC
            # Make a function that just returns the set temperature from .json
            T_in_func = lambda t : T_in_const
            
        else: 
            # - Otherwise read values from dictionary
            # Matrix containint times and temperatures
            Tin_matrix = np.asarray(dyn_BC['inlet gas temperature in time (t[s], T_in[C])'])
            # Extract respective arrays from the matrix
            T_in = Tin_matrix[:,1]
            time_T_in = Tin_matrix[:,0]
        
            if len(T_in) == 0 or len(time_T_in) == 0:
                raise ValueError('In "dynamic boundary conditions", there are empty lists in temperature inlet profiles')
                
            if len(T_in) == 1: # If one element in list 
                T_in_func = lambda t : T_in[0] # Dont bother with interpolation, just return that value
         
            else: # If multiple values in list
                    # - First check some things 
                    if len(time_T_in) != len(T_in): # Amount of inlet temperatures should match number of time points
                        raise ValueError('In "dynamic boundary conditions", length of t_T_in and T_in does not match!')
                        
                    if not ( sorted(time_T_in) == time_T_in).all(): # Time should be given in ascending order
                        raise ValueError('In "dynamic boundary conditions", t_T_in should be given in ascending order')
                        
                    def T_in_func(t): 
                        """
                        Interpolates inlet gas temperature in time
            
                        Parameters
                        ----------
                        t : float
                            [s] time
            
                        Returns
                        -------
                        T_inlet : float
                            Inlet temperature at time t
                        """
                        
                        T_inlet = np.interp(t, time_T_in, T_in)
                        
                        return T_inlet
                    
                    
                    
        
        # --- Inlet feed 
        # Get condition first
        condition = dyn_BC['use dynamic inlet mass flow (yes / no)'].lower()
        if condition not in yn_list: 
            raise NameError('Dynamic boundary conditions: dynamic inlet mass flow choice not recognized')
        
        if condition == 'no': # If we're not using this dynamic BC
            # Make a function that just returns the set temperature from .json
            WF_in_func = lambda t : WF_in_const 
        
        else: 
            # - Otherwise read values from dictionary
            # Matrix containint times and temperatures
            WFin_matrix = np.asarray(dyn_BC['inlet mass flow in time (t[s], W_cat/F_CH3OH[kg s mol-1])'])
            # Extract respective arrays from the matrix
            WF_in = WFin_matrix[:,1]
            time_WF_in = WFin_matrix[:,0]
            
            if len(WF_in) == 0 or len(time_WF_in) == 0: # If nothing in list(s) 
                raise ValueError('In "dynamic boundary conditions", there are empty lists in inlet mass flow profiles')
            
            if len(WF_in) == 1: # If one element in list 
                WF_in_func = lambda t : WF_in[0] # Dont bother with interpolation, just return that value
                
            else: # If multiple values in list
                    # - First check some things 
                    if len(time_WF_in) != len(WF_in): # Amount of inlet feeds points should match number of time points
                        raise ValueError('In "dynamic boundary conditions", length of t_WF_in and WF_in does not match!')
                        
                    if not ( sorted(time_WF_in) == time_WF_in).all(): # Time should be given in ascending order
                        raise ValueError('In "dynamic boundary conditions", t_WF_in should be given in ascending order')
                        
                    def WF_in_func(t): 
                        """
                        Interpolates inlet gas feed (W/F_CH3OH) in time
            
                        Parameters
                        ----------
                        t : float
                            [s] time
            
                        Returns
                        -------
                        WF_inlet : float
                            Inlet feed at time t
                        """
                        
                        WF_inlet = np.interp(t, time_WF_in, WF_in)
                        
                        return WF_inlet
        
        
        
        # --- Pressure time profile
        
        # Get condition first
        condition = dyn_BC['use dynamic pressure (yes / no)'].lower()
        if condition not in yn_list: 
            raise NameError('Dynamic boundary conditions: dynamic pressure choice not recognized')
        
        if condition == 'no': # If we're not using this dynamic BC
            # Make a function that just returns the set temperature from .json
            p_ref_func = lambda t : p_ref_const
        
        else: 
             # - Otherwise read values from dictionary
             # Matrix containint times and temperatures
             p_matrix = np.asarray(dyn_BC['pressure in time (t[s], p[bar])'])
             # Extract respective arrays from the matrix
             p_ref = p_matrix[:,1]
             time_p_ref = p_matrix[:,0]
             
             if len(p_ref) == 0 or len(time_p_ref) == 0: # If nothing in list(s) 
                 raise ValueError('In "dynamic boundary conditions", there are empty lists in pressure profiles')
        
             # - Make function
             if len(p_ref) == 1: # If one element in list 
                 p_ref_func = lambda t : p_ref[0] * 1e5 # Dont bother with interpolation, convert to Pa and return that value
                
             else: # If multiple values in list
                    # - First check some things 
                    if len(time_p_ref) != len(p_ref): # Amount of inlet feeds points should match number of time points
                        raise ValueError('In "dynamic boundary conditions", length of t_p_ref and p_ref does not match!')
                            
                    if not ( sorted(time_p_ref) == time_p_ref).all(): # Time should be given in ascending order
                        raise ValueError('In "dynamic boundary conditions", t_p_ref should be given in ascending order')
                        
                    def p_ref_func(t): 
                        """
                        Interpolates reference pressure (p_ref) in time
            
                        Parameters
                        ----------
                        t : float
                            [s] time
            
                        Returns
                        -------
                        p_ref : float
                            Reference pressure at time t
                        """
                        
                        p_reference = np.interp(t, time_p_ref, p_ref) * 1e5 # Also, value is in bar so convert to pascal (1e5)
                        
                        return p_reference


    return T_wall_func, T_in_func, WF_in_func, p_ref_func




def read_cont_sim_data(cont_data_dir_path):
    """
    Reads .json files from which we continue the simulation

    Parameters
    ----------
    cont_dir_path : 
        Path to continuation directory
    Returns
    -------
    return_list : list
        List of variables and settings
        
    """
    
    # Load the simulation setup file
    setup_file = json.load(open(os.path.join(cont_data_dir_path, 'simulation_info.json')))
    
    t_files = [] # Empty list for directories
    for file in os.listdir(cont_data_dir_path):
        if  file.startswith('t_'): # Get a list of all t_ files
            t_files.append(file)
    t_files.sort(key=lambda f: float(''.join(filter(str.isdigit, f))), reverse=True) # Sort them
    # Load the last written t_ file
    t_file = json.load(open(os.path.join(cont_data_dir_path, t_files[0]))) 
    
    # --- Read setup file 
    SC_ratio = float(setup_file['S/C'])    # Steam to carbon ratio of inlet gas
    n_tubes = int(setup_file['n tubes'])   # Number of reactor tubes
    l_tube = float(setup_file['tube l'])   # Reactor tube length
    d_tube = float(setup_file['tube r'])*2   # Reactor tube diameter
    
    N = float(setup_file['aspect ratio'])  # Reactor aspect ratio
    epsilon = float(setup_file['porosity']) # Reactor porosity
    
    cells_ax = int(setup_file['n ax cells'])     # Number of axial cells in mesh
    cells_rad = int(setup_file['n rad cells'])   # Number of radial cells in mesh
    
    cat_shape = str(setup_file['cat shape'])             # Catalyst shape
    cat_dimensions = setup_file['cat dimensions']   # Catalys dimensions 
    rho_cat = float(setup_file['cat rho'])                 # Catalyst density
    rho_cat_bulk = float(setup_file['cat rho bulk'])       # Catalyst bulk density
    cat_composition = setup_file['cat composition'] # Catalyst composition 
    cat_cp = float(setup_file['cat cp'])            # Catalyst Cp
    fresh_cat_BET_area = float(setup_file['fresh cat BET'])     # BET area of fresh catalyst 
    
    # Convert possible lists to np.arrays
    
    if type(cat_dimensions) == list:
        cat_dimensions = np.asarray(cat_dimensions)
    else:
        cat_dimensions = float(cat_dimensions)
    
    if type(cat_composition) == list:
        cat_composition = np.asarray(cat_composition)
    else:
        cat_composition = float(cat_composition)
    
    
    # --- Read timestep file
    
    # Simulation time
    t = float(t_file['t'])
    
    # Specie concentration arrays
    field_CH3OH = t_file['CH3OH']
    field_H2O = t_file['H2O']
    field_H2 = t_file['H2']
    field_CO2 = t_file['CO2']
    field_CO = t_file['CO']
    
    # Join them in one 3D array
    field_Ci_n = np.dstack((field_CH3OH, field_H2O, field_H2, field_CO2, field_CO))
    # Shuffle around the axes to match wanted output
    field_Ci_n = np.swapaxes(field_Ci_n, 1, 2)
    field_Ci_n = np.swapaxes(field_Ci_n, 0, 1)
    # This array doesn't contain ghost cells - rememberd
    
    # Read fields of temperature, pressure, velocity, and BET aread
    field_T = np.asarray(t_file['T'])
    field_p = np.asarray(t_file['p'])
    field_v = np.asarray(t_file['v'])
    field_BET = np.asarray(t_file['BET'])
    
    
    # --- Put all variables into a nice list
    return_list = [t, field_Ci_n, field_T, field_p, field_v, field_BET,\
                   SC_ratio, n_tubes, l_tube, d_tube, N, epsilon, cells_ax, cells_rad,\
                   cat_shape, cat_dimensions, rho_cat, rho_cat_bulk, cat_composition, cat_cp, fresh_cat_BET_area]
    
    
    return return_list




def create_result_dir(userdef_dir_name, continued_sim, sim_type, timestamp_choice, timestamp):
    """
    Create directories for results

    Parameters
    ----------
    userdef_dir_name : string
        User defined result directory name
    continued_sim : string
        (yes / no) Are we continuing the simulation from previous file?
    sim_type : string
        (steady / dynamic) Simulation type
    timestamp_choice : string
        (yes / no) Do you want timestamp on your (newly created) directory name?
    timestamp : string
        Timestamp string

    Returns
    -------
    result_subdir_path : string
        String path to specific results subdirectory
    t_data_path : string 
        String path to t_ data directory
    old_sim_data_path : string
        Sttring path to subdirectory of previously continued simulation

    """
    
    # --- Check if results directory exists
    solver_dir = os.getcwd() # Current directory
    all_results_dir = os.path.join(solver_dir, 'results') # Directory of all results
    
    
    # Get a paths to all results directory
    if not os.path.exists(all_results_dir): #check if 'results' folder exists
        os.mkdir(all_results_dir)  # Make 'results' directory if it's not there for some reason
    
    
    # --- Results main directory 
    if continued_sim == 'no': # If we have a new simulation
    
        # Define name of directory
        if userdef_dir_name =="": # If the name is empty, use only timestamp
            result_dir_name = timestamp
        else: # if there is a user defined name, use that
            result_dir_name = userdef_dir_name # If result directory is named by user, use that name
            if timestamp_choice == 'yes': # Append timestamp if defined so by the user
                result_dir_name = result_dir_name + '_' + timestamp
            
        # Make path to currently specified result directory
        result_dir_path = os.path.join(solver_dir, 'results', result_dir_name) # make path to directory of results
        
        # Check if path already exists
        if os.path.exists(result_dir_path): #Check if path exists and change result dir name if it does
            instance = 1 # make counter
          
            while os.path.exists(result_dir_path): # while the paths exist
                results_dir_name_temp = result_dir_name  + '_(' + str(instance) + ')' # keep changing directory name 
                result_dir_path = os.path.join(solver_dir, 'results', results_dir_name_temp) # make path
                instance += 1
                
            result_dir_name = results_dir_name_temp # adopt tested result directory name
            result_dir_path = os.path.join(solver_dir, 'results', result_dir_name) # make path
           
        # Create the directory    
        os.mkdir(result_dir_path) #create the results directory at defined path
        
        # Set simulation start time as 0.0 - this is used in naming the directory
        start_t = 0.0
        sim_number = 0 # also used in naming the subdirectory
        old_sim_data_path = '' # Path to old (continued) simulation data
        
    # --- If we're continuing the simulation
    else: 
        # We've already passed the directory name, so make a path from it
        result_dir_path = os.path.join(solver_dir, 'results', userdef_dir_name) 
        
        # --- Get the latest simulation subdirectory
        subdirs = [] # Empty list for directories
        for file in os.listdir(result_dir_path): # List everything in this directory
            if os.path.isdir(os.path.join(result_dir_path, file)) and file.startswith('SIM') : # Check if it's a SIM directory
                subdirs.append(file)
    
        if len(subdirs) > 0: # If there is anything in this list
            subdirs.sort() # Sort this list 
            latest_dir = subdirs[-1] # Last created directory in this list 
            last_sim_number = int(latest_dir.split('_')[0].split('.')[-1]) # Get just the number
            sim_number = last_sim_number + 1 # Add a number to this
        else: 
            raise NameError('Continuation of dynamic simulation not possible:\nNo SIM subdirectories in {0}'.format(userdef_dir_name)) 
    
        # --- Get the latest simulation time from t_file
        old_sim_data_path = os.path.join(result_dir_path, latest_dir, 'sim_data')
        
        t_files = [] # Empty list for directories
        for file in os.listdir(old_sim_data_path):
            if  file.startswith('t_'): # Get a list of all t_ files
                t_files.append(file)
        t_files.sort(key=lambda f: float(''.join(filter(str.isdigit, f))),reverse=True) # Sort them
        # Load the last written t_ file
        t_file = json.load(open(os.path.join(old_sim_data_path, t_files[0]))) 
        start_t = round(t_file['t'], 5) # Read start time
        
    
    
    # Make a new subdirectory name string
    new_subdir_name = 'SIM.' + str(sim_number).zfill(2) + '_t_' + str(float(start_t)) 
    
    if sim_type == 'steady': # Add .steady to subdir name if current sim is steady state
        new_subdir_name =  new_subdir_name + '.steady'
    
    
    # Make paths
    result_subdir_path = os.path.join(result_dir_path, new_subdir_name) # Result subdirectory 
    sim_data_path = os.path.join(result_subdir_path, 'sim_data')          # data subdirectory
    
    # Make directories
    os.mkdir(result_subdir_path)
    os.mkdir(sim_data_path) 
    
    return result_subdir_path, sim_data_path



def get_old_sim_data_path(userdef_dir_name):
    
    solver_dir = os.getcwd() # Current directory
    result_dir_path = os.path.join(solver_dir, 'results', userdef_dir_name) 
    
    # --- Get the latest simulation subdirectory
    subdirs = [] # Empty list for directories
    for file in os.listdir(result_dir_path): # List everything in this directory
        if os.path.isdir(os.path.join(result_dir_path, file)) and file.startswith('SIM') : # Check if it's a SIM directory
            subdirs.append(file)

    if len(subdirs) > 0: # If there is anything in this list
        subdirs.sort() # Sort this list 
        latest_dir = subdirs[-1] # Last created directory in this list 
    else: 
        raise NameError('Continuation of dynamic simulation not possible:\nNo SIM subdirectories in {0}'.format(userdef_dir_name))
    
    old_sim_data_path = os.path.join(result_dir_path, latest_dir, 'sim_data')
    
    return old_sim_data_path



def open_log(log_choice, result_path, logname, original_stdout):
    """
    Sets whether terminal outputs are saved to a log file or not

    Parameters
    ----------
    log_choice : boolean
        Do you want to write terminal outputs to a log file?
        
    result_path : 
        Path to results directory
        
    logname : 
        Log file name
        
    original_stdout : 
        Original instance of sys.stdout()

    Returns
    -------
    sysexit : lambda
        Custom sysexit function to wrap up log writing and close file

    """
    
    log_path = os.path.join(result_path, logname)
    
    if log_choice == 'yes': #Log saving options
            
        # if __name__ == '__main__': #dont open new files when this is called by other .py files
        
        sys.stdout = open(log_path, 'w') #Write terminal output to file
            
        def sysexit(reason='\nUndefined exit'): # Redefine sys.exit to return the original stdout
            print(reason)
            sys.stdout.close()
            sys.stdout = original_stdout
            sys.exit(reason)
        
    else:  #Use normal sys.exit if log not used
        sysexit = lambda reason='\nUndefined exit' : sys.exit(reason) 
    
    return sysexit




def copy_json(input_json_fname, result_path):
    """
    Copies the user input .json file to the results directory

    Parameters
    ----------
    input_json_fname : str
        Input .json file name
    result_path : 
        Path to the associated results directory
        

    Returns
    -------
    None.

    """
    
    shutil.copy(input_json_fname, result_path)
    
    return





def write_user_inputs(input_json_fname):
    '''
    Parameters
    ----------
    input_json_fname : str
        Input .json file name
        '''
        
    
    text = open(input_json_fname).readlines() # read all lines in .json file
    
    # print some decorations
    print('\n----- .json input file: -----\n')
    
    # Print actual .json lines
    for line in text:
        print(line, end=' ')
    
    # Print decorations
    print('\n\n----- END OF .json FILE -----\n')
    
    return




















''' 
==============================================================================================================================
------------------------------------------------------------------------------------------------------------------------------
--------------------- OUTPUTS & POST-SOLVER
------------------------------------------------------------------------------------------------------------------------------
==============================================================================================================================
'''



def close_log(log_choice, original_stdout):
    """
    Closes the stdout writing and restores original stdout

    Parameters
    ----------
    log_choice : boolean
        Do you want to write terminal outputs to a log file?
        
    original_stdout : 
        Original instance of sys.stdout()

    Returns
    -------
    None.

    """
    
    if log_choice == True:
        sys.stdout.close()
        sys.stdout = original_stdout

    return





def output_write_settings(saving_json_out):
    """
    Define function that either writes the output json file, or does nothing depending on the choice

    Parameters
    ----------
    saving_json_out : 
        Choice for whether or not to save output .json files
        
    Returns
    -------
    write_out_json()
        A function that writes (or doesn't) the simulation output .json file

    """
    
    if saving_json_out == 'yes':
        
        # Some internet magic - class for encoding np.arrays to be suitable for json
        class NumpyArrayEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)
        
        # Define writing function for unsteady simulation
        def write_out_json(data_list_names, data_list, path_saving_data_dir, current_timestep, iteration=None):
            """
            Function that writes the simulation output .json file

            Parameters
            ----------
            data_list_names : list
                A corresponding list of field data names 
            data_list : list
                A list of field variables that we want to save
            path_saving_data_dir : 
                The path to directory where we save the data
            current_timestep : 
                [s] Current timestep used in file naming
            iteration : 
                [-] Current iteration of steady state solver

            Returns
            -------
            ticker : 
                Ticker for when to write out the file 
            """
           
            # Create a dictionary with data list and corresponding names as keys
            my_dict = dict(zip(data_list_names, data_list))
            
            # Make a filename from current time step
            if iteration == None:
                print('Writing .json at timestep:', current_timestep)
                out_filename = 't_' + str(current_timestep) + '.json'
            else: 
                print('Writing .json at iteration:', iteration)
                out_filename = 't_' + str(round(current_timestep, 5)) + '_iteration_' + str(iteration) + '.json'
                
            # Make a path from filename 
            out_file_path = os.path.join(path_saving_data_dir, out_filename)
            
            # Write to json
            with open(out_file_path, "w") as write_file:
                json.dump(my_dict, write_file, cls=NumpyArrayEncoder, indent=4)
             
            
            return 
        
        
        def write_info_json(data_list_names, data_list, path_saving_data_dir):
            
            # Create a dictionary with data list and corresponding names as keys
            my_dict = dict(zip(data_list_names, data_list))
            
            # Make a filename
            out_filename = 'simulation_info.json'
            
            # Make a path from filename 
            out_file_path = os.path.join(path_saving_data_dir, out_filename)
                
            # Write to json
            with open(out_file_path, "w") as write_file:
                json.dump(my_dict, write_file, cls=NumpyArrayEncoder, indent=4)
            
            print('\nWriting simulation_info.json...')
            
            return
    
    
    else: # If we're not saving out files, do nothing
        def write_out_json(data_list_names, data_list, path_saving_data_dir, current_timestep, iteration=None):
            # Do nothing            
            return  
        
        def write_info_json(data_list_names, data_list, path_saving_data_dir):
            # Do nothing
            return
    
    return write_out_json , write_info_json



def steady_sim_cleanup(saving_json_out, path_saving_data_dir, allowed_files = 2):
    """
    Clean up the directory from .json iteration files

    Parameters
    ----------
    saving_json_out : 
        Are we even outputting .jsons in this sim
    path_saving_data_dir : 
        Path to data directory
    allowed_files : optional
        Maximum allowed files left after the simulation. The default is 2.

    Returns
    -------
    None.

    """
    # If there are no jsons outputted, just skip
    if saving_json_out == 'no':
        return
    
    
    if allowed_files == 'all':
        pass # Dont delete anything if we're keeping every file
    else: 
    
        t_files = [] # Empty list for files starting with t_
        for file in os.listdir(path_saving_data_dir):
            if  file.startswith('t_'): # Get a list of all t_ files
                t_files.append(file)
        t_files.sort(key=lambda f: float(''.join(filter(str.isdigit, f))) ) # Sort them
        
        
        if len(t_files) <= allowed_files: # If there are less files than allowed, do nothing
            pass
        else: 
            del t_files[-allowed_files:] # Otherwise remove last x files from the list 
            
            for file in t_files:
                path = os.path.join(path_saving_data_dir, file) # Make path
                os.remove(path) # Delete 
        
        
    
    return










