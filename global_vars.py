 # -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 13:11:43 2024

@author: bgrenko
"""
import numpy as np


class inletClass: # Create an empty class that will contain methods for calculating Twall
    M_i = np.asarray([32.042, 18.01528, 2.01568, 44.01, 28.01])/1000 # [kg mol-1] Molar masses of 5 gases [CH3OH, H2O, H2, CO2, CO]
    pass 

inlet = inletClass() # Define an instance

def set_inlet_flowrate_unit(spec):

    if spec == 0: # W_cat/F_CH3OH [kg s mol-1]  
        def CX(self, flow_rate):
            n_CH3OH_in = self.W_cat / flow_rate # [mol s-1] Get inlet methanol molar
            n_H2O_in = n_CH3OH_in * self.SC_ratio # [mol s-1] Get inlet steam molar
            n_i_in = np.asarray([n_CH3OH_in, n_H2O_in, 0, 0, 0, 0]) # [mol s-1] array of all molar inlets
            
            n_in_mix = sum(n_i_in) # [mol s-1] Total (mixture) molar inlet
            X_i_in = n_i_in / n_in_mix # [-] molar fraction array
            
            return n_i_in, X_i_in
    
    elif spec == 1: # MOLAR flow rate METHANOL [mol s-1]
        def CX(self, flow_rate):
            n_H2O_in = flow_rate * self.SC_ratio # [mol s-1] Get inlet steam molar
            n_i_in = np.asarray([flow_rate, n_H2O_in, 0, 0, 0, 0]) # [mol s-1] array of all molar inlets
            
            n_in_mix = sum(n_i_in) # [mol s-1] Total (mixture) molar inlet
            X_i_in = n_i_in / n_in_mix # [-] molar fraction array
            return n_i_in, X_i_in
        
    elif spec == 2: # MOLAR flow rate METHANOL [kmol h-1]
        def CX(self, flow_rate):
            flow_rate *= (1/3.6) # Convert to [mol s-1]
            
            n_H2O_in = flow_rate * self.SC_ratio # [mol s-1] Get inlet steam molar
            n_i_in = np.asarray([flow_rate, n_H2O_in, 0, 0, 0, 0]) # [mol s-1] array of all molar inlets
            
            n_in_mix = sum(n_i_in) # [mol s-1] Total (mixture) molar inlet
            X_i_in = n_i_in / n_in_mix # [-] molar fraction array
            return n_i_in, X_i_in
        
    elif spec == 3: # MOLAR flow rate MIXTURE [mol s-1]
        def CX(self, flow_rate):
            X_i_in = np.asarray([ 1/(self.SC_ratio+1), self.SC_ratio/(self.SC_ratio + 1), 0, 0, 0, 0]) # Get ratios from steam to carbon ratio
            n_i_in = flow_rate * X_i_in # [mol s-1] Get individual molar flow from X_i
            return n_i_in, X_i_in
        
    elif spec == 4: # MOLAR flow rate MIXTURE [kmol h-1]
        def CX(self, flow_rate):
            flow_rate *= (1/3.6) # First convert to [mol s-1]
            X_i_in = np.asarray([ 1/(self.SC_ratio+1), self.SC_ratio/(self.SC_ratio + 1), 0, 0, 0, 0]) # Get ratios from steam to carbon ratio
            n_i_in = flow_rate * X_i_in # [mol s-1] Get individual molar flow from X_i
            return n_i_in, X_i_in
        
    elif spec == 5: # MASS flow rate METHANOL [kg s-1]
        def CX(self, flow_rate):
            X_i_in = np.asarray([ 1/(self.SC_ratio+1), self.SC_ratio/(self.SC_ratio + 1), 0, 0, 0, 0]) # Get ratios from steam to carbon ratio
            n_CH3OH_in = flow_rate / self.M_i[0] # [mol s-1] Molar flow of methanol
            n_i_in = np.asarray([n_CH3OH_in, n_CH3OH_in*self.SC_ratio, 0, 0, 0, 0]) # Molar flow of species
            return n_i_in, X_i_in
    
    elif spec == 6: # MASS flow rate METHANOL [kg h-1]
        def CX(self, flow_rate):
            flow_rate *= (1/3600) # First convert to [kg s-1]
            X_i_in = np.asarray([ 1/(self.SC_ratio+1), self.SC_ratio/(self.SC_ratio + 1), 0, 0, 0, 0]) # Get ratios from steam to carbon ratio
            n_CH3OH_in = flow_rate / self.M_i[0] # [mol s-1] Molar flow of methanol
            n_i_in = np.asarray([n_CH3OH_in, n_CH3OH_in*self.SC_ratio, 0, 0, 0, 0]) # Molar flow of species
            return n_i_in, X_i_in
        
    elif spec == 7: # MASS flow rate MIXTURE [kg s-1]
        def CX(self, flow_rate):
            X_i_in = np.asarray([ 1/(self.SC_ratio+1), self.SC_ratio/(self.SC_ratio + 1), 0, 0, 0, 0]) # Get ratios from steam to carbon ratio
            M_mix = sum(self.M_i * X_i_in) # [kg mol-1] Get mixture molecular mass
            n_mix = flow_rate / M_mix # [mol s-1] Mixture molar flow rate
            n_i_in = n_mix * X_i_in # [mol s-1] Individual molar flow rates
            return n_i_in, X_i_in
        
    elif spec == 8: # MASS flow rate MIXTURE [kg h-1]
        def CX(self, flow_rate):
            flow_rate *= (1/3600) # First convert to [kg s-1]
            X_i_in = np.asarray([ 1/(self.SC_ratio+1), self.SC_ratio/(self.SC_ratio+1), 0, 0, 0, 0]) # Get ratios from steam to carbon ratio
            M_mix = sum(self.M_i * X_i_in) # [kg mol-1] Get mixture molecular mass
            n_mix = flow_rate / M_mix # [mol s-1] Mixture molar flow rate
            n_i_in = n_mix * X_i_in # [mol s-1] Individual molar flow rates
            return n_i_in, X_i_in
        
    elif spec == 100: # MOLAR flow rate MIXTURE [mol s-1]
         def CX(self, flow_rate):
             X_i_in = self.my_X_i # Ratios are custom set in the class already
             n_i_in = flow_rate * X_i_in # [mol s-1] Get individual molar flow from X_i
             return n_i_in, X_i_in
         
    elif spec == 101: # MASS flow rate MIXTURE [kg s-1]
         def CX(self, flow_rate):
             X_i_in = self.my_X_i # Ratios are custom set in the class already
             M_mix = sum(self.M_i * X_i_in) # [kg mol-1] Get mixture molecular mass
             n_mix = flow_rate / M_mix # [mol s-1] Mixture molar flow rate
             n_i_in = n_mix * X_i_in # [mol s-1] Individual molar flow rates
             return n_i_in, X_i_in
            
    else:
        def CX(self, flow_rate):
            raise ValueError('Selection for inlet flow rate unit not recognized (' + str(flow_rate) + ') - Check definition in input .json file')
            return 
    
    setattr(inletClass, 'CX', CX) # Add the defined function to the class
    return 




# Flux limiters 
class FLClass: # Create an empty class that will contain methods for slope limiting
    pass 

flux = FLClass() # Define an instance

def set_flux_limiter(spec):
    spec = spec.lower()
    
    # Define flux limiter function
    if spec == 'none':
        def limiter(self, r_left, r_right):
            return 1,1
        
    elif spec == 'minmod':
        def limiter(self, r_left, r_right):
            phi_r_left = np.maximum(0, np.minimum(1, r_left))
            phi_r_right = np.maximum(0, np.minimum(1, r_right))
            return phi_r_left, phi_r_right
        
    elif spec == 'superbee':
        def limiter(self, r_left, r_right):
            phi_r_left = np.clip(np.maximum(np.minimum(2*r_left, 1), np.minimum(r_left,2)), 0, None)
            phi_r_right = np.clip(np.maximum(np.minimum(2*r_right, 1), np.minimum(r_right,2)), 0, None)
            return phi_r_left, phi_r_right
        
    elif spec == 'koren':
        def limiter(self, r_left, r_right):
            phi_r_left = np.maximum(0, np.minimum(2*r_left, np.minimum((1+2*r_left)/3, 2)))
            phi_r_right = np.maximum(0, np.minimum(2*r_right, np.minimum((1+2*r_right)/3, 2)))
            return phi_r_left, phi_r_right
    
    elif spec == 'vanleer':
        def limiter(self, r_left, r_right):
            phi_r_left = (r_left + np.abs(r_left)) / (1 + np.abs(r_left))
            phi_r_right = (r_right + np.abs(r_right)) / (1 + np.abs(r_right))
            return phi_r_left, phi_r_right
        
    # elif spec == 'test':
    #     def limiter(self, r_left, r_right):
    #         return phi_r_left, phi_r_right
    
    else:
        raise ValueError('Selected flux limiter not recognized (' + str(spec) + ') - Check definition in input .json file')
            
    # Add the defined functions to the class    
    setattr(FLClass, 'limiter', limiter) 
        
    return 
    
    
    
def set_ratio_of_gradients(scheme):
    # Define function for smoothness parameter (r) 
    if scheme == 'upwind_1o':
        def ratio_of_gradients(self, P, W, WW, E):
            return 1,1
    
    elif scheme == 'laxwendroff':
        def ratio_of_gradients(self, P, W, WW, E):
            r_den = (P - W) 
            r_den[r_den==0] = 1 # Denominator cant be zero
            r_left = (W - WW) /r_den
            
            r_den = (E - P) 
            r_den[r_den==0] = 1 # Denominator cant be zero
            r_right = (P - W) /r_den
            return r_left, r_right
        
    elif scheme in ['upwind_2o', 'beamwarming']: # Upwind schemes require fluxes at different faces 
        def ratio_of_gradients(self, P, W, WW, E):
            # WWW = np.roll(WW, 1, 2) # Make a WWW point
            # WWW[:, :, 0] = WW[:, :, 0] # Set inlet 
            # r_den = (W - WW) 
            # r_den[r_den==0] = 1 # Denominator cant be zero
            # r_left = (WW - WWW) /r_den
            
            r_den = (P - W) 
            r_den[r_den==0] = 1 # Denominator cant be zero
            r_right = (W - WW) /r_den
            
            r_left = r_right
            
            return r_left, r_right
    
    else:
        raise ValueError('Selected numerical scheme not recognized (' + str(scheme) + ') - Check definition in input .json file')
    
    # Add the defined functions to the class    
    setattr(FLClass, 'ratio_of_gradients', ratio_of_gradients) 
    
    return 




# Pressure propagation
# Choice of set pressure at inlet or outlet decides whether we populate the pressure field forward or backward
class pressureControlPoint: # Create an empty class that will contain functions for populating the pressure field
    pass 

pressureSetPoint = pressureControlPoint() # Define an instance

def set_pressure_calculation_scheme(setpoint):
    # This is a function which defines whether we propagate pressure forward or backwards 
    
    if setpoint == 'inlet': 
        # Forward propagation from the inlet setpoint 
        def populate_pressure_field(self, dp_array, dz, z_centers_array, p_ref, cell_face_z_positions):
            '''
            dp_array : array of pressure drops
            dz : value of cell dz
            z_centers_array : array of z center coordinates
            p_ref : reference pressure (inlet)
            cell_face_z_positions : array of z cell face coordinates
            '''
            
            # Array of cumulative pressure drop at cell faces - at the start only populated with zero
            dpdz_c_face = np.asarray([0])
            
            for dp in dp_array: # calculate the cumulative pressure drop along the z axis and 
                dpdz_c_face = np.append(dpdz_c_face, dpdz_c_face[-1] + dp*dz)
                
            # interpolate between the edges to get center cumulative pressure values
            dpdz_c_centers = np.interp(z_centers_array, cell_face_z_positions, dpdz_c_face)
            # Add dp values to static field
            p_array = p_ref + dpdz_c_centers
            
            # Inlet and outlet pressure values are first and last items in p_edges array
            # Inlet pressure is unchanged, 
            p_inlet = p_ref 
            p_outlet = p_ref + dpdz_c_face[-1]
            
            return p_inlet, p_outlet, p_array
        
    elif setpoint == 'outlet':
        # Forward propagation from the outlet setpoint backwards
        def populate_pressure_field(self, dp_array, dz, z_centers_array, p_ref, cell_face_z_positions):
            
            # Array of cumulative pressure drop at cell faces - at the start only populated with zero
            dpdz_c_face = np.asarray([0])
            
            for dp in np.flip(dp_array): # calculate the cumulative pressure drop along the z axis and 
                # Array is flipped because we populate it backwards (outlet to inlet)   
                dpdz_c_face = np.append(dpdz_c_face, dpdz_c_face[-1] - dp*dz)
            
            dpdz_c_face = np.flip(dpdz_c_face)  # Flip the array again
                
            # interpolate between the edges to get center cumulative pressure values
            dpdz_c_centers = np.interp(z_centers_array, cell_face_z_positions, dpdz_c_face)
            # Add dp values to static field
            p_array = p_ref + dpdz_c_centers
            
            # Inlet and outlet pressure values are first and last items in p_edges array
            # Outlet pressure is unchanged, inlet includes cumulative pressure build
            p_outlet = p_ref 
            p_inlet = p_ref + dpdz_c_face[0]
            
            return p_inlet, p_outlet, p_array
    else: 
        raise ValueError('Selected pressure control point not recognized (' + str(setpoint) + ') - Check definition in input .json file')
        
    # Add the defined function to the class    
    setattr(pressureControlPoint, 'populate_pressure_field', populate_pressure_field) 
    
    return











