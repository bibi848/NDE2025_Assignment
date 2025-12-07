# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import ctypes
import json

"""
NOTE the files 
    fn_simulate_data_ex5_v2.dll
    fn_simulate_data_weld_v5.dll
must be present in the same folder for this function to work.

"""
class emxArray_real_T(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_double)),
                ("size", ctypes.POINTER(ctypes.c_int)),
                ("allocatedSize", ctypes.c_int),
                ("numDimensions", ctypes.c_int),
                ("canFreeData", ctypes.c_bool)]
    
class emxArray_char_T(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_char)),
                ("size", ctypes.POINTER(ctypes.c_int)),
                ("allocatedSize", ctypes.c_int),
                ("numDimensions", ctypes.c_int),
                ("canFreeData", ctypes.c_bool)]


#------------------------------------------------------------------------------
#Python wrapper function fn_simulate_data_ex5_v2

dll = np.ctypeslib.load_library ('fn_simulate_data_ex5_v2',  os.path.dirname(__file__))
ct_simulate_data_ex5_v2 = dll.fn_simulate_data_ex5_v2
ct_simulate_data_ex5_v2.argtypes = [ctypes.c_int,                       #no_elements (input)
                                    ctypes.c_double,                    #element_pitch (input)
                                    ctypes.c_double,                    #element_width (input)
                                    ctypes.c_double,                    #centre_freq (input)
                                    ctypes.c_int,                       #time_pts (input)
                                    ctypes.POINTER(emxArray_real_T),    #time (output)
                                    ctypes.POINTER(emxArray_real_T),    #fmc_data (output)
                                    ctypes.POINTER(ctypes.c_double),    #element_positions (output)
                                    ctypes.POINTER(ctypes.c_int)]       #element_positions_size (output)
ct_simulate_data_ex5_v2.restype = None

def fn_simulate_data_ex5_v2(no_elements, 
                            element_pitch, 
                            element_width, 
                            centre_freq, 
                            time_pts):
   
    """
    Simulates the ultrasonic data for exercise 5
    
    The input parameters should be self explanatory from their names. See the 
    exercise worksheet for more details. The function will simulate FMC data 
    based on these parameters and returns:
        fmc_data_arr - a 3D (m x m x n) array of data with dimensions 
            corresponding to transmitter, receiver, and time
        time_arr - an n-element vector of times associated with time-dimension
            of fmc_data_arr
        element_positions - an m-element vector of the positions of element 
            in the array used to simulate fmc_data_arra
    """
   #Prepare time ouput
    time = emxArray_real_T()
    time.size = (ctypes.c_int * 2)(1, 1)
    time.data = ctypes.POINTER(ctypes.c_double)()
    time.allocatedSize = 0
    time.numDimensions = 2
    time.canFreeData = False
    
    #Prepare fmc_data output
    fmc_data = emxArray_real_T()
    fmc_data.size = (ctypes.c_int * 3)(1, 1, 1)
    fmc_data.data = ctypes.POINTER(ctypes.c_double)()
    fmc_data.allocatedSize = 0
    fmc_data.numDimensions = 3
    fmc_data.canFreeData = False

    #Prepare element_positions output
    element_positions = np.zeros(no_elements)

    #Call the function in the DLL
    ct_simulate_data_ex5_v2(no_elements, 
                            element_pitch, 
                            element_width, 
                            centre_freq, 
                            time_pts,
                            ctypes.byref(time), 
                            ctypes.byref(fmc_data),
                            element_positions.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            (ctypes.c_int * 2)(*element_positions.shape))
    
   
    fmc_data_arr = np.ctypeslib.as_array(fmc_data.data, shape=(fmc_data.size[2], fmc_data.size[1], fmc_data.size[0]))
    time_arr  = np.ctypeslib.as_array(time.data, shape=(time.size[0],))
    
    return (fmc_data_arr, time_arr, element_positions)

#------------------------------------------------------------------------------
# Load the DLL and define the function prototype for fn_simulate_data_weld_v5
dll = np.ctypeslib.load_library ('fn_simulate_data_weld_v5',  os.path.dirname(__file__))
ct_simulate_data_weld_v5 = dll.fn_simulate_data_weld_v5
ct_simulate_data_weld_v5.argtypes = [ctypes.POINTER(emxArray_char_T),   #sample (input)
                                     ctypes.c_double,                   #scan_position (input)
                                     ctypes.c_double,                   #no_elements (input)
                                     ctypes.c_double,                   #element_pitch (input)
                                     ctypes.c_double,                   #element_width (input)
                                     ctypes.c_double,                   #first_element_position(input)
                                     ctypes.c_double,                   #centre_freq (input)
                                     ctypes.POINTER(emxArray_real_T),   #time (output)
                                     ctypes.POINTER(emxArray_real_T),   #fmc_data (output)
                                     ctypes.POINTER(ctypes.c_double),   #element_positions (output)
                                     ctypes.POINTER(ctypes.c_int),      #element_positions_size (output)
                                     ctypes.POINTER(emxArray_real_T)]   #points (output)
ct_simulate_data_weld_v5.restype = None


#Python wrapper function fn_simulate_data_weld_v5
def fn_simulate_data_weld_v5(sample, 
                             scan_position, 
                             no_elements, 
                             element_pitch, 
                             element_width, 
                             first_element_position, 
                             centre_freq):
    """Simulates the ultrasonic data for the summative coursework
    
    The input parameter sample is a string containing a code that determines
    what data is simulated by the function. For the blind trial data the string
    should be your University of Bristol username, e.g. ab12345. The other 
    input parameters should be self explanatory from their names. See the 
    exercise worksheet for more details. The function will simulate FMC data 
    based on these parameters and returns:
        fmc_data_arr - a 3D (m x m x n) array of data with dimensions 
            corresponding to transmitter, receiver, and time
        time_arr - an n-element vector of times associated with time-dimension
            of fmc_data_arr
        element_positions - an m-element vector of the positions of element 
            in the array used to simulate fmc_data_arra
    """

    #Deal with sample_in input string
    sample_bytes = sample.encode()
    sample_str = emxArray_char_T()
    sample_str.size = (ctypes.c_int * 2)(1, len(sample_bytes))
    sample_str.data = ctypes.cast(sample_bytes, ctypes.POINTER(ctypes.c_char))
    sample_str.allocatedSize = len(sample_bytes)
    sample_str.numDimensions = 2
    sample_str.canFreeData = False
    
    #Prepare time ouput
    time = emxArray_real_T()
    time.size = (ctypes.c_int * 2)(1, 1)
    time.data = ctypes.POINTER(ctypes.c_double)()
    time.allocatedSize = 0
    time.numDimensions = 2
    time.canFreeData = False
    
    #Prepare fmc_data output
    fmc_data = emxArray_real_T()
    fmc_data.size = (ctypes.c_int * 3)(1, 1, 1)
    fmc_data.data = ctypes.POINTER(ctypes.c_double)()
    fmc_data.allocatedSize = 0
    fmc_data.numDimensions = 3
    fmc_data.canFreeData = False
    
    #Prepare points output
    points = emxArray_real_T()
    points.size = (ctypes.c_int * 2)(1, 1)
    points.data = ctypes.POINTER(ctypes.c_double)()
    points.allocatedSize = 0
    points.numDimensions = 2
    points.canFreeData = False
    
    #Prepare element_positions output
    element_positions = np.zeros(no_elements)

    #Call the function in the DLL
    ct_simulate_data_weld_v5(ctypes.byref(sample_str), 
                             scan_position, 
                             no_elements, 
                             element_pitch, 
                             element_width, 
                             first_element_position, 
                             centre_freq, 
                             ctypes.byref(time), 
                             ctypes.byref(fmc_data),
                             element_positions.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                             (ctypes.c_int * 2)(*element_positions.shape),
                             ctypes.byref(points))
    
    if fmc_data.data:
        fmc_data_arr = np.ctypeslib.as_array(fmc_data.data, shape = (fmc_data.size[2], fmc_data.size[1], fmc_data.size[0]))
    else:
        fmc_data_arr = np.zeros(0)
    if time.data:
        time_arr = np.ctypeslib.as_array(time.data, shape = (time.size[0], ))
    else:
        time_arr = np.zeros(0)
    
    return (fmc_data_arr, time_arr, element_positions)


#------------------------------------------------------------------------------
#Generally useful functions
def fn_hanning_band_pass(no_pts, start_rise_fract, end_rise_fract, start_fall_fract, end_fall_fract):
    """Returns a vector containing a Hanning band-pass function.
    
    The vector will be no_pts long, it will usually start at zero, rise 
    smoothly to one, remain at one, then return smoothly to zero. The points
    where the transitions start and end are determined by the arguments
    start_rise_fract, end_rise_fract, start_fall_fract, and end_fall_fract.
    All of these are expressed as fractions of the total length of the vector,
    where 0 is the first point and 1 is the last point.
    """
    return fn_hanning_hi_pass(no_pts, start_rise_fract, end_rise_fract) * fn_hanning_lo_pass(no_pts, start_fall_fract, end_fall_fract)

def fn_hanning_hi_pass(no_pts, start_rise_fract, end_rise_fract):
    """Returns a vector containing a Hanning high-pass function.
    
    The vector will be no_pts long, it will usually start at zero, rise 
    smoothly to one, and remain at one. The points where the transition starts
    and ends are determined by the arguments start_rise_fract and end_rise_fract.
    Both of these are expressed as fractions of the total length of the vector,
    where 0 is the first point and 1 is the last point.
    """
    x = np.linspace(0, 1, no_pts)
    window = 0.5 * (1 - np.cos(np.pi * (x - start_rise_fract) / (end_rise_fract - start_rise_fract))) * (x > start_rise_fract)
    window[x > end_rise_fract] = 1
    return window

def fn_hanning_lo_pass(no_pts, start_fall_fract, end_fall_fract):
    """Returns a vector containing a Hanning low-pass function.
    
    The vector will be no_pts long, it will usually start at one, fall
    smoothly to zero, and remain at zero. The points where the transition starts
    and ends are determined by the arguments start_fall_fract and end_fall_fract.
    Both of these are expressed as fractions of the total length of the vector,
    where 0 is the first point and 1 is the last point.
    """
    x = np.linspace(0, 1, no_pts)
    window = 0.5 * (1 + np.cos(np.pi * (x - start_fall_fract) / (end_fall_fract - start_fall_fract))) * (x < end_fall_fract);
    window[x < start_fall_fract] = 1
    return window

def fn_hanning(no_pts, peak_pos_fract, half_width_fract):
    """Returns a vector containing a Hanning function.
    
    The vector will be no_pts long, it will usually start at zero, rise smoothly
    to one and then return smoothly to zero. The position and width of the peak
    are determined by the arguments peak_pos_fract and half_width_fract.
    Both of these are expressed as fractions of the total length of the vector,
    where 0 is the first point and 1 is the last point.
    """
    x = np.linspace(0, 1, no_pts)
    window = 0.5 * (1 + np.cos((x - peak_pos_fract) / half_width_fract * np.pi)) * \
        ((x >= (peak_pos_fract - half_width_fract)) & (x <= (peak_pos_fract + half_width_fract)))
    return window

def fn_sinc(x):
    """Evaluates the sinc function, sin(pi * x) / (pi * x)
    
    Input x should be an Numpy array. The function will return correct values
    without warnings even as abs(x) -> 0.
    """
    eps = np.finfo(np.float64).eps
    i = np.abs(x) < eps
    x[i] = 1
    y = np.sin(np.pi * x) / (np.pi * x)
    y[i] = 1
    return y

def fn_load_bearing_data(fname):
    """Loads the ultrasonic data for the bearing casing.
    
    Returns 
        pos - m-element vector of transducer positions
        time - n-element vector of time values
        voltage - m x n 2D array of measured signals where first dimension is 
            transducer positon and second is time
    """
    f = open(fname)
    data = json.load(f)
    pos = np.array(data['pos'])
    time = np.array(data['time'])
    time = time.T
    voltage = np.array(data['voltage'])
    voltage = voltage.T
    return (pos, time, voltage)

def fn_load_joint_data(fname):
    """Loads the ultrasonic data measured off or on an adhesive joint.
    
    Input fname is either joint_off_adhesive.json or joint_on_adhesive.json
    
    Returns
        time - m-element vector of time values
        voltage - m-element vector of voltages
    
    *TODO describe return arguments
    """
    f = open(fname)
    data = json.load(f)
    time = np.array(data['time'])
    time = time.T
    voltage = np.array(data['voltage'])
    voltage = voltage.T
    return (time, voltage)