# utils/time.py
import numpy as np

def get_frame_array(array1):
    return np.array([i for i in range(len(array1))])

def get_time_array(array2, fps):
    return np.array([i/fps for i in range(len(array2))])
