import numpy as np

def flatten_3d_array(arr, i_lim, j_lim, k_lim):
    """Takes a 3d array, and returns a 1d array.

    Args: 
        arr (list): A 3d array (i_lim, j_lim, k_lim)
        i_lim (int): The length of the first dimension of arr
        j_lim (int): The length of the second dimension of arr
        k_lim (int): The length of the third dimension of arr
    
    Returns:
        A 1d list of length i_lim * j_lim * k_lim
    """
    return  [ arr[i][j][k] 
              for i in range(i_lim) 
              for j in range(j_lim) 
              for k in range(k_lim) ]

def flat_array_idx(i, j, k, i_lim, j_lim, k_lim):
    """Used to get the index of flattened arrays as a 3d arrays.
    
    This is used to access arrays that have been flattened
    by flatten_3d_array.

    Args:
        i/j/k (int): The indices to access the array as arr[i][j][k]
        i_lim (int): The length of the first dimension of arr
        j_lim (int): The length of the second dimension of arr
        k_lim (int): The length of the third dimension of arr

    Returns:
        An int representing the array index of the flattened array
            that functions as the accessor to the index [i][j][k]
            in the original 3d array.
    """
    return i * j_lim * k_lim + j * k_lim + k

def nd_array_idx(idx, i_lim, j_lim, k_lim):
    """Used to get the 3d index from a flat array index.

    This is the inverse of flat_array_idx.

    Args:
        idx (int): The index to access the flat array.
        i_lim (int): The length of the first dimension of arr
        j_lim (int): The length of the second dimension of arr
        k_lim (int): The length of the third dimension of arr

    Returns:
        Three ints representing the i, j, k index to access the
            original 3d array.
    """
    return np.floor(idx / (j_lim * k_lim)), \
           np.floor((idx % (j_lim * k_lim)) / k_lim), \
           np.floor(idx % (k_lim))
