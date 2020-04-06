import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from enum import IntEnum

#################################################################
# 3D to flattened array utilities
#################################################################

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

def flat_3d_array_idx(i, j, k, i_lim, j_lim, k_lim):
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

def flat_2d_array_idx(i, j, i_lim, j_lim):
    """Analogous to flat_array_idx, except for 2d arrays.
    
    Args and Return: 
        See above.
    """
    return i * j_lim + j

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
    return int(idx / (j_lim * k_lim)), \
           int((idx % (j_lim * k_lim)) / k_lim), \
           int(idx % (k_lim))

#################################################################
# Graphing utilities
#################################################################

def plot_ffts_spectrogram(ffts, sample_rate, file_name=None):
    """This function plots a spectrogram generated by stft

    Args:
        ffts (np.ndarray): An matrix of ffts, where the ffts[f, t]
            provides the complex value of the fft at frequency bin f
            at frame t.
        sample_rate (int): The sample rate at which the stft was taken.
        file_name (str): To be put into the title of the plot.
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(ffts), ref=np.max), 
        y_axis='linear', x_axis='time', sr=sample_rate)
    plt.title('Power spectrogram{}'.format(
        "" if file_name is None else " of {}".format(file_name)))
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

def plot_mel_spectrogram(mel_freq_spec, sample_rate, file_name=None):
    """This function plots a mel spectrogram generated by ffts_to_mel

    Args:
        mel_freq_spec (np.ndarray): An matrix of mel spectra, where the 
            mel_freq_spec[m, t] provides the value of the mel bin m at frame t.
        sample_rate (int): The sample rate at which the stft was taken.
        file_name (str): To be put into the title of the plot.
    """
    plt.figure(figsize=(10, 4))
    S_dB = librosa.power_to_db(mel_freq_spec, ref=np.max)
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sample_rate, fmax=sample_rate/2.0)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.show()

def plot_mfcc(mfccs, sample_rate, file_name=None):
    """This function plots a mfcc generated by ffts_to_mel

    Args:
        mfccs (np.ndarray): An matrix of MFCC features, where
            mfccs[m, t] provides the values of the MFCC feature m at frame t.
        sample_rate (int): The sample rate at which the stft was taken.
        file_name (str): To be put into the title of the plot.
    """

    plt.figure(figsize=(10,4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()

def plot_loss_graph(loss_arr, val_loss_arr=None, acc_arr=None, val_acc_arr=None):
    """This function is used to plot the loss graph of the fit function.
    
    Args:
        loss_arr (list): An array of floats for training loss at each epoch.
        val_loss_arr (list): An array of floats for validation loss at each epoch.
        acc_arr (list): An array of floats for training accuracy at each epoch.
        val_acc_arr (list): An array of floats for validation acc at each epoch.
    """

    plt.figure(figsize=(15, 10))
    plt.plot(loss_arr, 'r-', label='loss')
    if val_loss_arr != None:
        plt.plot(val_loss_arr, 'm-', label='val loss')
    if acc_arr != None:
        plt.plot(acc_arr, 'b-', label='train accuracy')
    if val_acc_arr != None:
        plt.plot(val_acc_arr, 'g-', label='val accuracy')
    plt.title("Loss plot")
    plt.xlabel("Epoch")
    plt.legend(loc='best')
    plt.show()
    print('Training Loss before/after: {}, {}'
          .format(loss_arr[0], loss_arr[-1]))
    if val_loss_arr != None:
        print('Validation Loss before/after: {}, {}'
              .format(val_loss_arr[0], val_loss_arr[-1]))
    if acc_arr != None:
        print('Training accuracy before/after: {}, {}'
              .format(acc_arr[0], acc_arr[-1]))
    if val_acc_arr != None:
        print('Validation accuracy before/after: {}, {}'
              .format(val_acc_arr[0], val_acc_arr[-1]))

#################################################################
# Normalization utilities
#################################################################

class NormMode(IntEnum):
    REAL_TO_ZERO_ONE = 0
    NONNEG_TO_ZERO_ONE = 1
    REAL_TO_NEG_ONE_ONE = 2
    NEG_ONE_ONE_TO_ZERO_ONE = 3

class DenormMode(IntEnum):
    ZERO_ONE_TO_REAL = 0
    ZERO_ONE_TO_NONNEG = 1
    NEG_ONE_ONE_TO_REAL = 2
    ZERO_ONE_TO_NEG_ONE_ONE = 3

def normalize_rows(mat, norm_mode):
    """This function normalizes each row of mat, and returns the normalizing factors.
    
    We normalize along the rows, so e.g.
    [ [1, -5, 3 ],      [ [0.2, -1, 0.6 ],       [ 5, 
      [3, -3, 1 ],  -->   [1, -1, 0.333 ],  and    3,
      [-2, 4, 3 ] ]       [-0.5, 1, .75 ] ]        4 ]

    Example:
        To retrieve the original rows, we can use:
        normed_mat, scales = normalize_rows(mat, NormMode.REAL_TO_ZERO_ONE);
        original_mat = denormalize_rows(normed_mat, DenormMode.ZERO_ONE_TO_REAL, scales)
        And original_mat will be identical to mat.

    Args:
        mat (np.ndarray): An array of arrays where mat[r] is the rth row.
        norm_mode(NormMode): Which normalization mode to use.
            REAL_TO_ZERO_ONE: Normalize real values to [0, 1]
            NONNEG_TO_ZERO_ONE: Normalize non-negative values to [0, 1]
            REAL_TO_NEG_ONE_ONE: Normalize real values to [-1, 1]
            NEG_ONE_ONE_TO_ZERO_ONE: Normalize [-1, 1] to [0, 1]
    Returns:
        normed_mat (np.ndarray): The normalized matrix.
        norm_vec (np.array): A vector of factors, which can be used as a divisor to
            normed_mat to retrieve the original matrix scale. Note that this 
            is only returned for modes other than NEG_ONE_ONE_TO_ZERO_ONE, where
            there's no real scale involved.
    """
    if norm_mode == NormMode.REAL_TO_ZERO_ONE:
        normed_mat = librosa.util.normalize(mat, axis=1)
        scale_factors = normed_mat[:, 0] / mat[:, 0]
        return normed_mat / 2 + 0.5, scale_factors
    if norm_mode == NormMode.REAL_TO_NEG_ONE_ONE or norm_mode == NormMode.NONNEG_TO_ZERO_ONE:
        normed_mat = librosa.util.normalize(mat, axis=1)
        scale_factors = normed_mat[:, 0] / mat[:, 0]
        return normed_mat, scale_factors
    if norm_mode == NormMode.NEG_ONE_ONE_TO_ZERO_ONE:
        return mat / 2 + 0.5

def normalize(mat, scale_factors=None):
    normed_mat = mat / np.max(mat)
    scale_factors = normed_mat[:, 0] / mat[:, 0]
    return normed_mat, scale_factors

def denormalize_rows(mat, denorm_mode, scale_factors=None):
    """This function denormalizes each row of mat, given an array of scale_factors.
    
    We denormalize along the rows, so e.g.
    [ [0.2, -1, 0.6 ],       [ 5,      [ [1, -5, 3 ],
      [1, -1, 0.333 ],  and    3,  -->   [3, -3, 1 ],
      [-0.5, 1, .75 ] ]        4 ]       [-2, 4, 3 ] ]

    Example:
        To retrieve the original rows, we can use:
        normed_mat, scales = normalize_rows(mat, NormMode.REAL_TO_ZERO_ONE);
        original_mat = denormalize_rows(normed_mat, scales, DenormMode.ZERO_ONE_TO_REAL)
        And original_mat will be identical to mat.

    Args:
        mat (np.ndarray): An array of arrays where mat[r] is the rth row.
        norm_mode(NormMode): Which normalization mode to use.
            ZERO_ONE_TO_REAL: Denormalize values from [0, 1] to real
            ZERO_ONE_TO_NONNEG: Denormalize values from [0, 1] to non-negative
            NEG_ONE_ONE_TO_REAL: Denormalize values from [-1, 1] to real
            ZERO_ONE_TO_NEG_ONE_ONE: Denormalize values from [0, 1] to [-1, 1]
        scale_factors (np.array): The scale factors to denormalize each row by.
    Returns:
        normed_mat (np.ndarray): The normalized matrix
    """
    if denorm_mode == DenormMode.ZERO_ONE_TO_REAL:
        denormed_mat = np.array([ (mat[idx] * 2 - 1) / scale_factors[idx] 
                                  for idx in range(mat.shape[0]) ])
        return denormed_mat
    if denorm_mode == DenormMode.NEG_ONE_ONE_TO_REAL or denorm_mode == DenormMode.ZERO_ONE_TO_NONNEG:
        denormed_mat = np.array([ mat[idx] / scale_factors[idx] 
                                  for idx in range(mat.shape[0]) ])
        return denormed_mat
    if denorm_mode == DenormMode.ZERO_ONE_TO_NEG_ONE_ONE:
        return mat * 2 - 1
