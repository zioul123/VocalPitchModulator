# @Louiz
def load_wav_files(file_paths):
    """Takes a list of filepaths, and returns a 2D array with all their data.

    The function also ensures that the resulting array only contains mono data.
    
    Args:
        file_paths (list): A list of filenames, where each filename is a 
            .wav file to be added to the output.

    Returns:
        signal_data (np.ndarray): A 2D matrix that contains the audio data of
            all specified file paths, such that
            signal_data[i] provides the waveform of the ith file in file_paths.
            The dimensions are (len(file_paths), [length of file at filepath])
    """
    assert(file_path[-4:] == '.wav')
    """
    !! Write code here !!
    """

# @Rachel/Shaun For your use
def compute_hop_length(win_length, overlap):
    """Utility function to compute the hop_length.

    This is used by stft, istft, and ffts_to_melspectrogram.

    Args:
        win_length (int): The size of a window
        overlap (float): The amount of overlap between each window.

    Returns:
        hop_length (int): The computed hop_length.
    """
    """
    !! Write code here !!
    """

# @Rachel/Shaun, this is the whole "Basic Preprocessing" part
def stft(waveform, win_length=1024, overlap=.5, window='hann'):
    """Takes a waveform and returns a 2D complex-valued matrix (spectrogram).
    
    The function performs STFT, i.e. windowing and performing FFT on each 
    window. This is a wrapper for the librosa.core.stft function.

    Args:
        waveform (np.array): An array of amplitudes representing a signal.
        win_length (int): The size of each window (and corresponding FFT)
        overlap (float): The amount of overlap between each window. This
            translates to the hop_length.
        window (str): The window to use, specified by scipy.signal.get_window. 

    Returns:
        ffts (np.ndarray): A 2D complex-valued matrix such that
            np.abs(ffts[f, t]) is the magnitude (of freq bin f at frame t)
            np.angle(ffts[f, t]) is the phase
            The dimensions are (win_length, [number of frames for waveform])
    """
    """
    !! Write code here !!
    """

# @Rachel/Shaun, this is the whole "Postprocess" part
def istft(ffts, win_length=1024, overlap=.5, window='hann'):
    """Takes a 2D complex-valued matrix (spectrogram) and returns a waveform.

    This function performs ISTFT, and is a wrapper for librosa.core.istft.

    Args:
        ffts (np.ndarray): A 2D complex-valued matrix such that ffts[f, t]
            is the complex number representing the FFT value for the spectrum
            of freq bin f at frame t.
            The dimensions are (win_length, [number of frames for waveform])
        win_length (int): The size of each window (and corresponding FFT)
        overlap (float): The amount of overlap between each window. This
            translates to the hop_length.
        window (str): The window to use, specified by scipy.signal.get_windo

    Returns:
        waveform (np.array): An array of amplitudes representing a signal.
    """
    """
    !! Write code here !!
    """

# @Rachel/Shaun This is the "Mel Filter 1" and "Mel Filter 2"
def ffts_to_mel(ffts, win_length=1024, overlap=.5, n_mels=256, 
    n_mfcc=20, skip_mfcc=False):
    """Converts a spectrogram to a mel-spectrogram and MFCC.
    
    This function is a wrapper for librosa.feature.melspectrogram and
    librosa.feature.mfcc.

    Args:
        ffts (np.ndarray): A 2D complex-valued matrix such that ffts[f, t]
            is the complex number representing the FFT value for the spectrum
            of freq bin f at frame t.
            The dimensions are (win_length, [number of frames for waveform])
        win_length (int): The size of each window (and corresponding FFT)
        overlap (float): The amount of overlap between each window. This
            translates to the hop_length.
        n_mels (int): The number of Mel bands to generate.
        n_mfcc (int): The number of MFCC features to compute.
        skip_mfcc(boolean): If True, do compute the MFCC, as we only want the
            spectrogram information.

    Returns:
        mel_spectrogram (np.ndarray): A 2D matrix such that
            mel_spectrogram[m, t] is the magnitude of mel bin m at frame t
            The dimensions are (n_mels, [ffts.shape[1]])
        mfcc (np.ndarray): A 2D matrix such that
            mfcc[m, t] is the magnitude of the mth feature at frame t 
    """
    """
    !! Write code here !!
    Louiz's note: Please handle sampling rate properly, we assume always 44100.
    Check out librosa.filters.mel if unsure how to write the arguments to call 
    librosa.feature.melspectrogram.
    """
    if (!skip_mfcc):
        """
        !! Write code to compute MFCC here !!
        """

# @Zach This is the "Pitch Shift" 
def simple_fft_pitch_shift(fft, shift_amt):
    """Takes a single fft vector and shifts all values in the frequency domain.

    This is a "naive" pitch shift that simply up-shifts the values in the 
    given fft, and is not expected to sound natural. Note that this is done on 
    a SINGLE time slice. For shifting of an entire spectrogram, use 
    simple_ffts_pitch_shift instead.

    Explanation:
    Assume that each value in fft (e.g. fft[f]), is given by a (value, freq) 
    pair. We shift the pitch by multiplying the frequency values by 
    (2**(shift_amt/12)), and interpolating the values back into the original 
    frequency bin values (since the fft bins must keep their original frequency 
    resolution). 

    Example:
    Assume a frequency resolution of 20Hz, where bin 0: 0Hz, bin 1: 20 Hz etc.
    If z = fft[3], where bin 3 is the frequency 60Hz, we denote this
    as (z, 60). We also have z' = fft[2], denoted as (z', 40).

    So shifting the pitch by 1, we end up with (z, 63.57) and (z', 42.4).
    So to get the value at 60Hz, we will need to interpolate between z' to z.

    Args:
        fft (np.array): A complex-valued array such that ffts[f] is the 
            complex number representing the FFT value for the spectrum of freq 
            bin f. Dimensions are (win_length,), where win_length is the 
            number of windows used to generate this fft.
        shift_amt (int): The number of semitones to shift the pitch by. The
            expected range is [-15, 15].

    Returns:
        shifted_fft (np.array): A (win_length,) array with the shifted fft.
    """
    assert(-15 <= shift_amt and shift_amt <= 15)
    """
    !! Write code here !!
    """

def simple_ffts_pitch_shift(ffts, shift_amt):
    """Takes a 2D spectrogram, and pitch_shifts each time slice.

    Args:
        ffts (np.ndarray): A 2D complex-valued matrix such that ffts[f, t]
            is the complex number representing the FFT value for the spectrum
            of freq bin f at frame t.
            The dimensions are (win_length, [number of frames for waveform])
        shift_amt (int): The number of semitones to shift the pitch by. The
            expected range is [-15, 15].
    Returns:
        shifted_ffts (np.ndarray): A spectrogram of equal dimensions to ffts,
            with shifted frequency space.
    """ 
    assert(-15 <= shift_amt and shift_amt <= 15)
    return np.array([ simple_fft_pitch_shift(fft, shift_amt) for fft in ffts.T ]).T
