import os
import csv

import scipy.io as sio
from scipy.io import wavfile
from scipy.io.wavfile import write
import scipy.signal as sis
import scipy.fftpack as fftpack

import librosa
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Sample rate and bit depth
sample_rate = 44100
short_max = 32768

def create_data_ref_list(file_path, n_pitches, n_vowels, n_people):
    """Create a list of all the filenames in the dataset.

    To access the list of files, use data_ref_list[vowel_idx][pitch_idx]
    A specific filename is accessed with data_ref_list[vowel_idx][pitch_idx][person_idx]

    Args:
        filepath (str): The file path to the csv file containing the
            filenames of all .wav files
        n_pitches (int): The number of pitches in our dataset
        n_vowels (int): The number of vowels in our dataset
        n_people (int): The number of people in our dataset

    Returns:
        data_ref_list (list): A list of filenames, organized by word, then
            pitch, then people (n_vowels, n_pitches, n_people)
    """
    data_ref_list = [ [ [] for pIdx in range(0, n_pitches) ]
                      for wIdx in range(0, n_vowels) ]

    with open(file_path) as dataset_csv:
        reader = csv.reader(dataset_csv, delimiter=',')
        for idx, row in enumerate(reader):
            if idx == 0: continue
            filename, vowel_idx, pitch_idx, personNum = row
            data_ref_list[int(vowel_idx)][int(pitch_idx)].append(filename)
    return data_ref_list

def create_data_label_pairs(n_pitches):
    """Create a dictionary/array of data-label pairs.

    This provides an array of 3-tuples, as well as a dictionary of
    arrays, where each subarray contains 3-tuples, with elements:
    [shift_amt, input_pitch_idx, label_pitch_Idx],
    where input_pitch_idx is the input, label_pitch_Idx is the desired output.

    This is used so that we can feed data-label pairs to our neural net.
    We should use each 3-tuple n_people * n_vowels times, with
    the files referenced via data_list[vowel_idx][pitch_idx][person_idx],
    and the pitch shift amount provided to the NN.

    Args:
        n_pitches (int): The number of pitches in our dataset

    Returns:
        data_label_pairs: An array where each element is a 3-tuple of
            [shift_amt, input_pitch_idx, label_pitch_idx].
        data_label_pairs_dict: A dictionary of dimension n_pitchShifts, where each
            element is an array (n_words * n_startingPitches,), where:
            n_pitchShifts: the number of possible pitch shifts,
            n_startingPitches: the number of starting pitches for that
            shift_amt value
    """
    def append_tuple(shift_amt, pitch_idx):
        # [shift_amt, input_pitch_idx, label_pitch_Idx]
        data_label_pairs_dict[shift_amt].append(
            [shift_amt, pitch_idx, pitch_idx + shift_amt])
        data_label_pairs.append(
            [shift_amt, pitch_idx, pitch_idx + shift_amt])

    data_label_pairs = []
    data_label_pairs_dict = {}
    for pIdx in range(-n_pitches + 1, n_pitches):
        data_label_pairs_dict[pIdx] = []

    # Pitch indices range from 0-15, so we can shift from -15 to 15 pitches up.
    # First loop: shift_amt 0 to 15
    for shift_amt in range(0, n_pitches):
        # Iterate through available pitch shift starting points
        for pitch_idx in range(0, n_pitches - shift_amt):
            append_tuple(shift_amt, pitch_idx)
    # Second loop: shift_amt -15 to -1
    for shift_amt in range(-n_pitches + 1, 0):
        for pitch_idx in range(n_pitches - 1, -1 - shift_amt, -1):
            append_tuple(shift_amt, pitch_idx)

    return data_label_pairs, data_label_pairs_dict

def load_wav_files(rel_path, data_list):
    """Takes a list of filepaths, and returns a 2D array with all their data.

    The function also ensures that the resulting array only contains mono data.

    Args:
        rel_path (str): The relative path of the filenames in file_paths.
            i.e. the filepath would be rel_path/file_paths[i]
        file_paths (list): A list of filenames, where each filename is a
            .wav file to be added to the output.

    Returns:
        signal_data (np.ndarray): A 2D matrix that contains the audio data of
            all specified file paths, such that
            signal_data[i] provides the waveform of the ith file in file_paths.
            The dimensions are (len(file_paths), [length of file at filepath])
    """
    result = []
    for idx, file_path in enumerate(data_list):
        assert(file_path[-4:] == '.wav')
        s_r, short_data = sio.wavfile.read(os.path.join(rel_path,file_path))
        assert(s_r == sample_rate)
        result.append(short_data / short_max)
    return np.array(result)

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
    return int(win_length * (1 - overlap))

# @Rachel/Shaun, this is the whole "Basic Preprocessing" part
def stft(waveform, win_length=1024, overlap=.5, window='hann', plot=True):
    """Takes a waveform and returns a 2D complex-valued matrix (spectrogram).

    The function performs STFT, i.e. windowing and performing FFT on each
    window. This is a wrapper for the librosa.core.stft function.

    Args:
        waveform (np.array): An array of amplitudes representing a signal.
        win_length (int): The size of each window (and corresponding FFT)
        overlap (float): The amount of overlap between each window. This
            translates to the hop_length.
        window (str): The window to use, specified by scipy.signal.get_window.
        plot (bool): If true, plot the spectrogram.

    Returns:
        ffts (np.ndarray): A 2D complex-valued matrix such that
            np.abs(ffts[f, t]) is the magnitude (of freq bin f at frame t)
            np.angle(ffts[f, t]) is the phase
            The dimensions are (win_length, [number of frames for waveform])
    """
    waveform_norm = librosa.util.normalize(waveform)
    hop_length = compute_hop_length(win_length, overlap)
    waveform_stft = librosa.core.stft(waveform_norm, n_fft=win_length, hop_length=hop_length, win_length=win_length, window=window)

    if plot:
        librosa.display.specshow(librosa.amplitude_to_db(waveform_stft, ref=np.max), y_axis='log', x_axis='time')
        plt.title('Power spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.show()

    return waveform_stft

# @Rachel/Shaun, this is the whole "Postprocess" part
def istft(ffts, win_length=1024, overlap=.5, window='hann', save_file=False, file_name=''):
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
        window (str): The window to use, specified by scipy.signal.get_window
        save_file (bool): If true, save of waveform to audio wav file.
        file_name (str): The respective file name for the audio wav file.

    Returns:
        waveform (np.array): An array of amplitudes representing a signal.
    """
    hop_length = compute_hop_length(win_length, overlap)
    waveform_istft = librosa.core.istft(ffts, hop_length=hop_length, win_length=win_length, window=window)

    if save_file:
        # change the path and file name accordingly
        librosa.output.write_wav('output_wav/' + file_name + '.wav', waveform_istft, sample_rate)

    return waveform_istft

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
    if not skip_mfcc:
        print("Computing MFCC")
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

    freqs = librosa.core.fft_frequencies(sample_rate,1024)
    shifted_freqs = freqs * np.power(2, shift_amt/12)
    shifted_fft = np.interp(freqs, shifted_freqs, fft)

    return shifted_fft

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
