import os
import csv

import scipy.io as sio
from scipy.io import wavfile
from scipy.io.wavfile import write
import scipy.signal as sis
import scipy.fftpack as fftpack

import numpy as np
from librosa import core

from Utils import *
from VPM import stft, simple_ffts_pitch_shift, create_data_ref_list, load_wav_files, istft, compute_hop_length

# Constants that should not change without the dataset being changed
n_pitches = 16
n_vowels = 12
n_people = 3

# These dictionaries are more for reference than anything
label_to_vowel = { 0: "bed",  1: "bird",   2: "boat",  3: "book",
                   4: "cat",  5: "dog",    6: "feet",  7: "law",
                   8: "moo",  9: "nut",   10: "pig",  11: "say" }

vowel_to_label = { "bed": 0,  "bird": 1,  "boat":  2, "book":  3,
                   "cat": 4,  "dog":  5,  "feet":  6    , "law":   7,
                   "moo": 8,  "nut":  9,  "pig":  10, "say":  11}

noteidx_to_pitch = {  0: "A2",   1: "Bb2",  2: "B2",   3: "C3",
                      4: "Db3",  5: "D3",   6: "Eb3",  7: "E3",
                      8: "F3",   9: "Gb3", 10: "G3",  11: "Ab3",
                     12: "A3",  13: "Bb3", 14: "B3",  15: "C4" }

def preprocess_data():
    data_ref_list = create_data_ref_list(os.path.join("Data", 'dataset_files.csv'), n_pitches, n_vowels, n_people)

    flat_data_ref_list = flatten_3d_array(data_ref_list, n_vowels, n_pitches, n_people)

    all_wav_data = load_wav_files(os.path.join("Data", "dataset"), flat_data_ref_list)

    return all_wav_data, flat_data_ref_list

    # all_spectrograms = np.array([stft(waveform, overlap=.75, plot=False) for waveform in all_wav_data])
    #
    # print("Pre processing completed")
    # return all_spectrograms, flat_data_ref_list


def resample_pitch_shift(all_wav, shift_amt, flat_data_ref_list):

    new_sample_rate, factor = compute_new_sample_rate(44100, shift_amt)
    shifted_wav = np.array([core.resample(track_wav, 44100, new_sample_rate) for track_wav in all_wav])

    print("done resampling")
    index = 0

    return shifted_wav


def compute_new_sample_rate(base_sample_rate, shift_amt):
    factor = (2** ( - shift_amt/12))
    new_sample_rate = base_sample_rate * factor

    return new_sample_rate, factor

# def apply_pitch_shift(all_fft, shift_amt):
#
#     all_pitched_spectra = np.array([simple_ffts_pitch_shift(track_fft, shift_amt) for track_fft in all_fft])
#     print(all_pitched_spectra[0])
#
#     return all_pitched_spectra


def postprocess_data(shifted_wav, flat_data_ref_list, shift_amt, n_ffts, overlap):

    all_ffts = np.array([stft(waveform, overlap=overlap, plot=False) for waveform in shifted_wav])

    new_sample_rate, factor = compute_new_sample_rate(44100, int(shift_amt))
    d_fast_arr = np.array([librosa.phase_vocoder(ffts, factor, hop_length=compute_hop_length(n_ffts, overlap)) for ffts in all_ffts])

    # write to file
    index = 0
    for fft in d_fast_arr:
        wav_istft = istft(fft, overlap=overlap, save_file=True, file_name=flat_data_ref_list[index] + " pitched_up_by " + numSemitonesToShift)
        # wavform_griffinlim = core.griffinlim(fft, hop_length=compute_hop_length(n_ffts, overlap), win_length=n_ffts)

        librosa.output.write_wav('output_wav/' + flat_data_ref_list[index] + " pitched_up_by " + numSemitonesToShift + '.wav', wav_istft, 44100)
        print(index)
        index += 1

    index = 0

    print("Post processing completed")


if __name__ == '__main__':
    all_wav, flat_data_ref_list = preprocess_data()

    numSemitonesToShift = input("Enter the number of semitones to up shift from range [-15, 15]: ")

    shifted_wav = resample_pitch_shift(all_wav, int(numSemitonesToShift), flat_data_ref_list)

    # all_shifted_ffts = apply_pitch_shift(all_ffts, int(numSemitonesToShift))
    postprocess_data(shifted_wav, flat_data_ref_list, numSemitonesToShift, 1024, 0.5)
