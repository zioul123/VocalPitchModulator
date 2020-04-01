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


    all_spectrograms = np.array([stft(waveform, overlap=.75, plot=False) for waveform in all_wav_data])

    print("Pre processing completed")
    return all_spectrograms, flat_data_ref_list


def apply_pitch_shift(all_fft, shift_amt):

    all_pitched_spectra = np.array([simple_ffts_pitch_shift(track_fft, shift_amt) for track_fft in all_fft])
    print(all_pitched_spectra[0])

    return all_pitched_spectra


def postprocess_data(all_fft, flat_data_ref_list, numSemitonesToShift):

    index = 0
    for fft in all_fft:
        # istft(fft, overlap=.75, save_file=True, file_name=flat_data_ref_list[index] + " pitched_up_by " + numSemitonesToShift)
        wavform_griffinlim = core.griffinlim(fft, hop_length=compute_hop_length(1024, 0.75), win_length=1024)
        librosa.output.write_wav('output_wav/' + flat_data_ref_list[index] + " pitched_up_by " + numSemitonesToShift + '.wav', wavform_griffinlim, 44100)
        index += 1

    print("Post processing completed")


if __name__ == '__main__':
    all_ffts, flat_data_ref_list = preprocess_data()

    numSemitonesToShift = input("Enter the number of semitones to up shift from range [-15, 15]: ")

    all_shifted_ffts = apply_pitch_shift(all_ffts, int(numSemitonesToShift))

    postprocess_data(all_shifted_ffts, flat_data_ref_list, numSemitonesToShift)
