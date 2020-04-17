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

from Utils import *
from VPM import *
from ANN import *

# Hardcoded data - information used based on architecture.
paths = [ "Archi-0_ModelInfo.txt", "Archi-1_ModelInfo.txt", "Archi-2_ModelInfo.txt", "Archi-3_ModelInfo.txt" ]
n_inputs = [ 256, 296, 266, 512 ]
nhids =  [ 256, 296, 266, 512 ]

# The Timbre Encoder model in case it's being used
TE_path = os.path.join('model_data', 'TimbreEncoder', 'TimbreVAE-IdealFixedNorm-40-36-10.pt')
TE = TimbreVAE(n_mfcc=40, n_hid=36, n_timb=10)
TE.load_state_dict(torch.load(TE_path, map_location=torch.device('cpu')))
TE.eval()

def loadModels(archi = 3, modelsDirectory="model_data\\VPMModel"):
    """Load the models from the given model info directory.
    Returns a tuple - the models, as well as the max mels used to normalize.
    Args:
        archi (int): The mmodel architecture to make use of.
        modelsDirectory (str): The directory the models and model information are stored.
            This should follow the structure:
                - model_data
                  - VPMModel
                    - ModelInfo.txt
                    - <.pt models>
    Returns:
        models (list): A list of pytorch models.
        np.array (len(models)): the normalizing factor used for each model.
    """
    models = []; maxMels = []
    with open(os.path.join(modelsDirectory, paths[archi]), 'r', newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for idx, row in enumerate(reader):
            if idx == 0: continue
            _, shiftAmt, maxMel, path = row
            maxMels.append(maxMel)

            # Hardcoded settings here
            model = BaseFFN(n_input=n_inputs[archi], n_hid=nhids[archi], n_output=256)
            model.load_state_dict(torch.load(os.path.join(modelsDirectory, path), map_location=torch.device('cpu')))
            model.eval()

            models.append(model)
    return models, np.array(maxMels, dtype=np.float32);

def pitchShift(mode, models, maxMels, wavform, pitchShiftAmt, getPureShift=False):
    """Perform the pitch shift, using the models and maxMels array provided.
    Args:
        mode (int): Which architecture to use
        models (list): A list of pytorch models.
        np.array (len(models)): the normalizing factor used for each model.
        wavform (np.array): An array of floats representing the waveform to shift
        pitchShiftAmt (int): The pitch shift amount. Should be [-5, 5]
    Returns:
        gl_abs_waveform (np.array): The shifted wav file put through the decoder
        shifted_wav (np.array): The purely shifted wav file
    """
    n_ffts = 2048; overlap = 0.75; n_mels = 256; global_max_mels = 18.21612; global_max_mfcc = 9.774869; global_max_timbre = 3.7099004
    sr = 44100

    pitchShiftAmt = int(pitchShiftAmt)
    assert (-5 <= pitchShiftAmt and pitchShiftAmt <= 5)
    modelIdx = pitchShiftAmt + 5

    spectrogram = stft(wavform, win_length=n_ffts, overlap=overlap, plot=False)

    # Only mode 1 and 2 use MFCC
    if mode == 1 or mode == 2:
        wav_mels_prenorm, wav_mfcc_prenorm = ffts_to_mel(spectrogram, n_mels=n_mels, n_mfcc=40, skip_mfcc=False)
    else:
        wav_mels_prenorm = ffts_to_mel(spectrogram, n_mels=n_mels, skip_mfcc=True)

    wav_mels_prenorm = wav_mels_prenorm.T;
    wav_mels_logged = np.log(wav_mels_prenorm)
    wav_mels = wav_mels_logged / global_max_mels

    # Mode 1 and 2 use MFCC
    if mode == 1 or mode == 2:
        wav_mfcc_prenorm = wav_mfcc_prenorm.T
        wav_mfcc_logged = np.log(np.abs(wav_mfcc_prenorm))
        wav_mfcc = wav_mfcc_logged / global_max_mfcc
    # Mode 2 uses timbre encoder
    if mode == 2:
        wav_mfcc_tensor = torch.Tensor(wav_mfcc)
        wav_timbre_prenorm = TE.get_z(wav_mfcc_tensor).detach().numpy()
        wav_timbre = wav_timbre_prenorm / global_max_timbre

    shifted_wav, shifted_spectrogram = resample_pitch_shift(np.array([wavform]), pitchShiftAmt, overlap, n_ffts)
    shifted_wav = shifted_wav[0]; shifted_spectrogram = shifted_spectrogram[0]
    shifted_wav_mels_prenorm = ffts_to_mel(shifted_spectrogram, n_mels=n_mels, skip_mfcc=True)
    shifted_wav_mels_prenorm = shifted_wav_mels_prenorm.T
    shifted_wav_mels_logged = np.log(shifted_wav_mels_prenorm)
    shifted_wav_mels = shifted_wav_mels_logged / maxMels[modelIdx]

    # Truncate excess windows if off by a few
    # print("Shapes: Orig: {}, Shifted: {}".format(wav_mels.shape, shifted_wav_mels.shape))
    if (wav_mels.shape[0] < shifted_wav_mels.shape[0]):
        shifted_wav_mels = shifted_wav_mels[0:wav_mels.shape[0]]
    if (wav_mels.shape[0] > shifted_wav_mels.shape[0]):
        wav_mels = wav_mels[0:shifted_wav_mels.shape[0]]
        if mode == 1 or mode == 2:
            wav_mfcc = wav_mfcc[0:shifted_wav_mels.shape[0]]
        if mode == 2:
            wav_timbre = wav_timbre[0:shifted_wav_mels.shape[0]]

    model = models[modelIdx]

    if (mode == 0):
        wav_input = torch.tensor(wav_mels).float()
    if (mode == 1):
        wav_input = torch.tensor(np.concatenate((wav_mfcc, shifted_wav_mels), axis=1)).float()
    if (mode == 2):
        wav_input = torch.tensor(np.concatenate((wav_timbre, shifted_wav_mels), axis=1)).float()
    if (mode == 3):
        wav_input = torch.tensor(np.concatenate((wav_mels, shifted_wav_mels), axis=1)).float()
    # print("Input to model shape: {}".format(wav_input.shape))
    wav_predicted = model(wav_input).detach().numpy();
    wav_denorm_mels = np.e ** (wav_predicted * global_max_mels)
#     wav_denorm_mels = np.e ** (wav_predicted * maxMels[modelIdx]) # This denormalization doesn't work well

    gl_abs_waveform = librosa.feature.inverse.mel_to_audio(
        wav_denorm_mels.T, sr=sr, n_fft=n_ffts,
        hop_length=compute_hop_length(n_ffts, overlap),
        win_length=n_ffts)
    if not getPureShift:
        return np.array(gl_abs_waveform, dtype=np.float32), np.array(shifted_wav, dtype=np.float32)
    else:
        return np.array(gl_abs_waveform, dtype=np.float32)

# The wrapper functions
modelss, maxMelss = zip(*[ loadModels(archi, os.path.join("model_data", "VPMModel")) for archi in range(4) ])

PitchShift0 = lambda wavform, pitchShiftAmt: pitchShift(0, modelss[0], maxMelss[0], wavform, pitchShiftAmt)
PitchShift1 = lambda wavform, pitchShiftAmt: pitchShift(1, modelss[1], maxMelss[1], wavform, pitchShiftAmt)
PitchShift2 = lambda wavform, pitchShiftAmt: pitchShift(2, modelss[2], maxMelss[2], wavform, pitchShiftAmt)
PitchShift3 = lambda wavform, pitchShiftAmt: pitchShift(3, modelss[3], maxMelss[3], wavform, pitchShiftAmt)

# Recommended is identical to PitchShift0
PitchShift = lambda wavform, pitchShiftAmt: pitchShift(0, modelss[0], maxMelss[0], wavform, pitchShiftAmt)
