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

def loadModels(modelsDirectory="model_data\\VPMModel"):
    """Load the models from the given model info directory.
    Returns a tuple - the models, as well as the max mels used to normalize.
    Args:
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
    with open(os.path.join(modelsDirectory, "ModelInfo.txt"), 'r', newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for idx, row in enumerate(reader):
            if idx == 0: continue
            _, shiftAmt, maxMel, path = row
            maxMels.append(maxMel)
            
            # Hardcoded settings here
            model = TimbreFNN(n_input=512, n_hid=512, n_mels=256)
            model.load_state_dict(torch.load(os.path.join(modelsDirectory, path)))
            model.eval()
            
            models.append(model)
    return models, np.array(maxMels, dtype=np.float32);

def pitchShift(models, maxMels, wavform, pitchShiftAmt):
    """Perform the pitch shift, using the models and maxMels array provided.
    Args:
        models (list): A list of pytorch models.
        np.array (len(models)): the normalizing factor used for each model.
        wavform (np.array): An array of floats representing the waveform to shift
        pitchShiftAmt (int): The pitch shift amount. Should be [-5, 5]  
    Returns:
        gl_abs_waveform (np.array): The shifted wav file put through the decoder
        shifted_wav (np.array): The purely shifted wav file
    """
    n_ffts = 2048; overlap = 0.75; n_mels = 256; global_max_mels = 18.21612
    sr = 44100
    
    pitchShiftAmt = int(pitchShiftAmt)
    assert (-5 <= pitchShiftAmt and pitchShiftAmt <= 5)
    modelIdx = pitchShiftAmt + 5
    
    spectrogram = stft(wavform, win_length=n_ffts, overlap=overlap, plot=False) 
    wav_mels_prenorm = ffts_to_mel(spectrogram, n_mels=n_mels, skip_mfcc=True) 
    wav_mels_prenorm = wav_mels_prenorm.T; 
    wav_mels_logged = np.log(wav_mels_prenorm)
    wav_mels = wav_mels_logged / global_max_mels
    
    shifted_wav, shifted_spectrogram = resample_pitch_shift(np.array([wavform]), pitchShiftAmt, overlap, n_ffts)
    shifted_wav = shifted_wav[0]; shifted_spectrogram = shifted_spectrogram[0]
    shifted_wav_mels_prenorm = ffts_to_mel(shifted_spectrogram, n_mels=n_mels, skip_mfcc=True) 
    shifted_wav_mels_prenorm = shifted_wav_mels_prenorm.T
    shifted_wav_mels_logged = np.log(shifted_wav_mels_prenorm)
    shifted_wav_mels = shifted_wav_mels_logged / maxMels[modelIdx]
    
    # Truncate excess windows if off by a few
    if (wav_mels.shape[0] != shifted_wav_mels.shape[0]):
        shifted_wav_mels = shifted_wav_mels[0:wav_mels.shape[0]]

    model = models[modelIdx]
    
    wav_input = torch.tensor(np.concatenate((wav_mels, shifted_wav_mels), axis=1)).float()
    wav_predicted = model(wav_input).detach().numpy();
    wav_denorm_mels = np.e ** (wav_predicted * global_max_mels)
#     wav_denorm_mels = np.e ** (wav_predicted * maxMels[modelIdx]) # This denormalization doesn't work well
    
    gl_abs_waveform = librosa.feature.inverse.mel_to_audio(
        wav_denorm_mels.T, sr=sr, n_fft=n_ffts, 
        hop_length=compute_hop_length(n_ffts, overlap), 
        win_length=n_ffts)
    return gl_abs_waveform, shifted_wav

# The wrapper functions
models, maxMels = loadModels(os.path.join("model_data", "VPMModel"))
PitchShift = lambda wavform, pitchShiftAmt: pitchShift(models, maxMels, wavform, pitchShiftAmt)
