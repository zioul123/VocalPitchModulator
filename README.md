# Vocal Pitch Modulator
A vocal pitch modulator that uses Machine Learning for realistic voice change. This is a project for NUS's CS4347 (Sound and Music Computing).

## Motivation
The goal of vocal pitch modulation in this project is to maintain a “realistic” sound when the pitch is changed. When we use conventional modulation techniques, increasing the pitch of an audio file by more than around 3 semitones tends to make people sound like chipmunks, while lowering the pitch by more than around 3 semitones makes them sound demonic/dopey. However, there are people who speak at lower and higher pitches without sounding this way, so it is not simply the case that lower/higher pitched voices sound this way, but the spectral characteristics must be adjusted in a suitable manner to keep realism. As such, in this project, we wish to employ machine learning to adjust pitch without losing realism.

## Repository Structure
Figures detailing the implementation of the Vocal Pitch Modulator can be found in the [Documentation](https://github.com/zioul123/VocalPitchModulator/Documentation)  folder.

Tentatively, the API that is to be programmed can be found in [VPM.py](https://github.com/zioul123/VocalPitchModulator/VPM.py). 

The dataset we are training our Artificial Neural Networks with can be found in the [Data/dataset](https://github.com/zioul123/VocalPitchModulator/Data/dataset) folder. Inside the [Data](https://github.com/zioul123/VocalPitchModulator/Data) folder, you will also find the list of files along with the relevant labels in [dataset_files.csv](https://github.com/zioul123/VocalPitchModulator/Data/dataset_files.csv). You will also be able to find the Jupyter Notebooks that were used to generate the dataset, and the raw file list, but we are not including the raw files in this repository, so these will not be for use, but for reference.

## System Pipeline
The following is the proposed modulation pipeline:  
![Modulation Pipeline](/Documentation/Figures/Modulation_Pipeline.png)  

## Data Organization
For a walkthrough of the typical data processing that we conducted, refer to the [Data Processing for Training Walkthrough](https://github.com/zioul123/VocalPitchModulator/blob/master/Data%20Processing%20for%20Training%20Walkthrough.ipynb).

The following are additional aids which illustrates the organization of our data.
![Data List](/Documentation/Figures/Data_List.png)  
![Data-Label Pairs](/Documentation/Figures/Data-Label_Pairs.png)

## Training Pipeline
The following illustrates the training pipelines for the encoder and decoders. There are two possible timbre-extrating neural nets we attempted. The first is a classifier which takes MFCC and tries to output the vowel label, while the other is an autoencoder which takes an MFCC, reduces it from 20 to 4 dimensions, and attempts to reconstruct the original MFCC.
![Encoder Training Pipeline](/Documentation/Figures/Timbre-Enc_Training.png)
![TimbreVAE Training Pipeline](/Documentation/Figures/Timbre-VAE_Training.png)

This is the proposed decoder that makes use of the Timbre encoder:
![Decoder Training Pipeline](/Documentation/Figures/Decoder_Training.png)
