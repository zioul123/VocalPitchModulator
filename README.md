# Vocal Pitch Modulator
A vocal pitch modulator that uses Machine Learning for realistic voice change. This is a project for NUS's CS4347 (Sound and Music Computing).

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisities
Make sure you have the right environment to run. <br/>
Run  the `requirements.txt` file to install neccessary modules. <br/>

For anaconda environment: <br/>

    conda install requirements.txt
 
## Motivation
The goal of vocal pitch modulation in this project is to maintain a “realistic” sound when the pitch is changed. When we use conventional modulation techniques, increasing the pitch of an audio file by more than around 3 semitones tends to make people sound like chipmunks, while lowering the pitch by more than around 3 semitones makes them sound demonic/dopey. However, there are people who speak at lower and higher pitches without sounding this way, so it is not simply the case that lower/higher pitched voices sound this way, but the spectral characteristics must be adjusted in a suitable manner to keep realism. As such, in this project, we wish to employ machine learning to adjust pitch without losing realism.

## Repository Structure
Figures detailing the implementation of the Vocal Pitch Modulator can be found in the [Documentation](https://github.com/zioul123/VocalPitchModulator/Documentation)  folder.

Tentatively, the API that is to be programmed can be found in [VPM.py](https://github.com/zioul123/VocalPitchModulator/VPM.py). 

The dataset we are training our Artificial Neural Networks with can be found in the [Data/dataset](https://github.com/zioul123/VocalPitchModulator/Data/dataset) folder. Inside the [Data](https://github.com/zioul123/VocalPitchModulator/Data) folder, you will also find the list of files along with the relevant labels in [dataset_files.csv](https://github.com/zioul123/VocalPitchModulator/Data/dataset_files.csv). You will also be able to find the Jupyter Notebooks that were used to generate the dataset, and the raw file list, but we are not including the raw files in this repository, so these will not be for use, but for reference.

The audio files produced by different experiments can be found in [many_expts](https://github.com/zioul123/VocalPitchModulator/many_expts) folder.

## Vocal Pitch Modulation System Design
The following is the proposed modulation pipeline:  
![Vocal Pitch Modulation System Design](/Documentation/Figures/VocalPitchModulationSystemdesign.png)<br/>
Image 1: Overall Vocal Pitch Modulation System Design

Please refer to [Vocal Pitch Modulation Audio and Waveform presentation]() page to listen to our reconstructed results for each methods and experiments we took.
#### The above System Design includes 4 stages:
#### 1. Pre-processing Stage
As seen in Image 1, pre-processing stage converts audio wav file, apply STFT function and create either STFT, Mel-spectrum or MFCC to be used for further processing.
This stage also includes data and pitch pairing for ANN Training which will be further shown in stage 3.
 
#### 2. Pitch Shifting and Training Stage 
As seen in Image 1, pitch shift takes into the mel-spectrum, train and output the pitch shifted mel-spectrum to be fed for further timbre training in stage 3.
 
#### 3. Further Processing and ANN Timbre Conformation Training Pipeline
##### Proposed Architecture 0
![architecture0.png](/Documentation/Figures/architecture0.png)
##### Proposed Architecture 1
![architecture0.png](/Documentation/Figures/architecture1.png)
##### Proposed Architecture 2
![architecture0.png](/Documentation/Figures/architecture2.png)

Architecture 2 has an additional Timbre Encoder ANN and the pipline is as follows:
![Encoder Training Pipeline](/Documentation/Figures/Timbre-Enc_Training.png)
![TimbreVAE Training Pipeline](/Documentation/Figures/Timbre-VAE_Training.png)

##### Proposed Architecture 3
![architecture0.png](/Documentation/Figures/architecture3.png)


#### 4. Post-processing stage
As seen in Image 1, post-processing stage reconstruct the STFT function to give our output result.
 

## Data Organization
For a walkthrough of the typical data processing that we conducted, refer to the [Data Processing for Training Walkthrough](https://github.com/zioul123/VocalPitchModulator/blob/master/Data%20Processing%20for%20Training%20Walkthrough.ipynb).

The following image is the vowel diagram we followed for dataset collection.
![vowel diagram](/Documentation/Figures/voweldiagram.png)


The following are additional aids which illustrates the organization of our data.
![Data List](/Documentation/Figures/Data_List.png)  
![Data-Label Pairs](/Documentation/Figures/Data-Label_Pairs.png)
