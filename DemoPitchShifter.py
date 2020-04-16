import os
import csv
import numpy as np
import VPM
import Utils
import pyaudio
import wave
import librosa

def load_wav_file_from_stdin():

    filename = input("Enter the filename including the .wav extension: ")
    return load_wav_file(filename)


def load_wav_file(filename):

    filenameList = [filename]

    all_wav_data = VPM.load_wav_files("./", filenameList)

    wav_data = all_wav_data[0]
    print(wav_data)

    return wav_data


def record_wav_file():
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1
    fs = 44100  # Record at 44100 samples per second
    seconds = 5

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    print('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for 3 seconds
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    print('Finished recording')

    # Save the recorded data as wav file
    wf = wave.open("output.wav", 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

    # load the saved wavfile
    return load_wav_file("output.wav")


def save_wav_file(file):

    filename = input("Enter the filename including the .wav extension: ")
    librosa.output.write_wav('output_wav/' + filename, file, 44100)

