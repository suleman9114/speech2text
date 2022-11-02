# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 12:00:22 2022

@author: WPC-2
"""

import torch
from denoiser import pretrained
import soundfile as sf
from denoiser.dsp import convert_audio
import speech_recognition as sr

from miniaudio import SampleFormat, decode

# load data

model = pretrained.dns64()
r = sr.Recognizer()

with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source)

    print("Please say something")

    audio = r.listen(source)


data = audio.get_wav_data()

decoded_audio = decode(data, nchannels=1, sample_rate=audio.sample_rate, output_format=SampleFormat.SIGNED16)
decoded_audio = torch.FloatTensor(decoded_audio.samples)
decoded_audio /= (1 << 15)
decoded_audio = decoded_audio.reshape(1, len(decoded_audio))


wav = convert_audio(decoded_audio, audio.sample_rate, model.sample_rate, model.chin)


with torch.no_grad():
    denoised = model(wav[None])[0]

d_b = wav.data.numpy().transpose()
d_b1 = denoised.data.numpy().transpose()


sf.write("new_file.wav", d_b, model.sample_rate) 
sf.write("denoised.wav", d_b1, model.sample_rate)


