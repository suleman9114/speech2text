import torch
from denoiser import pretrained
import soundfile as sf
from denoiser.dsp import convert_audio
import speech_recognition as sr
import numpy as np
from miniaudio import SampleFormat, decode
from vosk import Model, KaldiRecognizer


def get_audio_mic():
    
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
    
        print("Please say something")
    
        audio = r.listen(source)
    return audio

def audio_decode(audio):
    data = audio.get_wav_data()
    
    decoded_audio = decode(data, nchannels=1, sample_rate=audio.sample_rate, output_format=SampleFormat.SIGNED16)
    decoded_audio = torch.FloatTensor(decoded_audio.samples)
    decoded_audio /= (1 << 16)
    decoded_audio = decoded_audio.reshape(1, len(decoded_audio))
    return decoded_audio

def denoising(decoded_audio, audio):
    wav = convert_audio(decoded_audio, audio.sample_rate, model.sample_rate, model.chin)
    
    
    with torch.no_grad():
        denoised = model(wav[None])[0]
    
    d_b = wav.data.numpy().transpose()
    d_b1 = denoised.data.numpy().transpose()
    
    sf.write("new_file.wav", d_b, model.sample_rate) 
    sf.write("denoised.wav", d_b1, model.sample_rate)
    return d_b1

def google_speech2txt(d_b1):
    global audio_source
    scaled = np.int16(d_b1.transpose() / np.max(np.abs(d_b1.transpose())) * 32767)
    
    
    audio_source = sr.AudioData(scaled, 16000, 2)
    try:    
     	recog = r.recognize_google(audio_source, language = 'en-US')    
     	print("From Google, You said: " + recog)
    except sr.UnknownValueError:    
     	print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:   
     	print("Could not request results from Google Speech Recognition service; {0}".format(e))
    return recog
         
def vosk_speech2txt():
    recognizer.AcceptWaveform(audio_source.get_wav_data())
    text = recognizer.Result()
    print(f"From Vosk, You said: {text[14:-3]}")
    print("done\n\n\n")
    return text


model2 = Model(r"C:\Users\WPC-2\Downloads\Compressed\Models\vosk-model-en-in-0.5")
recognizer = KaldiRecognizer(model2, 16000)


model = pretrained.dns64()
r = sr.Recognizer()

while True:
    audio = get_audio_mic()
    decoded_audio = audio_decode(audio)
    denoised_audio = denoising(decoded_audio, audio)
    google_txt = google_speech2txt(denoised_audio)
    vosk_txt = vosk_speech2txt()




































