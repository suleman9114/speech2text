import torch
from denoiser import pretrained
import soundfile as sf
from denoiser.dsp import convert_audio
import speech_recognition as sr
import numpy as np
from miniaudio import SampleFormat, decode
from vosk import Model, KaldiRecognizer

model2 = Model(r"C:\Users\WPC-2\Downloads\Compressed\vosk-model-small-en-in-0.4")
recognizer = KaldiRecognizer(model2, 16000)

# load data

model = pretrained.dns64()
r = sr.Recognizer()
while True:
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
    
        print("Please say something")
    
        audio = r.listen(source)
    
    
    data = audio.get_wav_data()
    
    decoded_audio = decode(data, nchannels=1, sample_rate=audio.sample_rate, output_format=SampleFormat.SIGNED16)
    decoded_audio = torch.FloatTensor(decoded_audio.samples)
    decoded_audio /= (1 << 16)
    decoded_audio = decoded_audio.reshape(1, len(decoded_audio))
    
    
    wav = convert_audio(decoded_audio, audio.sample_rate, model.sample_rate, model.chin)
    
    
    with torch.no_grad():
        denoised = model(wav[None])[0]
    
    d_b = wav.data.numpy().transpose()
    d_b1 = denoised.data.numpy().transpose()
    
    
    sf.write("new_file.wav", d_b, model.sample_rate) 
    sf.write("denoised.wav", d_b1, model.sample_rate)
    
    
    from IPython.display import Audio
    cc = Audio(d_b1.transpose(), rate=16000)
    
    
    
    #raw_audio = np.float16(d_b1).tobytes()
    
    
    with sr.AudioFile("denoised.wav") as source:
        audio_source = r.record(source)
        try:    
         	recog = r.recognize_google(audio_source, language = 'en-US')    
         	print("From Google, You said: " + recog)
        except sr.UnknownValueError:    
         	print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:   
         	print("Could not request results from Google Speech Recognition service; {0}".format(e))
             
        
        recognizer.AcceptWaveform(audio_source.get_wav_data())
        text = recognizer.Result()
        print(f"From Google, You said: {text[14:-3]}")
        print("done\n\n\n")










































