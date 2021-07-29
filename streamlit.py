import streamlit as st
import librosa
import librosa.display
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import tempfile
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
import requests
from config import DEEPL
import datetime
BASEFILE = "audio.wav"
PREFIX = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
SAMPLERATE = 16000  
DURATION = 10 # seconds
LANG = {"English":"facebook/wav2vec2-base-960h","German":"facebook/wav2vec2-large-xlsr-53-german","Italian":"facebook/wav2vec2-large-xlsr-53-italian"} 
TRANSLATE = {"English":"EN","German":"DE","Italian":"IT"}
CHOICE_LIST = {"Record audio":0,"Upload audio(.wav)":1}

@st.cache
def speech2text(file_name):
    tokenizer = Wav2Vec2Processor.from_pretrained(LANG[lang_selected])
    model = Wav2Vec2ForCTC.from_pretrained(LANG[lang_selected])
    speech, rate = librosa.load(file_name,sr=16000,duration = DURATION)
    input_values = tokenizer(speech,sampling_rate=16000,return_tensors='pt').input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits,dim=-1)
    transcript = tokenizer.decode(predicted_ids[0])
    return speech, transcript

def translate_deepl(text,lang_to_translate):
    url = "https://api-free.deepl.com/v2/translate"
    body = {
        "text": text,
        "auth_key": DEEPL['API_KEY'],
        "target_lang": TRANSLATE[lang_to_translate],
    }

    response = requests.post(url, data=body)
    translated_text = response.json()["translations"][0]["text"]
    return translated_text

def wave_plot(speech):
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(211)
    ax.set_title('Raw wave of ' + 'audio.wav')
    ax.set_ylabel('Amplitude')
    ax.plot = librosa.display.waveplot(speech, sr=16000)
    return st.write(fig)

if __name__=="__main__":
    
    st.title("Welcome to Voice Translator -  Speech-to-Text")
    lang_selected = st.selectbox('Select the language you speak', LANG.keys())
    
    lang_translate=[]
    for i in range(len(LANG.keys())):
        if list(LANG)[i] != lang_selected:
            lang_translate.append(list(LANG)[i])
    
    lang_tranlate_selection = st.selectbox('Select the language for translation', lang_translate)
    
    choice = st.radio("Please select from the two options",CHOICE_LIST.keys())
    
    try:
        if CHOICE_LIST[choice] == 0:
            FILENAME  = "_".join([PREFIX, BASEFILE])
            if st.button('Record'):
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                with st.spinner(f'Recording for {DURATION} seconds ....'):
                    mydata = sd.rec(int(SAMPLERATE * DURATION), samplerate=SAMPLERATE,
                        channels=1, blocking=True)
                    sd.wait()
                    sf.write(FILENAME, mydata, SAMPLERATE)
                st.success("Recording completed")
                try:
                    audio_file = open(FILENAME, 'rb')
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/wav')

                    transcript = speech2text(FILENAME)[1]
                    st.write("Translated text : ",transcript)
                    
                    st.write("Waveform of audio")
                    wave_plot(speech2text(FILENAME)[0])
                    
                    translated = translate_deepl(transcript,lang_tranlate_selection)
                    st.write("Translated text: ",translated)
        
                    
                except:
                    st.write("Please record audio first")
                   
        
        if CHOICE_LIST[choice] == 1:
            
            st.subheader("Upload a audio file in .wav format")
            st.write("Please upload audio in the language selected above")
            audio_upload = st.file_uploader("")
            
            if audio_upload is not None:
                
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(audio_upload.read())
                
                audio_file = open(tfile.name, 'rb')
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/wav')
        
                transcript_uploaded = speech2text(tfile.name)[1]
                st.write("Audio Transcript : ",transcript_uploaded)
                
                st.write("Waveform of audio")
                wave_plot(speech2text(tfile.name)[0])
                
                translated_uploaded = translate_deepl(transcript_uploaded,lang_tranlate_selection)
                st.write("Translated text: ",translated_uploaded)
    except:
        st.write("Choose an option")
            