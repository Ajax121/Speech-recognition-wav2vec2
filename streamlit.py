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
from gtts import gTTS
import datetime

BASEFILE = "audio.wav"
OUTFILE = "audio_output.wav"
PREFIX = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
SAMPLERATE = 16000  
DURATION = 10 # seconds
LANG = {"English":"facebook/wav2vec2-base-960h","German":"facebook/wav2vec2-large-xlsr-53-german"} 
TRANSLATE = {"English":"en","German":"de"}
CHOICE_LIST = {"Record audio":0,"Upload audio(.wav)":1}

@st.cache
def speech_to_text(speech):
    tokenizer = Wav2Vec2Processor.from_pretrained(LANG[lang_selected])
    model = Wav2Vec2ForCTC.from_pretrained(LANG[lang_selected])
    input_values = tokenizer(speech,sampling_rate=SAMPLERATE,return_tensors='pt').input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits,dim=-1)
    transcript = tokenizer.decode(predicted_ids[0])
    return transcript

def translate_deepl(text,lang_to_translate):
    url = "https://api-free.deepl.com/v2/translate"
    body = {
        "text": text,
        "auth_key": DEEPL["API_KEY"],
        "target_lang": TRANSLATE[lang_to_translate],
    }

    response = requests.post(url, data=body)
    translated_text = response.json()["translations"][0]["text"]
    return translated_text

def google_voice(translated_text,lang_to_translate):
    tts = gTTS(text = translated_text, lang = TRANSLATE[lang_to_translate])
    file_name = "_".join([PREFIX, OUTFILE])
    tts.save(file_name)
    audio_file = open(file_name, 'rb')
    audio_bytes = audio_file.read()
    return st.audio(audio_bytes, format='audio/wav')

def wave_plot(speech_var):
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(211)
    ax.set_title('Raw wave of ' + 'audio.wav')
    ax.set_ylabel('Amplitude')
    ax.plot = librosa.display.waveplot(speech_var, sr=SAMPLERATE)
    return st.write(fig)

if __name__=="__main__":
    
    st.image("https://developer-blogs.nvidia.com/wp-content/uploads/2019/12/automatic-speech-recognition_updated.png")
    st.title("Welcome to Voice Translator")
    st.subheader("Speech-to-Text Recognition")
    lang_selected = st.selectbox('Select the language you speak', LANG.keys())
    
    lang_translate=[]
    for i in range(len(LANG.keys())):
        if list(LANG)[i] != lang_selected:
            lang_translate.append(list(LANG)[i])
    
    lang_translate_selection = st.selectbox('Select the language for translation', lang_translate)
    
    choice = st.radio("Please select from the two options",CHOICE_LIST.keys())
    
    try:
        if CHOICE_LIST[choice] == 0:
            FILENAME  = "_".join([PREFIX, BASEFILE])
            if  st.button('Record'):
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

                    speech, rate = librosa.load(FILENAME,sr=SAMPLERATE)
                    st.write("Waveform of audio")
                    wave_plot(speech)
                    speech, rate = librosa.load(FILENAME,sr=SAMPLERATE)
                    transcript_uploaded = speech_to_text(speech)

                    st.write("Audio Transcript : ",transcript_uploaded)
                
                    translated_uploaded = translate_deepl(transcript_uploaded,lang_translate_selection)
                    st.write("Translated text: ",translated_uploaded)
                    st.write('Listen to translated text')
                    google_voice(translated_uploaded,lang_translate_selection)
                except:
                    st.write("Please record audio first")
                    
        
        if CHOICE_LIST[choice] == 1:
            
            st.subheader("Upload a audio file in .wav format")
            st.write("Please upload audio in the language selected above")
            audio_upload = st.file_uploader("")
            
            if audio_upload is not None:
                try:
                    tfile = tempfile.NamedTemporaryFile(delete=False) 
                    tfile.write(audio_upload.read())
                    
                    audio_file = open(tfile.name, 'rb')
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/wav')
            
                    speech, rate = librosa.load(tfile.name,sr=SAMPLERATE)
                    if st.button('View audio waveform'):
                        st.write("Waveform of audio")
                        wave_plot(speech)
                    if st.button('Translate'):
                        transcript_uploaded = speech_to_text(speech)

                        st.write("Audio Transcript : ",transcript_uploaded)
                        
                        translated_uploaded = translate_deepl(transcript_uploaded,lang_translate_selection)
                        st.write("Translated text: ",translated_uploaded)
                        st.write('Listen to translated text')
                        google_voice(translated_uploaded,lang_translate_selection) 
                except:
                    st.write("Please upload a valid file")
                    

    except:
        st.write("Choose an option")
            
