import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from scipy import signal
from scipy.io import wavfile

st.title('Voice spectrogram')
st.text("Little example code to create voice spectrogram")
fileObject = st.file_uploader(label="Please upload your file")
if fileObject:
    sample_rate, samples = wavfile.read(fileObject)
    frequencies, times, spectrogram_data = signal.spectrogram(samples, sample_rate)
    fig, ax = plt.subplots()
    ax = plt.pcolormesh(times, frequencies, np.log10(spectrogram_data))
    st.pyplot(fig)
