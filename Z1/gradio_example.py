import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import gradio as gr


def spectrogram(audio):
    sr, data = audio
    if len(data.shape) == 2:
        data = np.mean(data, axis=0)
    frequencies, times, spectrogram_data = signal.spectrogram(data, sr, window="hamming")
    plt.pcolormesh(times, frequencies, np.log10(spectrogram_data))
    return plt, audio


iface = gr.Interface(spectrogram,
                     inputs="microphone",
                     outputs=["plot", "audio"],
                     title="Voice spectrogram",
                     description="Simple app to create voice spectrogram from microphone input")
iface.launch()
