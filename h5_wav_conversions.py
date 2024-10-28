import h5py
import numpy as np
from scipy.io.wavfile import write
import tables
import wave


# h5_file = tables.open_file('./output_folder/audio_0_trim.h5', mode = 'r') 

# print("h5_file.root", h5_file.root)
# print("h5_file.root.time_data", h5_file.root.time_data)

# print("h5_file.root.time_data[:20,1]", h5_file.root.time_data[:20,1])

# def get_wav_channels(filename):
#     with wave.open(filename, 'r') as wav_file:
#         channels = wav_file.getnchannels()
#         return channels

# # Usage
# wav_file = './24y_10m_21d_20h_28m_13s/audio_0.wav'
# channels = get_wav_channels(wav_file)
# print(f"Number of channels: {channels}")




def h5_to_wav(h5_filename, wav_filename):
    h5_file = tables.open_file(h5_filename, mode = 'r') 
    audio_data = h5_file.root.time_data[:]

    sample_rate = 48000  # 48 kHz

    # # If the data is not normalized, scale appropriately based on its format
    # if np.issubdtype(audio_data.dtype, np.integer):
    #     # If integer, assume it's already in PCM format, no scaling needed
    #     audio_data = np.int16(audio_data)
    # elif np.issubdtype(audio_data.dtype, np.floating):
    #     # If floating-point but not normalized, assume a known maximum value
    #     max_val = np.max(np.abs(audio_data))  # Find the maximum absolute value
    #     audio_data = np.int16(audio_data / max_val * 32767)  # Scale to int16
    # else:
    #     raise ValueError("Unexpected data type in H5 file")

    audio_data = np.int16(audio_data)

    # Write the data to a WAV file
    write(wav_filename, sample_rate, audio_data)

# Usage
h5_to_wav('./output_chunks_5000/audio_0_chunk_5000_957.h5', './output_folder/output_audio_0_chunk_5000_957.wav')

h5_to_wav('./output_chunks_5000/audio_0_chunk_5000_689.h5', './output_folder/output_audio_0_chunk_5000_689.wav')




