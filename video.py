import acoular
import matplotlib.pylab as plt
from scipy.io import wavfile
import tables
import numpy as np
from os import path
import os
import glob
import cv2


output_folder = './output_chunks_5000/'
output_folder_img = './output_chunks_img_5000/'
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_folder_img, exist_ok=True)


# --------------------------------------------

fs, data = wavfile.read('./24y_10m_21d_20h_28m_13s/audio_0.wav')


chunk_duration = 0.25  # seconds
samples_per_chunk = int(chunk_duration * fs)

for i in range(0, len(data), samples_per_chunk):
    chunk_num = i // samples_per_chunk
    print("saving chunk number ", chunk_num)
    chunk = data[i:i + samples_per_chunk]

    if chunk.shape[0] < samples_per_chunk:
        break

    name = f"audio_0_chunk_5000_{chunk_num:03}.h5"
    acoularh5 = tables.open_file(output_folder + name, mode="w", title=name)
    acoularh5.create_earray('/', 'time_data', atom=tables.Float32Atom(), title='', 
                            filters=None, expectedrows=chunk.shape[0], 
                            chunkshape=None, 
                            byteorder=None, createparents=False, obj=chunk.astype(np.float32))
    acoularh5.set_node_attr('/time_data', 'sample_freq', fs)
    acoularh5.close()

# --------------------------------------------

# # Define microphone spacing in meters (144 mm = 0.144 m)
# mg = acoular.MicGeom( from_file="mic_array.xml" )
# # plot sounds
# rg = acoular.RectGrid(x_min=-1, x_max=1, y_min=-1, y_max=1, z=0.030, increment=0.001)

# st = acoular.SteeringVector( grid=rg, mics=mg )


# for chunk_file in sorted(os.listdir(output_folder)):
#     # start beamforming analysis
#     ts = acoular.TimeSamples( name=os.path.join(output_folder, chunk_file) )

#     # to do beamforming in freq domain
#     ps = acoular.PowerSpectra( time_data=ts, block_size=128, window="Hanning" )

#     bb = acoular.BeamformerCapon(freq_data=ps, steer=st)
#     pm = bb.synthetic(5000, 0)
#     Lm = acoular.L_p(pm)

#     # Save beamforming map as image
#     plt.figure()
#     plt.imshow(Lm.T, origin="lower", vmin=Lm.max() - 3, extent=rg.extend(), interpolation='bicubic')
#     plt.colorbar()
#     plt.title(f"Beamforming Map - {chunk_file}")
#     plt.xlabel("X Position (m)")
#     plt.ylabel("Y Position (m)")
#     image_path = os.path.join(output_folder_img, f"beamforming_{chunk_file}.png")
#     plt.savefig(image_path)
#     plt.close()

# --------------------------------------------



# # combine into video

# chunk_duration = 0.25  # seconds
# fs = 48000

# output_video = 'beamforming_output_video_4fps_5000Hz.mp4'

# images = sorted(glob.glob(os.path.join(output_folder_img, "*.png")))

# fps = 4


# # Load the first image to get frame dimensions
# frame = cv2.imread(images[0])
# height, width, _ = frame.shape

# # Initialize video writer
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# # Write each image to the video
# for image_path in images:
#     print(image_path)
#     frame = cv2.imread(image_path)
#     video_writer.write(frame)

# # Release video writer
# video_writer.release()

# print(f"Video saved as {output_video}")

# audio_file = './24y_10m_21d_20h_28m_13s/audio_0.wav'
# output_video = 'beamforming_output_video_4fps_5000Hz.mp4'
# final_output_video = 'final_beamforming_video_with_audio.mp4'
# #Combine video with audio using ffmpeg
# # Note: This command needs ffmpeg installed and in the system PATH.
# ffmpeg_command = f"ffmpeg -i {output_video} -i {audio_file} -c:v copy -c:a aac -strict experimental {final_output_video}"

# # Run the ffmpeg command
# os.system(ffmpeg_command)

# print(f"Final video with audio saved as {final_output_video}")




# --------------------------------------------


# # Read data from stereo WAV file
# fs, data = wavfile.read('./24y_10m_21d_20h_28m_13s/audio_0_trim.wav')
# print("fs", fs)
# print("data.shape[0]", data.shape[0])

# # Ensure data is in the correct shape (two channels)
# if data.ndim == 1:
#     # If it's mono, replicate the channel
#     data = np.stack([data, data], axis=-1)

# # Output
# folder = './output_folder/'
# name = 'audio_0_trim.h5'

# # Save to Acoular HDF5 format
# acoularh5 = tables.open_file(folder + name, mode="w", title=name)
# acoularh5.create_earray('/', 'time_data', atom=tables.Float32Atom(), title='', 
#                          filters=None, expectedrows=data.shape[0], 
#                          chunkshape=None, 
#                          byteorder=None, createparents=False, obj=data.astype(np.float32))
# acoularh5.set_node_attr('/time_data', 'sample_freq', fs)
# acoularh5.close()

# print(acoularh5)


# # Read data from WAV
# fs, data = wavfile.read('./24y_10m_21d_20h_28m_13s/audio_0.wav')

# # Ensure data is in the correct shape (handle mono/stereo cases)
# if data.ndim == 1:
#     data = data[:, np.newaxis]  # Convert mono to shape (samples, 1)

# # Convert data to float32 for Acoular
# data = data.astype(np.float32)

# # Output
# folder = './output_folder/'
# name = 'audio_0_NEW.h5'

# # Save to Acoular HDF5 format
# acoularh5 = tables.open_file(folder + name, mode="w", title=name)
# acoularh5.create_earray('/', 'time_data', atom=tables.Float32Atom(), title='', 
#                          filters=None, expectedrows=data.shape[0], 
#                          chunkshape=(256, data.shape[1]), obj=data)
# acoularh5.set_node_attr('/time_data', 'sample_freq', fs)
# acoularh5.close()





# # start beamforming analysis
# ts = acoular.TimeSamples( name='./output_folder/audio_0_NEW.h5' )

# # to do beamforming in freq domain
# ps = acoular.PowerSpectra( time_data=ts, block_size=128, window="Hanning" )

# # rg = acoular.RectGrid( x_min=-2, x_max=2, y_min=-2, y_max=2, z=2, increment=0.1 )

# # Define microphone spacing in meters (144 mm = 0.144 m)
# mg = acoular.MicGeom( from_file="mic_array.xml" )
# # Print microphone positions for verification
# print("Microphone positions:\n", mg.mpos)



# # Load microphone geometry from XML
# mg = acoular.MicGeom(from_file="mic_array.xml")

# # Check if microphone positions are loaded correctly
# print("Microphone positions:\n", mg.mpos)

# # Plot microphone positions
# plt.figure()  # Create a new figure for the microphone positions
# if mg.mpos.size > 0:
#     plt.plot(mg.mpos[0], mg.mpos[1], 'o')  # Plot x and y positions
#     plt.axis('equal')
#     plt.xlabel('X Position (m)')
#     plt.ylabel('Y Position (m)')
#     plt.title('Microphone Positions')
#     plt.show(block=False)  # Show the plot and keep it open
# else:
#     print("No microphone positions available.")


# # plot sounds
# rg = acoular.RectGrid(x_min=-1, x_max=1, y_min=-1, y_max=1, z=0.030, increment=0.001)

# st = acoular.SteeringVector( grid=rg, mics=mg )

# bb = acoular.BeamformerCapon(freq_data=ps, steer=st)
# pm = bb.synthetic(1000, 0)
# Lm = acoular.L_p(pm)


# print(Lm.shape)
# print(Lm)

# Plot beamforming map
# plt.figure()  # Create a new figure for the beamforming plot
# plt.imshow(Lm.T, origin="lower", vmin=Lm.max()-15, extent=rg.extend())  # Beamforming result plot
# plt.colorbar()  # Show color scale
# plt.title("Beamforming Map")
# plt.xlabel("X Position (m)")
# plt.ylabel("Y Position (m)")

# Plot beamforming map with bicubic interpolation
# plt.figure()  # Create a new figure for the beamforming plot
# plt.imshow(Lm.T, origin="lower", vmin=Lm.max() - 3,  # 3 dB threshold
#            extent=rg.extend(), interpolation='bicubic')  # Bicubic interpolation for smoother visuals
# plt.colorbar()  # Show color scale
# plt.title("Beamforming Map")
# plt.xlabel("X Position (m)")
# plt.ylabel("Y Position (m)")
# # plt.show()  # Show the plot

# plt.show(block=True)  # Ensure this plot stays open






# import acoular
# import matplotlib.pylab as plt
# from scipy.io import wavfile
# import numpy as np

# # Path to your stereo WAV file
# wav_file = './24y_10m_21d_20h_28m_13s/audio_0_trim.wav'

# # Start beamforming analysis, use WAV file directly
# ts = acoular.TimeSamples(name=wav_file)

# # Perform beamforming in the frequency domain
# ps = acoular.PowerSpectra(time_data=ts, block_size=128, window="Hanning")

# # Define microphone geometry
# mg = acoular.MicGeom(from_file="mic_array.xml")

# # Verify microphone positions
# print("Microphone positions:\n", mg.mpos)

# # Plot microphone positions
# plt.figure()
# if mg.mpos.size > 0:
#     plt.plot(mg.mpos[0], mg.mpos[1], 'o')
#     plt.axis('equal')
#     plt.xlabel('X Position (m)')
#     plt.ylabel('Y Position (m)')
#     plt.title('Microphone Positions')
#     plt.show(block=False)
# else:
#     print("No microphone positions available.")

# # Rectangular grid for sound source localization
# rg = acoular.RectGrid(x_min=-1, x_max=1,
#                       y_min=-1, y_max=1,
#                       z=1, increment=0.005)

# # Create a steering vector and perform beamforming
# st = acoular.SteeringVector(grid=rg, mics=mg)
# bb = acoular.BeamformerBase(freq_data=ps, steer=st)
# pm = bb.synthetic(5000, 3)  # Beamform at 5000 Hz

# # Convert beamforming result to decibel scale
# Lm = acoular.L_p(pm)

# # Print shape and values of the beamforming map
# print(Lm.shape)
# print(Lm)

# # Plot the beamforming map
# plt.figure()
# plt.imshow(Lm.T, origin="lower", vmin=Lm.max() - 15, extent=rg.extend())
# plt.colorbar()
# plt.title("Beamforming Map")
# plt.xlabel("X Position (m)")
# plt.ylabel("Y Position (m)")
# plt.show(block=True)
