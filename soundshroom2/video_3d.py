import acoular
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import wavfile
import tables
import numpy as np
from os import path
import os
import glob
import cv2

# Output folders for intermediate and final results
output_folder = './soundshroom2/output_chunks/'
output_folder_img = './soundshroom2/output_chunks_img/'
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_folder_img, exist_ok=True)

# ----------------------------------------------------------------------

fs, data = wavfile.read('./soundshroom2/recordings/wav_0_12ft_1000Hz_clean.wav')
print("fs, ", fs)
print("data.shape[1], num channels", data.shape[1])

chunk_duration = 0.1  # seconds
samples_per_chunk = int(chunk_duration * fs)

for i in range(0, len(data), samples_per_chunk):
    chunk_num = i // samples_per_chunk
    print("saving chunk number ", chunk_num)
    chunk = data[i:i + samples_per_chunk]

    if chunk.shape[0] < samples_per_chunk:
        break

    name = f"1000Hz_chunk_{chunk_num:05}.h5"
    acoularh5 = tables.open_file(output_folder + name, mode="w", title=name)
    acoularh5.create_earray('/', 'time_data', atom=tables.Float32Atom(), title='', 
                            filters=None, expectedrows=chunk.shape[0], 
                            chunkshape=None, 
                            byteorder=None, createparents=False, obj=chunk.astype(np.float32))
    acoularh5.set_node_attr('/time_data', 'sample_freq', fs)
    acoularh5.close()

# ----------------------------------------------------------------------


# Define microphone geometry from XML file
mg = acoular.MicGeom(from_file="./soundshroom2/mic_array.xml")

# 3D grid setup
rg = acoular.RectGrid3D(
    x_min=-60, x_max=60, 
    y_min=-60, y_max=60, 
    z_min=-3, z_max=10, 
    increment=0.5
)

# Steering vector for 3D beamforming
st = acoular.SteeringVector(grid=rg, mics=mg, steer_type="classic")

source_locations = []

# Process each chunk of the audio file
for chunk_file in sorted(os.listdir(output_folder)):
    ts = acoular.TimeSamples(name=os.path.join(output_folder, chunk_file))
    ps = acoular.PowerSpectra(time_data=ts, block_size=128, window="Hanning")
    
    # Functional Beamformer with 3D support
    bb = acoular.BeamformerFunctional(freq_data=ps, steer=st, gamma=8)
    
    # Compute synthetic data
    pm = bb.synthetic(1000, 12)  # Frequency = Hz, 3rd octave
    Lm = acoular.L_p(pm)  # Convert to dB

    # Find the location of the maximum intensity
    max_idx = np.unravel_index(np.argmax(Lm, axis=None), Lm.shape)
    source_location = (
        rg.x_min + max_idx[0] * rg.increment,
        rg.y_min + max_idx[1] * rg.increment,
        rg.z_min + max_idx[2] * rg.increment
    )

    print(f"Sound source location in {chunk_file}: {source_location}")

    source_locations.append(source_location)

    # 3D Visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    # Plot the sound source as a red point
    ax.scatter(*source_location, c="red", s=100, label="Sound Source")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Z Position (m)")
    ax.legend()
    ax.set_title(f"Sound Source Location - {chunk_file}")
    ax.set_xlim(rg.x_min, rg.x_max)
    ax.set_ylim(rg.y_min, rg.y_max)
    ax.set_zlim(rg.z_min, rg.z_max)
    
    # Save the 3D plot as an image
    image_path = os.path.join(output_folder_img, f"sound_source_{chunk_file}.png")
    plt.savefig(image_path)
    plt.close()

# Compute the average sound source location
average_location = np.mean(source_locations, axis=0)
print(f"Average sound source location: X={average_location[0]:.2f}, Y={average_location[1]:.2f}, Z={average_location[2]:.2f}")


# Combine images into a video
output_video = './soundshroom2/videos/3d_12ft_1000Hz_clean_2.mp4' #---------------------------------------------
images = sorted(glob.glob(os.path.join(output_folder_img, "*.png")))
fps = 10

frame = cv2.imread(images[0])
height, width, _ = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

for image_path in images:
    frame = cv2.imread(image_path)
    video_writer.write(frame)

video_writer.release()
print(f"Video saved as {output_video}")

# Combine video with audio using FFmpeg
audio_file = './soundshroom2/recordings/wav_0_12ft_1000Hz_clean.wav'
final_output_video = './soundshroom2/videos/3d_12ft_1000Hz_clean_audio_2.mp4' #---------------------------------------------
ffmpeg_command = (
    f"ffmpeg -i {output_video} -i {audio_file} -map 0:v -map 1:a "
    f"-c:v copy -c:a aac -b:a 192k -ac 2 -strict experimental {final_output_video}"
)
os.system(ffmpeg_command)
print(f"Final video with audio saved as {final_output_video}")



# -----------------------------------------------------------------------------------------------------------


# import acoular
# import matplotlib.pyplot as plt
# import os
# import numpy as np

# # Output folders for results
# output_folder = './soundshroom/output_chunks/'
# output_folder_img = './soundshroom/output_chunks_img/'
# os.makedirs(output_folder_img, exist_ok=True)

# # Define microphone geometry from XML file
# mg = acoular.MicGeom(from_file="./soundshroom/mic_array.xml")

# # 3D grid setup
# rg = acoular.RectGrid3D(
#     x_min=-200, x_max=200,
#     y_min=-200, y_max=200,
#     z_min=0, z_max=10,
#     increment=0.5  # Coarser resolution for faster processing
# )

# # Steering vector for 3D beamforming
# st = acoular.SteeringVector(grid=rg, mics=mg, steer_type="classic")

# # Process each chunk of the audio file
# for chunk_file in sorted(os.listdir(output_folder)):
#     # Load audio chunk
#     ts = acoular.TimeSamples(name=os.path.join(output_folder, chunk_file))
#     ps = acoular.PowerSpectra(time_data=ts, block_size=128, window="Hanning")

#     # Functional Beamformer
#     bb = acoular.BeamformerFunctional(freq_data=ps, steer=st, gamma=8)

#     # Beamforming power map (synthetic data)
#     pm = bb.synthetic(1800, 3)  # Frequency = 1800 Hz, 3rd octave
#     Lm = acoular.L_p(pm)  # Convert to dB

#     # Find grid index of the maximum power
#     max_idx = np.unravel_index(np.argmax(Lm, axis=None), Lm.shape)
    
#     # Convert index to real-world coordinates
#     x_max = rg.x_min + max_idx[0] * rg.increment
#     y_max = rg.y_min + max_idx[1] * rg.increment
#     z_max = rg.z_min + max_idx[2] * rg.increment

#     # Plot the sound source location
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection="3d")
    
#     # Plot the single sound source location as one large red dot
#     ax.scatter([x_max], [y_max], [z_max], color="red", s=300, marker="o", label="Sound Source")
    
#     # Set axis limits to match the grid range for consistency
#     ax.set_xlim([rg.x_min, rg.x_max])
#     ax.set_ylim([rg.y_min, rg.y_max])
#     ax.set_zlim([rg.z_min, rg.z_max])
    
#     ax.set_xlabel("X Position (m)")
#     ax.set_ylabel("Y Position (m)")
#     ax.set_zlabel("Z Position (m)")
#     ax.set_title(f"Sound Source Location - {chunk_file}")
#     ax.legend()
    
#     # Save the plot
#     image_path = os.path.join(output_folder_img, f"sound_source_{chunk_file}.png")
#     plt.savefig(image_path)
#     plt.close()

#     bb.h5f.close()
#     ts.h5f.close()


# import acoular
# import matplotlib.pyplot as plt
# from scipy.io import wavfile
# import os
# import numpy as np

# # Output folders for results
# output_folder = './soundshroom/output_chunks/'
# output_folder_img = './soundshroom/output_chunks_img/'
# os.makedirs(output_folder_img, exist_ok=True)

# # Define microphone geometry from XML file
# mg = acoular.MicGeom(from_file="./soundshroom/mic_array.xml")

# # 3D grid setup
# rg = acoular.RectGrid3D(
#     x_min=-200, x_max=200,
#     y_min=-200, y_max=200,
#     z_min=0, z_max=10,
#     increment=0.5  # Coarser resolution for faster processing
# )

# # Steering vector for 3D beamforming
# st = acoular.SteeringVector(grid=rg, mics=mg, steer_type="classic")

# # Process each chunk of the audio file
# for chunk_file in sorted(os.listdir(output_folder)):
#     # Load audio chunk
#     ts = acoular.TimeSamples(name=os.path.join(output_folder, chunk_file))
#     ps = acoular.PowerSpectra(time_data=ts, block_size=128, window="Hanning")

#     # Functional Beamformer
#     bb = acoular.BeamformerFunctional(freq_data=ps, steer=st, gamma=8)

#     # Beamforming power map (synthetic data)
#     pm = bb.synthetic(1800, 3)  # Frequency = 1800 Hz, 3rd octave
#     Lm = acoular.L_p(pm)  # Convert to dB

#     # Find grid index of the maximum power
#     max_idx = np.unravel_index(np.argmax(Lm, axis=None), Lm.shape)
    
#     # Convert index to real-world coordinates
#     x_max = rg.x_min + max_idx[0] * rg.increment
#     y_max = rg.y_min + max_idx[1] * rg.increment
#     z_max = rg.z_min + max_idx[2] * rg.increment

#     # Plot the sound source location
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection="3d")
#     ax.scatter([x_max], [y_max], [z_max], color="red", s=100, label="Sound Source")
    
#     ax.set_xlabel("X Position (m)")
#     ax.set_ylabel("Y Position (m)")
#     ax.set_zlabel("Z Position (m)")
#     ax.set_title(f"Sound Source Location - {chunk_file}")
#     ax.legend()
    
#     # Save the plot
#     image_path = os.path.join(output_folder_img, f"sound_source_{chunk_file}.png")
#     plt.savefig(image_path)
#     plt.close()


# print("Sound source localization plots saved.")

# # Combine images into a video
# output_video = './soundshroom/videos/sound_source_video_3d_6.mp4' #---------------------------------------------
# images = sorted(glob.glob(os.path.join(output_folder_img, "*.png")))
# fps = 10

# frame = cv2.imread(images[0])
# height, width, _ = frame.shape

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# for image_path in images:
#     frame = cv2.imread(image_path)
#     video_writer.write(frame)

# video_writer.release()
# print(f"Video saved as {output_video}")

# # Combine video with audio using FFmpeg
# audio_file = './soundshroom/recordings/wav_8_car.wav'
# final_output_video = './soundshroom/videos/sound_source_video_3d_with_audio_6.mp4' #---------------------------------------------
# ffmpeg_command = (
#     f"ffmpeg -i {output_video} -i {audio_file} -map 0:v -map 1:a "
#     f"-c:v copy -c:a aac -b:a 192k -ac 2 -strict experimental {final_output_video}"
# )
# os.system(ffmpeg_command)
# print(f"Final video with audio saved as {final_output_video}")








# import acoular
# import matplotlib.pyplot as plt
# import os
# import numpy as np

# # Output folders for results
# output_folder = './soundshroom/output_chunks/'
# output_folder_img = './soundshroom/output_chunks_img/'
# os.makedirs(output_folder_img, exist_ok=True)

# # Define microphone geometry from XML file
# mg = acoular.MicGeom(from_file="./soundshroom/mic_array.xml")

# # 3D grid setup
# rg = acoular.RectGrid3D(
#     x_min=-150, x_max=150,
#     y_min=-150, y_max=150,
#     z_min=0, z_max=30,
#     increment=1  # Coarser resolution for faster processing
# )

# # Steering vector for 3D beamforming
# st = acoular.SteeringVector(grid=rg, mics=mg, steer_type="classic")

# # Process each chunk of the audio file
# for chunk_file in sorted(os.listdir(output_folder)):
#     # Load audio chunk
#     ts = acoular.TimeSamples(name=os.path.join(output_folder, chunk_file))
#     ps = acoular.PowerSpectra(time_data=ts, block_size=128, window="Hanning")

#     # Functional Beamformer
#     bb = acoular.BeamformerFunctional(freq_data=ps, steer=st, gamma=8)

#     # Beamforming power map (synthetic data)
#     pm = bb.synthetic(1800, 3)  # Frequency = 1800 Hz, 3rd octave
#     Lm = acoular.L_p(pm)  # Convert to dB

#     # Find grid index of the maximum power
#     max_idx = np.unravel_index(np.argmax(Lm, axis=None), Lm.shape)
    
#     # Convert index to real-world coordinates
#     x_max = rg.x_min + max_idx[0] * rg.increment
#     y_max = rg.y_min + max_idx[1] * rg.increment
#     z_max = rg.z_min + max_idx[2] * rg.increment

#     # Plot the sound source location
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection="3d")
    
#     # Plot the single sound source location as one large red dot
#     ax.scatter([x_max], [y_max], [z_max], color="red", s=300, marker="o", label="Sound Source")
    
#     # Set axis limits to match the grid range for consistency
#     ax.set_xlim([rg.x_min, rg.x_max])
#     ax.set_ylim([rg.y_min, rg.y_max])
#     ax.set_zlim([rg.z_min, rg.z_max])
    
#     ax.set_xlabel("X Position (m)")
#     ax.set_ylabel("Y Position (m)")
#     ax.set_zlabel("Z Position (m)")
#     ax.set_title(f"Sound Source Location - {chunk_file}")
#     ax.legend()
    
#     # Save the plot
#     image_path = os.path.join(output_folder_img, f"sound_source_{chunk_file}.png")
#     plt.savefig(image_path)
#     plt.close()

# print("Sound source localization plots saved.")




# # Combine images into a video
# output_video = './soundshroom/videos/sound_source_video_3d_13.mp4'
# images = sorted(glob.glob(os.path.join(output_folder_img, "*.png")))
# fps = 10

# frame = cv2.imread(images[0])
# height, width, _ = frame.shape

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# for image_path in images:
#     frame = cv2.imread(image_path)
#     video_writer.write(frame)

# video_writer.release()
# print(f"Video saved as {output_video}")

# # Combine video with audio using FFmpeg
# audio_file = './soundshroom/recordings/wav_8_car.wav'
# final_output_video = './soundshroom/videos/sound_source_video_3d_with_audio_13.mp4'
# ffmpeg_command = (
#     f"ffmpeg -i {output_video} -i {audio_file} -map 0:v -map 1:a "
#     f"-c:v copy -c:a aac -b:a 192k -ac 2 -strict experimental {final_output_video}"
# )
# os.system(ffmpeg_command)
# print(f"Final video with audio saved as {final_output_video}")