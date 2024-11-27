import acoular
import matplotlib.pylab as plt
from scipy.io import wavfile
import tables
import numpy as np
from os import path
import os
import glob
import cv2


output_folder = './soundshroom/output_chunks/'
output_folder_img = './soundshroom/output_chunks_img/'
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_folder_img, exist_ok=True)

# # ------------------------------------------------------

fs, data = wavfile.read('./soundshroom/recordings/wav_8_car.wav')
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

    name = f"2000Hz_chunk_{chunk_num:05}.h5"
    acoularh5 = tables.open_file(output_folder + name, mode="w", title=name)
    acoularh5.create_earray('/', 'time_data', atom=tables.Float32Atom(), title='', 
                            filters=None, expectedrows=chunk.shape[0], 
                            chunkshape=None, 
                            byteorder=None, createparents=False, obj=chunk.astype(np.float32))
    acoularh5.set_node_attr('/time_data', 'sample_freq', fs)
    acoularh5.close()

# --------------------------------------------


# # Define microphone spacing in meters (144 mm = 0.144 m)
# mg = acoular.MicGeom( from_file="./soundshroom/mic_array.xml" )
# # plot sounds
# rg = acoular.RectGrid(x_min=-200, x_max=200, y_min=-200, y_max=200, z=0.1, increment=0.1)

# st = acoular.SteeringVector( grid=rg, mics=mg, steer_type="classic")


# # # Plot microphone positions
# # plt.figure()
# # if mg.mpos.size > 0:
# #     plt.plot(mg.mpos[0], mg.mpos[1], 'o')
# #     plt.axis('equal')
# #     plt.xlabel('X Position (m)')
# #     plt.ylabel('Y Position (m)')
# #     plt.title('Microphone Positions')
# #     plt.show(block=False)
# # else:
# #     print("No microphone positions available.")
# # plt.show(block=True)


# for chunk_file in sorted(os.listdir(output_folder)):
#     # start beamforming analysis
#     ts = acoular.TimeSamples( name=os.path.join(output_folder, chunk_file) )

#     # to do beamforming in freq domain
#     # ps = acoular.PowerSpectra( time_data=ts, block_size=128, window="Hanning" )
#     ps = acoular.PowerSpectra( time_data=ts, block_size=128, window="Hanning" )
#     # bin spacing = sample rate/block size, 48000 = 250 hz / 192


#     # bb = acoular.BeamformerBase(freq_data=ps, steer=st)
#     # bb = acoular.BeamformerCapon(freq_data=ps, steer=st)
#     bb = acoular.BeamformerFunctional(freq_data=ps, steer=st, gamma=8)


#     pm = bb.synthetic(1800, 3)
#     Lm = acoular.L_p(pm)

#     # Save beamforming map as image
#     plt.figure()
#     plt.imshow(Lm.T, origin="lower", vmin=Lm.max() - 5, extent=rg.extend(), interpolation='bicubic')
#     plt.colorbar()
#     plt.title(f"Beamforming Map - {chunk_file}")
#     plt.xlabel("X Position (m)")
#     plt.ylabel("Y Position (m)")
#     image_path = os.path.join(output_folder_img, f"beamforming_{chunk_file}.png")
#     plt.savefig(image_path)
#     plt.close()

    

# # --------------------------------------------

# # combine into video

# chunk_duration = 0.1  # seconds
# fs = 32000

# output_video = './soundshroom/videos/beamforming_output_video_5fps_car_18.mp4'

# images = sorted(glob.glob(os.path.join(output_folder_img, "*.png")))

# fps = 10


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

# audio_file = './soundshroom/recordings/wav_8_car.wav'
# output_video = './soundshroom/videos/beamforming_output_video_5fps_car_18.mp4'
# final_output_video = './soundshroom/videos/beamforming_output_video_5fps_audio_car_18.mp4'
# #Combine video with audio using ffmpegs
# # Note: This command needs ffmpeg installed and in the system PATH.
# # ffmpeg_command = f"ffmpeg -i {output_video} -i {audio_file} -c:v copy -c:a aac -strict experimental {final_output_video}"
# ffmpeg_command = (
#     f"ffmpeg -i {output_video} -i {audio_file} -map 0:v -map 1:a "
#     f"-c:v copy -c:a aac -b:a 192k -ac 2 -strict experimental {final_output_video}"
# )


# # Run the ffmpeg command
# os.system(ffmpeg_command)

# print(f"Final video with audio saved as {final_output_video}")




# # --------------------------------------------
