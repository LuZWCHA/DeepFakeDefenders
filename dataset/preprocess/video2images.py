import glob
import os
from pathlib import Path
import random
from matplotlib import pyplot as plt
import ffmpeg, tqdm
# 43721
output_dir = "/nasdata2/private/lzhao/workspace/kaggle/Deepfake/dataset/video/phase1/trainset_frames/images"


def get_video_info(source_video_path):
    probe = ffmpeg.probe(source_video_path)
    # print('source_video_path: {}'.format(source_video_path))
    format = probe['format']
    bit_rate = int(format['bit_rate'])/1000
    duration = format['duration']
    size = int(format['size'])/1024/1024
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    if video_stream is None:
        print('No video stream found!')
        return
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    num_frames = int(video_stream['nb_frames'])
    fps = int(video_stream['r_frame_rate'].split('/')[0])/int(video_stream['r_frame_rate'].split('/')[1])
    duration = float(video_stream['duration'])
    # print('width: {}'.format(width))
    # print('height: {}'.format(height))
    # print('num_frames: {}'.format(num_frames))
    # print('bit_rate: {}k'.format(bit_rate))
    # print('fps: {}'.format(fps))
    # print('size: {}MB'.format(size))
    # print('duration: {}'.format(duration))
    return duration

videos = glob.glob("/nasdata2/private/lzhao/workspace/kaggle/Deepfake/dataset/video/phase1/trainset/*.mp4")

# duration_list = []
# for input_file_name in tqdm.tqdm(videos):
#     duration = get_video_info(input_file_name)
#     duration_list.append(duration)

# plt.hist(duration_list, bins=60, log=True)
# plt.show()
# plt.savefig("duration_list.png")

# return

# 11334
fps = 5
for input_file_name in tqdm.tqdm(videos[43720 + 30972:]):
    video_stem = Path(input_file_name).stem
    ffmpeg.input(input_file_name) \
            .filter('fps', fps=fps, round = 'up') \
            .output(f"{os.path.join(output_dir, video_stem)}-%04d.jpg", **{'qscale:v': 3}, loglevel="quiet")\
            .global_args("-n") \
            .run()
 
output_dir = "/nasdata2/private/lzhao/workspace/kaggle/Deepfake/dataset/video/phase1/valset_frames/images"
fps = 10
for input_file_name in tqdm.tqdm(glob.glob("/nasdata2/private/lzhao/workspace/kaggle/Deepfake/dataset/video/phase1/valset/*.mp4")):
    video_stem = Path(input_file_name).stem
    ffmpeg.input(input_file_name)\
            .filter('fps', fps=fps, round = 'up')\
            .output(f"{os.path.join(output_dir, video_stem)}-%04d.jpg", **{'qscale:v': 3}, loglevel="quiet")\
            .run()