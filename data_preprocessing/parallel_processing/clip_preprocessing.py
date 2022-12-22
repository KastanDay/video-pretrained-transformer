import os 
import json
import cv2
import numpy as np
import clip
import torch
from PIL import Image
import pathlib
from pathlib import Path
import glob
import argparse
import jsonlines
import json
import time
import concurrent.futures
import skvideo.io
import imageio.v3 as iio
import av
import lovely_tensors as lt
lt.monkey_patch()

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # just for testing.
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3' # just for testing.


'''
⭐️ How to read the CLIP outputs ⭐️

itterate over arr_0 thru total_segments

path = '/scratch/bbki/kastanday/whisper/parallel_15_clip_output/LdMD528r6Xs_Jon\'s Daily Hustle_802_Lawn Care Equipment Setup Plans For 2021 - Upgrading Lawn Mowers.npz'
np_loaded = np.load(path, allow_pickle=True)
print(np_loaded)
np_loaded['arr_0'].item() # iterate here until `not .next()`

Docs: https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html#numpy.savez_compressed
'''

'''
⭐️ How to read the CLIP outputs ⭐️

itterate over arr_0 thru total_segments

path = '/scratch/bbki/kastanday/whisper/parallel_15_clip_output/LdMD528r6Xs_Jon\'s Daily Hustle_802_Lawn Care Equipment Setup Plans For 2021 - Upgrading Lawn Mowers.npz'
np_loaded = np.load(path, allow_pickle=True)
print(np_loaded)
np_loaded['arr_0'].item() # iterate here until `not .next()`

Docs: https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html#numpy.savez_compressed
'''


'''
INSTALL INSTRUCTIONS (STRICT dependencies, mostly due to Ray.):
conda create -n v3_clip_preprocessing_yt1b python=3.8.13 -y
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install pandas "ray[default]==1.13.0" more_itertools jsonlines pyarrow fastparquet pandas parquet ftfy regex tqdm git+https://github.com/openai/CLIP.git
conda install -c conda-forge -y git-lfs
cd into the git repo and run `git lfs install` and `git lfs pull`
(optional) pip install pretty_errors
'''

### GLOBALS SET ME 😁 ### 
MODEL_SIZE = 'ViT-L/14@336px'  # Best models are (1st) ViT-L/14@336px and (2nd) ViT-L/14. I don't recommend going lower.  
FRAME_SIZE_DIMENSION = 336
NUM_FRAMES_TO_SAVE_PER_SEGMENT = 1

class ClipPreprocessor: 
  def __init__(self, debug=True, num_frames_per_segment=1):
    self.debug = debug
    self.num_frames_per_segment = num_frames_per_segment
    
    # Load the model
    # self.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Delete CPU only code in clip_preprocessing.py")
    self.device = "cpu"
    print(f"Using {self.device}...")

    self.clip, self.clip_preprocess = clip.load(MODEL_SIZE, self.device)
    if '336' in MODEL_SIZE:
        assert FRAME_SIZE_DIMENSION >= 336, "Frame size must be at least 336px (by 336) for ViT-L/14@336px"
    
    if self.debug:
        print(f"Done setting up CLIP...")
        
  def run_clip_one_video(self, sample_dict):
    # sample dict format. It's always 100 segments!
    # sample_dict = {
    # '<full_video_filepath>': {'timestamp': midpoint, 'db_index': idx},   # list of timestamp midpoints (1 per frame)
    # '<full_video_filepath_2>': {'timestamp': midpoint, 'db_index': idx}, # list of timestamp midpoints (1 per frame)
    # }
    
    all_frames = []
    all_timestamps = []
    all_db_indexes = []
    print("getting frames")
    for video_filepath, time_and_db_index in sample_dict.items():
        print("video_filepath", video_filepath)
        print("dict", time_and_db_index)

        # todo: use multithreading (True)
        all_timestamps.extend([segment_dict['timestamp'] for segment_dict in time_and_db_index])
        all_db_indexes.extend([segment_dict['db_index'] for segment_dict in time_and_db_index])
        local_frames = extract_frames_from_video(video_filepath, all_timestamps, use_multithreading=True)
        all_frames.extend(local_frames)
        
        print("type of local_frames", type(local_frames))
        # print(local_frames, "frames extracted from", video_filepath)
        print("local_frames.shape", local_frames.shape) # <batch_size>, 360, 202, 3. <batch_size> is 100 rn.
        print("local_frames[0].shape", local_frames[0].shape)
        
    # resize frames (from roughly 360p --> clip dimensions). Might not be 360p if black bars detected.
    start_time = time.monotonic()
    all_frames = [self.clip_preprocess(Image.fromarray(frame).convert("RGB")) for frame in all_frames]
    print(f"⏰  Runtime of preprocessing: {(time.monotonic() - start_time):.2f} seconds")
    print("Running clip")
    all_pooled_clip_embeds = self.run_clip(all_frames)
    return all_frames, all_pooled_clip_embeds, all_timestamps, all_db_indexes
    
  def run_clip(self, frames):
    '''
    :param frames: list of np.ndarrays
    :returns: np.ndarrays
    '''
    # TODO: Not tested yet.
    image_input = torch.tensor(np.stack(frames)).to(self.device)
    if self.debug:
        print("Shape after stacking frames in run_clip()")
        print(image_input.shape)
        print(image_input)

    # text_inputs = torch.cat(text_inputs).to(self.device)

    print("RIGHT before running clip 📸")
    start_time = time.monotonic()
    with torch.inference_mode(): # even faster than no_grad()
        image_features = self.clip.encode_image(image_input)
        image_features = image_features.cpu().numpy().reshape(len(frames), self.num_frames_per_segment, -1) # -1 == 3.
        # text_features = self.clip.encode_text(text_inputs)
        # text_features = text_features.cpu().numpy()
    print(f"⏰ CLIP Runtime on {len(frames)*self.num_frames_per_segment} images: {(time.monotonic() - start_time):.2f} seconds")
    if self.debug:
        print("Clip features:")
        print(image_features.shape)
        # print(image_features) 

    return image_features # , text_features

'''
VIDEO PROCESSING FROM MERLOT RESERVE
https://github.com/rowanz/merlot_reserve/blob/main/mreserve/preprocess.py
'''
def extract_frames_from_video(video_file, times, use_multithreading=False, blackbar_threshold=32, max_perc_to_trim=.20):
  """
  Extracts multiple things from the video and even handles black bars
  :param video_file: what we are loading
  :param times: timestamps to use
  :param use_multithreading: Whether to use multithreading
  :param use_rgb whether to use RGB (default) or BGR
  :param blackbar_threshold: Pixels must be this intense for us to not trim
  :param max_perc_to_trim: Will trim 20% by default of the image at most in each dimension
  :return: Frames that are trimmed to not have any black bars
  """
  print("In extract frames from video")
  
  container = av.open(video_file)
  video_framerate = container.streams.video[0].average_rate
  
  
  def _extract(i):
    #   return i, extract_single_frame_from_video(video_file, times[i]) # todo: pass in video framerate.. for use in imageio.v3
      return i, kas_extract_single_frame_from_video(video_file, times[i], video_framerate) # todo: pass in video framerate.. for use in imageio.v3

  if not use_multithreading:
      frames = [_extract(i)[1] for i in range(len(times))]
  else:
      frames = [None for t in times]
      with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
          submitted_threads = (executor.submit(_extract, i) for i in range(len(times)))
          for future in concurrent.futures.as_completed(submitted_threads):
              try:
                  i, img = future.result()
                  frames[i] = img
              except Exception as exc:
                  print("Oh no {}".format(str(exc)), flush=True)

  if any([x is None for x in frames]):
      print(f"Fail on {video_file}", flush=True)
      return None

  frames = np.stack(frames)
#   y1, y2, x1, x2 = _detect_black_bars_from_video(frames, blackbar_threshold=blackbar_threshold,
#                                                   max_perc_to_trim=max_perc_to_trim)
  
  return frames
  print("Right before returning frames")
  return frames[:, y1:y2, x1:x2]

def _detect_black_bars_from_video(frames, blackbar_threshold=16, max_perc_to_trim=.2):
    """
    :param frames: [num_frames, height, width, 3]
    :param blackbar_threshold: Pixels must be this intense for us to not trim
    :param max_perc_to_prim: Will trim 20% by default of the image at most in each dimension
    :return:
    """
    # Detect black bars
    has_content = frames.max(axis=(0, -1)) >= blackbar_threshold
    h, w = has_content.shape

    y_frames = np.where(has_content.any(1))[0]
    if y_frames.size == 0:
        print("Oh no, there are no valid yframes")
        y_frames = [h // 2]

    y1 = min(y_frames[0], int(h * max_perc_to_trim))
    y2 = max(y_frames[-1] + 1, int(h * (1 - max_perc_to_trim)))

    x_frames = np.where(has_content.any(0))[0]
    if x_frames.size == 0:
        print("Oh no, there are no valid xframes")
        x_frames = [w // 2]
    x1 = min(x_frames[0], int(w * max_perc_to_trim))
    x2 = max(x_frames[-1] + 1, int(w * (1 - max_perc_to_trim)))
    return y1, y2, x1, x2

def kas_extract_single_frame_from_video(video_file, timestamp_sec, video_fps):
    start_time = time.monotonic()
    frame_number = int(timestamp_sec * video_fps)
    frame = iio.imread(
        f"{video_file}",
        index=frame_number,
        plugin="pyav",
    )
    print(f"⏰ 1 frame extracted runtime: {(time.monotonic() - start_time):.3f} seconds")
    return frame
    
def DEPRICATED_extract_single_frame_from_video(video_file, t):
    ## DEPRICATED -- prefer kas_extract_single_frame_from_video
    ## My version might be faster than  ffmpeg via skvideo interface
    # IDK, I'm just sick of ffmpeg trouble. This av seems more reliable for standard-complexity jobs. 
    # Reserve FFMPEG usage for really difficult/niche problems. 
  """
  Reads the video, seeks to the given second option
  :param video_file: input video file
  :param t: where 2 seek to
  :return: the frame at that timestep.
  """
  print("In extract SINGLE FRAME frame from video")
  start_time = time.monotonic()
  
  timecode = '{:.3f}'.format(t)
  print("timecode", timecode)
  # https://ffmpeg.org/ffmpeg-utils.html#time-duration-syntax. Time in seconds (decimal-notation) by default.
  input_dict = {'-ss': timecode, '-threads': '1'}
  
  print("Before skvideo reader")
  reader = skvideo.io.FFmpegReader(video_file,
                                    inputdict=input_dict,
                                    # outputdict={'-r': '1', '-q:v': '2', '-pix_fmt': 'rgb24', '-frames:v': '1'},
                                    outputdict={'-q:v': '2', '-pix_fmt': 'rgb24', '-frames:v': '1'},
                                    verbosity=0,
                                    )
  print("After skvideo reader")
  try:
      frame = next(iter(reader.nextFrame()))
  except StopIteration:
      frame = None
  print(f"⏰ Runtime: {(time.monotonic() - start_time):.3f} seconds")
  print("right before return in SINGLE frame")
  return frame